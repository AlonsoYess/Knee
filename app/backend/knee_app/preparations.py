"""Private preparation orchestration. Scientific operations are injected, never reimplemented."""
from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
import json
import re
import threading

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .db import opaque, utc
from .errors import ApiError
from .models import SourceReferenceSubmission, QualityReviewSubmission, PreparationRetry
from .private_files import PrivateFiles, receive_dicom, png_shape, digest_file


def encoded(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))


def fingerprint(value):
    try:
        return hashlib.sha256(encoded(value).encode()).hexdigest()
    except (TypeError, ValueError):
        raise ApiError(422, 'VALIDATION_FAILED', 'Revisa el formato de los datos enviados.')


def request_key(request):
    value = request.headers.get('idempotency-key')
    if value is None:
        raise ApiError(428, 'IDEMPOTENCY_REQUIRED', 'Se necesita una clave para conservar esta solicitud sin duplicarla.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', value):
        raise ApiError(422, 'VALIDATION_FAILED', 'La clave de la solicitud no es válida.')
    return value


def preparation_etag(row):
    return f'"{row["id"]}-v{row["version"]}"'


def require_preparation_version(request, row):
    supplied = request.headers.get('if-match')
    if supplied is None:
        raise ApiError(428, 'VERSION_REQUIRED', 'Recupera la versión de la preparación antes de guardar.')
    if supplied != preparation_etag(row):
        raise ApiError(412, 'VERSION_CONFLICT', 'La preparación cambió. Recupera su estado actual.', headers={'ETag': preparation_etag(row)})


class PreparationService:
    def __init__(self, db, files, processor, now):
        self.db, self.files, self.processor, self.now = db, files, processor, now
        self.upload_gate = threading.BoundedSemaphore(1)
        self.worker_gate = threading.RLock()

    def available(self):
        try:
            return self.processor is not None and self.processor.is_available()
        except Exception:
            return False

    def pipeline_version(self):
        return getattr(self.processor, 'pipeline_version', None)

    def qc_version(self):
        return getattr(self.processor, 'qc_policy_version', None)

    def case(self, connection, evaluation_id, owner):
        row = connection.execute('SELECT * FROM evaluations WHERE id=? AND owner_id=?', (evaluation_id, owner)).fetchone()
        if row is None:
            raise ApiError(404, 'RESOURCE_NOT_FOUND', 'No se encontró el recurso solicitado.')
        return row

    def get(self, connection, identifier, owner):
        row = connection.execute('SELECT p.* FROM preparations p JOIN evaluations e ON e.id=p.evaluation_id WHERE p.id=? AND e.owner_id=?', (identifier, owner)).fetchone()
        if row is None:
            raise ApiError(404, 'RESOURCE_NOT_FOUND', 'No se encontró el recurso solicitado.')
        return row

    def current(self, connection, row, owner):
        case = self.case(connection, row['evaluation_id'], owner)
        return case['preparation_id'] == row['id'] and case['side'] == row['side']

    def require_current(self, connection, row, owner):
        if not self.current(connection, row, owner):
            raise ApiError(409, 'PREPARATION_NOT_CURRENT', 'Esta preparación es histórica y no corresponde a la imagen y lado activos.')

    def prior(self, connection, owner, evaluation_id, operation, key, body_hash):
        row = connection.execute('SELECT * FROM preparation_requests WHERE owner_id=? AND evaluation_id=? AND operation=? AND request_key=?', (owner, evaluation_id, operation, key)).fetchone()
        if row is not None and not hmac.compare_digest(row['body_hash'], body_hash):
            raise ApiError(409, 'IDEMPOTENCY_CONFLICT', 'Esta clave ya se utilizó con otros datos.')
        return row

    def remember(self, connection, owner, evaluation_id, operation, key, body_hash, preparation_id, resource_id=None):
        connection.execute('INSERT INTO preparation_requests(owner_id,evaluation_id,operation,request_key,body_hash,preparation_id,resource_id,created_at) VALUES(?,?,?,?,?,?,?,?)', (owner, evaluation_id, operation, key, body_hash, preparation_id, resource_id, utc(self.now())))

    def record_files(self, connection, preparation_id, revision_id, directory_key, roles):
        for item in self.files.inventory(directory_key, roles):
            connection.execute('INSERT INTO artifacts(id,preparation_id,revision_id,role,object_key,sha256,byte_length) VALUES(?,?,?,?,?,?,?)', (opaque('art'), preparation_id, revision_id, item['role'], item['object_key'], item['sha256'], item['byte_length']))

    def snapshot(self, connection, identifier):
        row=connection.execute('SELECT * FROM preparations WHERE id=?',(identifier,)).fetchone()
        connection.execute('INSERT INTO preparation_revisions(id,preparation_id,source_dir,candidate_dir,source_json,candidate_json,reference_id,pipeline_version,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (row['revision_id'],identifier,row['source_dir'],row['candidate_dir'],row['source_json'],row['candidate_json'],row['reference_id'],row['pipeline_version'],row['status'],utc(self.now())))

    def reference_dto(self, connection, reference_id):
        if not reference_id:
            return None
        row = connection.execute('SELECT * FROM source_references WHERE id=?', (reference_id,)).fetchone()
        payload = json.loads(row['payload_json'])
        return {'id': row['id'], 'preparationId': row['preparation_id'], **payload, 'reviewerId': row['reviewer_id'], 'createdAt': row['created_at']}

    def review_dto(self, connection, review_id):
        if not review_id:
            return None
        row = connection.execute('SELECT * FROM quality_reviews WHERE id=?', (review_id,)).fetchone()
        return json.loads(row['payload_json']) if row else None

    def effective_review_id(self, connection, case):
        if not case['preparation_id'] or not case['review_id']:
            return None
        prepared=connection.execute('SELECT * FROM preparations WHERE id=? AND evaluation_id=?',(case['preparation_id'],case['id'])).fetchone()
        if prepared is None or prepared['side']!=case['side'] or prepared['status']!='ready' or prepared['pipeline_version']!=self.pipeline_version() or prepared['active_review_id']!=case['review_id']:
            return None
        review=self.review_dto(connection,case['review_id'])
        return case['review_id'] if review and review.get('qcPolicyVersion')==self.qc_version() and review.get('preparationRevisionId')==prepared['revision_id'] else None

    def dto(self, row, owner, *, reused=False):
        with closing(self.db.connect()) as connection:
            case = self.case(connection, row['evaluation_id'], owner)
            current = case['preparation_id'] == row['id'] and case['side'] == row['side']
            source = json.loads(row['source_json']) if row['source_json'] else None
            candidate = json.loads(row['candidate_json']) if row['candidate_json'] else None
            reference = self.reference_dto(connection, row['reference_id'])
            review = self.review_dto(connection, row['active_review_id']) if current and case['review_id'] == row['active_review_id'] else None
            if review and (review.get('qcPolicyVersion') != self.qc_version() or row['pipeline_version'] != self.pipeline_version()):
                review = None
            roles = {item['role']: item for item in connection.execute("SELECT * FROM artifacts WHERE preparation_id=? AND role!='internal' ORDER BY rowid", (row['id'],))}
        def url(role):
            return f'/api/v1/preparations/{row["id"]}/views/{role}?revision={row["revision_id"]}'
        source_context = None
        if source is not None:
            source_context = {'views': {role: url(role) for role in ('full_a','full_b','half_a','half_b') if role in roles}, 'fullNativeShape': source['fullNativeShape']}
        geometry = None
        if candidate and candidate.get('box') and all(role in roles for role in ('crop_a','crop_b')):
            x0, x1, y0, y1 = candidate['box']
            geometry = {'nativeHalfBox': [x0,y0,x1,y1], 'previewShapes': {role: png_shape(self.files.path(roles[role]['object_key'])) for role in ('crop_a','crop_b')}}
        context = None
        if row['status'] == 'ready':
            blockers = []
            if not current:
                blockers.append('PREPARATION_NOT_CURRENT')
            if not self.available():
                blockers.append('PROCESSOR_UNAVAILABLE')
            if row['pipeline_version'] != self.pipeline_version():
                blockers.append('PIPELINE_VERSION_MISMATCH')
            context = {'qcPolicyVersion': self.qc_version() or row['pipeline_version'], 'canReview': not blockers,
                       'blockerCodes': blockers, 'checks': [], 'originalPreviewUrl': url('half_a'), 'cropPreviewUrl': url('crop_a'),
                       'views': {role: url(role) for role in ('crop_a','crop_b') if role in roles}}
        error = None
        if row['error_json']:
            saved = json.loads(row['error_json'])
            error = {'code': saved['code'], 'message': saved['message'], 'requestId': row['id'], 'fields': [], 'retryable': saved.get('retryable',True)}
        return {'id': row['id'], 'evaluationId': row['evaluation_id'], 'sourceEvaluationRevisionId': row['source_revision_id'],
                'preparationRevisionId': row['revision_id'], 'image': {'imageId': row['image_id'], 'imageVersion': row['image_version']},
                'side': row['side'], 'status': row['status'], 'pipelineVersion': row['pipeline_version'], 'reviewContext': context,
                'error': error, 'createdAt': row['created_at'], 'updatedAt': row['updated_at'], 'activeReview': review,
                'sourceContext': source_context, 'nativeHalfShape': source['nativeHalfShape'] if source else None,
                'sourceReference': reference, 'cropGeometry': geometry, 'isCurrent': current, 'reused': reused}

    def failed(self, identifier, code='PROCESSING_FAILED', *, retryable=True):
        if not isinstance(code, str) or not re.fullmatch(r'[A-Z0-9_]{1,80}', code):
            code = 'PROCESSING_FAILED'
        message='El localizador se abstuvo de producir un candidato evaluable para esta fuente. No se aprobó el recorte.' if code=='CROP_ABSTAIN' else 'No se pudo completar esta preparación. Conservamos la fuente y la referencia guardada para revisar o reintentar.'
        error = {'code': code, 'message': message,'retryable':retryable}
        with self.db.transaction() as connection:
            connection.execute("UPDATE preparations SET status='failed',error_json=?,version=version+1,active_review_id=NULL,updated_at=? WHERE id=?", (encoded(error), utc(self.now()), identifier))
            connection.execute('UPDATE evaluations SET review_id=NULL WHERE preparation_id=?', (identifier,))

    def worker_failure(self, identifier, error):
        code = getattr(error, 'code', 'PROCESSING_FAILED')
        status = 422 if code in {'INVALID_DICOM','UNSUPPORTED_DICOM','INVALID_REFERENCE','INVALID_OBSERVATION','IMAGE_REJECTED','SOURCE_REFERENCE_INVALID','QC_INVALID','CROP_ABSTAIN'} else 503
        self.failed(identifier,code,retryable=status==503)
        raise ApiError(status, code if isinstance(code,str) and re.fullmatch(r'[A-Z0-9_]{1,80}',code) else 'PROCESSING_FAILED', 'No se pudo completar la preparación con el procedimiento congelado. Revisa su estado antes de reintentar.', retryable=status == 503, headers={'Location': f'/api/v1/preparations/{identifier}'})

    def process_source(self, identifier):
        stage = self.files.stage()
        try:
            with self.worker_gate:
                if not self.available():
                    raise ApiError(503, 'PROCESSOR_UNAVAILABLE', 'El procedimiento de preparación no está configurado.')
                with closing(self.db.connect()) as connection:
                    row = connection.execute('SELECT * FROM preparations WHERE id=?', (identifier,)).fetchone()
                result = self.processor.source(self.files.path(row['source_key']), row['side'], stage)
                for key in ('fullNativeShape', 'nativeHalfShape'):
                    shape = result.get(key)
                    if not isinstance(shape,list) or len(shape)!=2 or any(type(x) is not int or x<=0 for x in shape):
                        raise ValueError('Invalid worker source shape')
                artifacts = result.get('artifacts', {})
                if set(artifacts) != {'full_a','full_b','half_a','half_b'}:
                    raise ValueError('Incomplete source views')
                for role, filename in artifacts.items():
                    if filename != f'{role}.png':
                        raise ValueError('Invalid source view name')
                    png_shape(stage / filename)
                directory = self.files.publish(stage)
                revision = opaque('prv')
                with self.db.transaction() as connection:
                    self.record_files(connection, identifier, revision, directory, {name:role for role,name in artifacts.items()})
                    connection.execute("UPDATE preparations SET source_dir=?,source_json=?,status='awaiting_reference',error_json=NULL,revision_id=?,version=version+1,pipeline_version=?,updated_at=? WHERE id=?", (directory,encoded(result),revision,self.pipeline_version(),utc(self.now()),identifier))
                    self.snapshot(connection,identifier)
        except Exception as error:
            self.files.discard(stage)
            self.worker_failure(identifier,error)

    def process_crop(self, identifier):
        stage = self.files.stage()
        try:
            with self.worker_gate:
                if not self.available():
                    raise ApiError(503,'PROCESSOR_UNAVAILABLE','El procedimiento congelado no está disponible.')
                with closing(self.db.connect()) as connection:
                    row = connection.execute('SELECT * FROM preparations WHERE id=?',(identifier,)).fetchone()
                    ref = self.reference_dto(connection,row['reference_id'])
                if row['pipeline_version'] != self.pipeline_version():
                    raise ApiError(503,'PROCESSOR_VERSION_MISMATCH','La fuente corresponde a otra versión del procedimiento.')
                if ref is None or not ref['centerResolved']:
                    raise ApiError(409,'REFERENCE_UNRESOLVED','La referencia debe estar resuelta antes de preparar el recorte.')
                result = self.processor.crop(self.files.path(row['source_dir']),stage)
                if result.get('status') != 'candidate':
                    raise ApiError(409,'CROP_ABSTAIN','El localizador no produjo un candidato evaluable.')
                box = result.get('box')
                if not isinstance(box,list) or len(box)!=4 or any(type(x)is not int for x in box) or box[1]<=box[0] or box[3]<=box[2]:
                    raise ValueError('Invalid native candidate geometry')
                artifacts = result.get('artifacts',{})
                if set(artifacts) != {'crop_a','crop_b'}:
                    raise ValueError('Incomplete candidate views')
                for role,filename in artifacts.items():
                    if filename != f'{role}.png':
                        raise ValueError('Invalid candidate view name')
                    png_shape(stage / filename)
                directory = self.files.publish(stage)
                revision = opaque('prv')
                with self.db.transaction() as connection:
                    self.record_files(connection,identifier,revision,directory,{name:role for role,name in artifacts.items()})
                    connection.execute("UPDATE preparations SET candidate_dir=?,candidate_json=?,status='ready',revision_id=?,version=version+1,error_json=NULL,updated_at=? WHERE id=?",(directory,encoded(result),revision,utc(self.now()),identifier))
                    self.snapshot(connection,identifier)
        except Exception as error:
            self.files.discard(stage)
            self.worker_failure(identifier,error)

    def activate(self, connection, case, preparation, owner, reused):
        review_id = None
        if preparation['status'] == 'ready' and preparation['pipeline_version'] == self.pipeline_version():
            saved = self.review_dto(connection, preparation['active_review_id'])
            if saved and saved.get('qcPolicyVersion') == self.qc_version():
                review_id = preparation['active_review_id']
        connection.execute('UPDATE evaluations SET preparation_id=?,review_id=?,version=version+1,updated_at=? WHERE id=?', (preparation['id'],review_id,utc(self.now()),case['id']))
        connection.execute('INSERT INTO preparation_activations(id,evaluation_id,preparation_id,reviewer_id,reused,created_at) VALUES(?,?,?,?,?,?)',(opaque('act'),case['id'],preparation['id'],owner,int(reused),utc(self.now())))

    def upload(self, request, owner, evaluation_id, stage, source_sha):
        key = request_key(request)
        try:
            with self.db.transaction() as connection:
                case = self.case(connection,evaluation_id,owner)
                prior = connection.execute("SELECT * FROM preparation_requests WHERE owner_id=? AND evaluation_id=? AND operation='upload' AND request_key=?",(owner,evaluation_id,key)).fetchone()
                if prior:
                    previous = self.get(connection,prior['preparation_id'],owner)
                    body_hash = fingerprint({'sha':source_sha,'side':previous['side']})
                    self.prior(connection,owner,evaluation_id,'upload',key,body_hash)
                    self.files.discard(stage)
                    return previous,200,True
                supplied = request.headers.get('if-match')
                expected = f'"{case["id"]}-v{case["version"]}"'
                if supplied is None:
                    raise ApiError(428,'VERSION_REQUIRED','Recupera el caso antes de cargar su imagen.')
                if supplied != expected:
                    raise ApiError(412,'VERSION_CONFLICT','El caso cambió durante la carga.',headers={'ETag':expected})
                if case['side'] not in {'right','left'}:
                    raise ApiError(422,'SIDE_REQUIRED','Guarda primero el lado del caso.')
                body_hash = fingerprint({'sha':source_sha,'side':case['side']})
                existing = connection.execute('SELECT * FROM preparations WHERE evaluation_id=? AND source_sha=? AND side=?',(evaluation_id,source_sha,case['side'])).fetchone()
                if existing:
                    self.remember(connection,owner,evaluation_id,'upload',key,body_hash,existing['id'])
                    self.activate(connection,case,existing,owner,True)
                    self.files.discard(stage)
                    return existing,200,True
                identifier,revision = opaque('prep'),opaque('prv')
                directory = self.files.publish(stage)
                source_key = f'{directory}/source.dcm'
                stamp = utc(self.now())
                connection.execute("INSERT INTO preparations(id,evaluation_id,source_revision_id,image_id,image_version,source_sha,side,source_key,status,revision_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(identifier,evaluation_id,case['revision_id'],opaque('img'),opaque('iv'),source_sha,case['side'],source_key,'processing',revision,stamp,stamp))
                self.record_files(connection,identifier,revision,directory,{'source.dcm':'dicom'})
                prepared = connection.execute('SELECT * FROM preparations WHERE id=?',(identifier,)).fetchone()
                self.remember(connection,owner,evaluation_id,'upload',key,body_hash,identifier)
                self.activate(connection,case,prepared,owner,False)
            self.process_source(identifier)
            with closing(self.db.connect()) as connection:
                return self.get(connection,identifier,owner),202,False
        except BaseException:
            self.files.discard(stage)
            raise

    def seal_reference(self, request, owner, identifier, payload):
        key = request_key(request)
        body = payload.model_dump()
        body_hash = fingerprint(body)
        operation = f'reference:{identifier}'
        with closing(self.db.connect()) as connection:
            row = self.get(connection,identifier,owner)
            prior = self.prior(connection,owner,row['evaluation_id'],operation,key,body_hash)
            if prior:
                return self.reference_dto(connection,prior['resource_id']),200
            self.require_current(connection,row,owner)
            require_preparation_version(request,row)
            if row['reference_id']:
                raise ApiError(409,'REFERENCE_ALREADY_SEALED','La referencia de esta fuente ya está sellada y no se puede editar.')
            if row['status'] != 'awaiting_reference' or not row['source_dir']:
                raise ApiError(409,'SOURCE_NOT_READY','Espera a que estén disponibles las vistas originales.')
            if payload.preparationRevisionId != row['revision_id'] or payload.confirmedSide != row['side']:
                raise ApiError(409,'REFERENCE_SOURCE_MISMATCH','La referencia no corresponde a esta revisión y lado.')
            if not payload.originalAndHalfInspected or not payload.outcomeBlinded or len(payload.note.strip()) < 35:
                raise ApiError(422,'SOURCE_REFERENCE_INVALID','Confirma la revisión original, el cegamiento y la nota de la referencia.')
            source = json.loads(row['source_json'])
            if payload.centerResolved:
                if payload.centerXY is None:
                    raise ApiError(422,'SOURCE_REFERENCE_INVALID','Indica el centro en coordenadas nativas de la mitad.')
                x,y = payload.centerXY
                height,width = source['nativeHalfShape']
                if not 0 <= x < width or not 0 <= y < height:
                    raise ApiError(422,'SOURCE_REFERENCE_INVALID','El centro debe estar dentro de la mitad nativa.')
            elif payload.centerXY is not None:
                raise ApiError(422,'SOURCE_REFERENCE_INVALID','Una referencia no resuelta debe conservar el centro ausente.')
        worker_ref = {'center_xy':payload.centerXY,'center_resolved':payload.centerResolved,'original_and_half_inspected':payload.originalAndHalfInspected,'outcome_blinded':payload.outcomeBlinded,'note':payload.note}
        try:
            with self.worker_gate:
                if not self.available():
                    raise ApiError(503,'PROCESSOR_UNAVAILABLE','El procedimiento congelado no está disponible.')
                normalized = self.processor.validate_reference(self.files.path(row['source_dir']),worker_ref)
        except Exception as error:
            if getattr(error,'code',None) == 'SOURCE_REFERENCE_INVALID':
                raise ApiError(422,'SOURCE_REFERENCE_INVALID','La referencia no cumple el procedimiento congelado.')
            self.worker_failure(identifier,error)
        with self.db.transaction() as connection:
            row = self.get(connection,identifier,owner)
            prior = self.prior(connection,owner,row['evaluation_id'],operation,key,body_hash)
            if prior:
                return self.reference_dto(connection,prior['resource_id']),200
            self.require_current(connection,row,owner)
            require_preparation_version(request,row)
            if row['reference_id']:
                raise ApiError(409,'REFERENCE_ALREADY_SEALED','La referencia ya está sellada.')
            reference_id = opaque('ref')
            connection.execute('INSERT INTO source_references(id,preparation_id,source_revision_id,payload_json,worker_json,reviewer_id,created_at) VALUES(?,?,?,?,?,?,?)',(reference_id,identifier,row['revision_id'],encoded(body),encoded(normalized),owner,utc(self.now())))
            connection.execute('UPDATE preparations SET reference_id=?,status=?,revision_id=?,version=version+1,error_json=NULL,updated_at=? WHERE id=?',(reference_id,'processing' if payload.centerResolved else 'awaiting_reference',opaque('prv'),utc(self.now()),identifier))
            self.snapshot(connection,identifier)
            self.remember(connection,owner,row['evaluation_id'],operation,key,body_hash,identifier,reference_id)
        if payload.centerResolved:
            self.process_crop(identifier)
        with closing(self.db.connect()) as connection:
            return self.reference_dto(connection,reference_id),201

    def retry(self, request, owner, identifier):
        key = request_key(request)
        operation = f'retry:{identifier}'
        with self.db.transaction() as connection:
            row = self.get(connection,identifier,owner)
            body_hash = fingerprint({'id':identifier,'reference':row['reference_id']})
            prior = self.prior(connection,owner,row['evaluation_id'],operation,key,body_hash)
            if prior:
                return row,200
            self.require_current(connection,row,owner)
            require_preparation_version(request,row)
            if row['status'] != 'failed':
                raise ApiError(409,'RETRY_NOT_AVAILABLE','Solo se reintenta una preparación fallida con su misma fuente y referencia.')
            reference = self.reference_dto(connection,row['reference_id'])
            if reference and not reference['centerResolved']:
                raise ApiError(409,'REFERENCE_UNRESOLVED','La referencia sellada no está resuelta y no permite calcular el recorte.')
            self.remember(connection,owner,row['evaluation_id'],operation,key,body_hash,identifier)
            connection.execute("UPDATE preparations SET status='processing',version=version+1,error_json=NULL,updated_at=? WHERE id=?",(utc(self.now()),identifier))
        if row['source_dir'] and reference:
            self.process_crop(identifier)
        else:
            self.process_source(identifier)
        with closing(self.db.connect()) as connection:
            return self.get(connection,identifier,owner),202

    def review(self, request, owner, identifier, payload):
        key = request_key(request)
        body = payload.model_dump()
        body_hash = fingerprint(body)
        operation = f'review:{identifier}'
        with closing(self.db.connect()) as connection:
            row = self.get(connection,identifier,owner)
            prior = self.prior(connection,owner,row['evaluation_id'],operation,key,body_hash)
            if prior:
                return self.review_dto(connection,prior['resource_id']),200
            self.require_current(connection,row,owner)
            require_preparation_version(request,row)
            if row['status']!='ready' or not row['reference_id'] or not row['candidate_dir']:
                raise ApiError(409,'PREPARATION_NOT_READY','Se necesita una preparación lista con referencia sellada.')
            if row['pipeline_version'] != self.pipeline_version():
                raise ApiError(409,'PROCESSOR_VERSION_MISMATCH','La preparación corresponde a otra versión del procedimiento.')
            if payload.preparationRevisionId!=row['revision_id'] or payload.confirmedSide!=row['side']:
                raise ApiError(409,'REVIEW_SOURCE_MISMATCH','La decisión no corresponde a la revisión y lado actuales.')
            if payload.confirmedCheckIds:
                raise ApiError(422,'UNKNOWN_CHECK','La observación completa sustituye a controles ficticios; no envíes identificadores inventados.')
            ref = connection.execute('SELECT worker_json FROM source_references WHERE id=?',(row['reference_id'],)).fetchone()
        assessment = None
        if payload.observation is not None:
            if 'knee_alias' in payload.observation:
                raise ApiError(422,'QC_INVALID','La identidad de la observación la asigna el servidor.')
            try:
                with self.worker_gate:
                    if not self.available():
                        raise ApiError(503,'PROCESSOR_UNAVAILABLE','El procedimiento congelado no está disponible.')
                    assessment = self.processor.assess(self.files.path(row['source_dir']),self.files.path(row['candidate_dir']),json.loads(ref['worker_json']),payload.observation)
                if not isinstance(assessment,dict) or type(assessment.get('accepted'))is not bool or type(assessment.get('geometryAdequate'))is not bool or not isinstance(assessment.get('reasons'),list) or any(not isinstance(item,str) for item in assessment['reasons']):
                    raise ValueError('Invalid frozen assessment')
            except Exception as error:
                if getattr(error,'code',None)=='QC_INVALID':
                    raise ApiError(422,'QC_INVALID','Completa todos los campos de la observación según el procedimiento congelado.')
                self.worker_failure(identifier,error)
        if payload.decision=='accepted' and (assessment is None or not assessment['accepted']):
            reasons = assessment.get('reasons',[]) if assessment else []
            fields = [{'path':'observation','code':str(code)[:128],'message':'Revisa este criterio de la observación congelada.'} for code in reasons if isinstance(code,str)]
            raise ApiError(409,'QC_NOT_ACCEPTABLE','La evaluación completa no permite aceptar este recorte.',fields=fields)
        with self.db.transaction() as connection:
            current = self.get(connection,identifier,owner)
            prior = self.prior(connection,owner,current['evaluation_id'],operation,key,body_hash)
            if prior:
                return self.review_dto(connection,prior['resource_id']),200
            self.require_current(connection,current,owner)
            require_preparation_version(request,current)
            if current['status']!='ready' or current['revision_id']!=payload.preparationRevisionId:
                raise ApiError(409,'PREPARATION_NOT_READY','La preparación cambió durante la evaluación.')
            review_id = opaque('qc')
            public_assessment={name:assessment[name] for name in ('accepted','reasons','geometryAdequate')} if assessment is not None else None
            saved = {'id':review_id,'evaluationId':row['evaluation_id'],'preparationId':identifier,'preparationRevisionId':row['revision_id'],
                     'image':{'imageId':row['image_id'],'imageVersion':row['image_version']},'side':row['side'],'qcPolicyVersion':self.qc_version(),
                     'decision':payload.decision,'confirmedCheckIds':payload.confirmedCheckIds,'reason':payload.reason,'reviewerId':owner,'createdAt':utc(self.now()),
                     'observation':payload.observation,'assessment':public_assessment}
            connection.execute('INSERT INTO quality_reviews(id,preparation_id,preparation_revision_id,payload_json,assessment_json,reviewer_id,created_at) VALUES(?,?,?,?,?,?,?)',(review_id,identifier,row['revision_id'],encoded(saved),encoded(assessment),owner,utc(self.now())))
            connection.execute('UPDATE preparations SET active_review_id=?,assessment_json=?,version=version+1,updated_at=? WHERE id=?',(review_id,encoded(assessment),utc(self.now()),identifier))
            connection.execute('UPDATE evaluations SET review_id=?,version=version+1,updated_at=? WHERE preparation_id=?',(review_id,utc(self.now()),identifier))
            self.remember(connection,owner,row['evaluation_id'],operation,key,body_hash,identifier,review_id)
        return saved,201


def install_preparation_routes(app, *, settings, db, processor, now, require_session, require_mutation, ownership, require_version):
    files = PrivateFiles(settings.private_files_path)
    service = PreparationService(db,files,processor,now)
    app.state.preparations,app.state.private_files = service,files

    @app.post('/api/v1/evaluations/{evaluationId}/preparations')
    async def upload(evaluationId:str,request:Request):
        session,_ = require_mutation(request)
        owner=session['user_id']
        key=request_key(request)
        with closing(db.connect()) as connection:
            case=ownership(connection,evaluationId,owner)
            prior=connection.execute("SELECT 1 FROM preparation_requests WHERE owner_id=? AND evaluation_id=? AND operation='upload' AND request_key=?",(owner,evaluationId,key)).fetchone()
            if not prior:
                require_version(request,case)
            if case['side'] is None and not prior:
                raise ApiError(422,'SIDE_REQUIRED','Guarda el lado del caso antes de cargar una imagen.')
        await run_in_threadpool(service.upload_gate.acquire)
        try:
            stage,sha,_=await receive_dicom(request,files,max_body=settings.max_request_bytes,max_file=settings.max_dicom_bytes)
            row,status,reused=await run_in_threadpool(service.upload,request,owner,evaluationId,stage,sha)
            return JSONResponse(service.dto(row,owner,reused=reused),status_code=status,headers={'ETag':preparation_etag(row),'Location':f'/api/v1/preparations/{row["id"]}'})
        finally:
            service.upload_gate.release()

    @app.get('/api/v1/preparations/{preparationId}')
    def get_preparation(preparationId:str,request:Request):
        session,_=require_session(request)
        with closing(db.connect()) as connection:
            row=service.get(connection,preparationId,session['user_id'])
        return JSONResponse(service.dto(row,session['user_id']),headers={'ETag':preparation_etag(row)})

    @app.get('/api/v1/preparations/{preparationId}/views/{view}')
    def get_view(preparationId:str,view:str,request:Request,revision:str):
        session,_=require_session(request)
        with closing(db.connect()) as connection:
            row=service.get(connection,preparationId,session['user_id'])
            if revision!=row['revision_id']:
                raise ApiError(409,'VIEW_NOT_CURRENT','Recupera la vista correspondiente a esta revisión.')
            role={'original':'half_a','crop':'crop_a'}.get(view,view)
            if role not in {'full_a','full_b','half_a','half_b','crop_a','crop_b'}:
                raise ApiError(404,'RESOURCE_NOT_FOUND','No se encontró el recurso solicitado.')
            artifact=connection.execute('SELECT * FROM artifacts WHERE preparation_id=? AND role=? ORDER BY rowid DESC LIMIT 1',(preparationId,role)).fetchone()
        if artifact is None:
            raise ApiError(409,'VIEW_NOT_READY','La vista solicitada todavía no está disponible.')
        path=files.path(artifact['object_key'])
        try:
            intact=path.is_file() and digest_file(path)==(artifact['sha256'],artifact['byte_length'])
        except OSError:
            intact=False
        if not intact:
            raise ApiError(503,'PRIVATE_FILE_UNAVAILABLE','No se pudo recuperar la vista privada.')
        return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Content-Disposition':'inline'})

    @app.post('/api/v1/preparations/{preparationId}/reference')
    def reference(preparationId:str,request:Request,payload:SourceReferenceSubmission):
        session,_=require_mutation(request)
        body,status=service.seal_reference(request,session['user_id'],preparationId,payload)
        return JSONResponse(body,status_code=status)

    @app.post('/api/v1/preparations/{preparationId}/retry')
    def retry(preparationId:str,request:Request,payload:PreparationRetry):
        session,_=require_mutation(request)
        row,status=service.retry(request,session['user_id'],preparationId)
        return JSONResponse(service.dto(row,session['user_id']),status_code=status,headers={'ETag':preparation_etag(row)})

    @app.post('/api/v1/preparations/{preparationId}/reviews')
    def review(preparationId:str,request:Request,payload:QualityReviewSubmission):
        session,_=require_mutation(request)
        body,status=service.review(request,session['user_id'],preparationId,payload)
        return JSONResponse(body,status_code=status)

    return service
