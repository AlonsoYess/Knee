"""Application lifecycle with invented pixels, geometry and decisions (not clinical validation)."""
from contextlib import closing
import hashlib
import json
import sqlite3
import uuid

import pytest

from conftest import ApiHarness, COOKIE, SECOND_EMAIL
from fake_imaging import SyntheticProcessor
from synthetic_dicoms import dicom_bytes, source_reference, qc_observation


@pytest.fixture
def imaging(tmp_path, contract, scientific_import_guard):
    processor = SyntheticProcessor()
    api = ApiHarness(tmp_path, contract, processor=processor)
    client = api.client()
    api.login(client)
    api.processor = processor
    api.doctor = client
    try:
        yield api
    finally:
        api.close()


def request_headers(api, version, key=None, client=None):
    return api.headers(client or api.doctor, **{'If-Match': version, 'Idempotency-Key': key or uuid.uuid4().hex})


def upload(api, case=None, data=None, key=None, version=None, client=None):
    client = client or api.doctor
    case = case or api.create(client)
    identifier = case.json()['id']
    version = version or client.get(f'/api/v1/evaluations/{identifier}').headers['etag']
    return client.post(f'/api/v1/evaluations/{identifier}/preparations',
        headers=request_headers(api, version, key, client),
        files={'file': ('../../PRIVATE_PATIENT_NAME.dcm', data if data is not None else dicom_bytes(), 'application/dicom')})


def prep_get(api, identifier, client=None):
    response = (client or api.doctor).get(f'/api/v1/preparations/{identifier}')
    api.check(response, 'get', '/preparations/{preparationId}', 200)
    return response


def seal(api, prepared, **overrides):
    body = {**source_reference(), 'preparationRevisionId': prepared.json()['preparationRevisionId'], **overrides}
    response = api.doctor.post(f"/api/v1/preparations/{prepared.json()['id']}/reference",
        headers=request_headers(api, prepared.headers['etag']), json=body)
    return response, body


def ready(api):
    case = api.create(api.doctor)
    prepared = upload(api, case)
    api.check(prepared, 'post', '/evaluations/{evaluationId}/preparations', 202)
    reference, _ = seal(api, prepared)
    api.check(reference, 'post', '/preparations/{preparationId}/reference', 201)
    return case, prep_get(api, prepared.json()['id'])


def review(api, prepared, **overrides):
    body = {'preparationRevisionId': prepared.json()['preparationRevisionId'], 'confirmedSide': 'right',
            'decision': 'accepted', 'confirmedCheckIds': [], 'reason': None, 'observation': qc_observation(), **overrides}
    response = api.doctor.post(f"/api/v1/preparations/{prepared.json()['id']}/reviews",
        headers=request_headers(api, prepared.headers['etag']), json=body)
    return response, body


def test_full_source_reference_crop_review_restore_and_prediction_stays_blocked(imaging):
    from knee_app.api import create_app
    api = imaging
    case = api.create(api.doctor)
    prepared = upload(api, case)
    source = api.check(prepared, 'post', '/evaluations/{evaluationId}/preparations', 202)
    assert source['status'] == 'awaiting_reference'
    assert source['nativeHalfShape'] == [241, 401]
    assert source['sourceReference'] is None and source['cropGeometry'] is None
    assert api.processor.calls == ['source']  # No localization before the independent reference.
    assert len(source['sourceContext']['views']) == 4
    old_view = source['sourceContext']['views']['half_a']
    image = api.doctor.get(old_view)
    assert image.status_code == 200 and image.content.startswith(b'\x89PNG\r\n\x1a\n')
    assert image.headers['content-type'] == 'image/png' and image.headers['x-content-type-options'] == 'nosniff'
    assert 'no-store' in image.headers['cache-control']
    assert all(value not in prepared.text for value in ('PRIVATE_PATIENT_NAME', 'SYNTHETIC^ONLY', str(api.path.parent)))
    sealed, _ = seal(api, prepared)
    api.check(sealed, 'post', '/preparations/{preparationId}/reference', 201)
    current = prep_get(api, source['id'])
    assert current.json()['status'] == 'ready'
    assert current.json()['cropGeometry'] == {'nativeHalfBox': [100, 70, 300, 170], 'previewShapes': {'crop_a': [100, 200], 'crop_b': [100, 200]}}
    assert api.processor.calls == ['source', 'reference', 'crop']
    assert api.doctor.get(old_view).status_code == 409
    accepted, body = review(api, current)
    result = api.check(accepted, 'post', '/preparations/{preparationId}/reviews', 201)
    assert result['decision'] == 'accepted' and result['assessment']['accepted'] is True
    assert result['observation'] == body['observation'] and result['reviewerId'] == api.user_ids[next(iter(api.user_ids))]
    restored_app = create_app(api.settings, clock=api.clock, processor=api.processor)
    restored = api.client(restored_app)
    api.login(restored)
    again = prep_get(api, source['id'], restored).json()
    assert again['sourceReference']['id'] == sealed.json()['id'] and again['activeReview'] == result
    assert restored.get(again['reviewContext']['views']['crop_b']).content.startswith(b'\x89PNG')
    evaluation = restored.get(f"/api/v1/evaluations/{case.json()['id']}").json()
    assert evaluation['reviewId'] == result['id']
    prediction = restored.post(f"/api/v1/evaluations/{case.json()['id']}/executions", headers=api.headers(restored), json={})
    assert prediction.status_code == 503 and prediction.json()['error']['code'] == 'ENGINE_NOT_AVAILABLE'


@pytest.mark.parametrize('change', [
    {'centerXY': [-1, 120]}, {'centerXY': [401, 120]}, {'centerXY': [200, 241]},
    {'centerXY': [True, 120]}, {'originalAndHalfInspected': False}, {'outcomeBlinded': False},
    {'note': ' ' * 40}, {'centerResolved': False, 'centerXY': [1, 2]}, {'confirmedSide': 'left'},
])
def test_invalid_reference_never_runs_crop(imaging, change):
    prepared = upload(imaging)
    response, _ = seal(imaging, prepared, **change)
    assert response.status_code in {409, 422}, response.text
    assert 'crop' not in imaging.processor.calls
    assert prep_get(imaging, prepared.json()['id']).json()['sourceReference'] is None


def test_unresolved_reference_is_immutable_and_cannot_accept_or_retry(imaging):
    prepared = upload(imaging)
    sealed, _ = seal(imaging, prepared, centerResolved=False, centerXY=None)
    assert sealed.status_code == 201
    current = prep_get(imaging, prepared.json()['id'])
    assert current.json()['status'] == 'awaiting_reference' and 'crop' not in imaging.processor.calls
    changed, _ = seal(imaging, current)
    assert changed.status_code == 409 and changed.json()['error']['code'] == 'REFERENCE_ALREADY_SEALED'
    assert review(imaging, current)[0].status_code == 409


def test_reference_idempotency_and_lost_response_do_not_reseal_or_recrop(imaging):
    prepared = upload(imaging)
    key = uuid.uuid4().hex
    body = {**source_reference(), 'preparationRevisionId': prepared.json()['preparationRevisionId']}
    url = f"/api/v1/preparations/{prepared.json()['id']}/reference"
    headers = request_headers(imaging, prepared.headers['etag'], key)
    first = imaging.doctor.post(url, headers=headers, json=body)
    second = imaging.doctor.post(url, headers=headers, json=body)
    assert first.status_code == 201 and second.status_code == 200 and first.json() == second.json()
    assert imaging.processor.calls.count('crop') == 1
    conflict = imaging.doctor.post(url, headers=headers, json={**body, 'centerXY': [201, 120]})
    assert conflict.status_code == 409 and conflict.json()['error']['code'] == 'IDEMPOTENCY_CONFLICT'


@pytest.mark.parametrize('observation', [None, {}, qc_observation(unclassified_mark_present=True), qc_observation(side_supported=False)])
def test_no_checkbox_or_incomplete_observation_can_bypass_frozen_acceptance(imaging, observation):
    _, current = ready(imaging)
    rejected, _ = review(imaging, current, observation=observation)
    assert rejected.status_code in {409, 422}
    assert prep_get(imaging, current.json()['id']).json()['activeReview'] is None


def test_administrative_rejection_requires_reason_and_is_never_scientific_acceptance(imaging):
    _, current = ready(imaging)
    missing, _ = review(imaging, current, decision='rejected', observation=None)
    assert missing.status_code == 422
    rejected, _ = review(imaging, current, decision='rejected', observation=None, reason='Imagen de demostración descartada por el revisor.')
    data = imaging.check(rejected, 'post', '/preparations/{preparationId}/reviews', 201)
    assert data['decision'] == 'rejected' and data['assessment'] is None and data['observation'] is None


def test_review_replay_is_stable_but_stale_new_decision_is_rejected(imaging):
    _, current = ready(imaging)
    key = uuid.uuid4().hex
    headers = request_headers(imaging, current.headers['etag'], key)
    body = {'preparationRevisionId': current.json()['preparationRevisionId'], 'confirmedSide': 'right', 'decision': 'accepted', 'confirmedCheckIds': [], 'reason': None, 'observation': qc_observation()}
    url = f"/api/v1/preparations/{current.json()['id']}/reviews"
    first = imaging.doctor.post(url, headers=headers, json=body)
    replay = imaging.doctor.post(url, headers=headers, json=body)
    assert first.status_code == 201 and replay.status_code == 200 and first.json() == replay.json()
    stale = imaging.doctor.post(url, headers=request_headers(imaging, current.headers['etag']), json=body)
    assert stale.status_code == 412
    assert imaging.processor.calls.count('assess') == 1


def test_clinical_edit_preserves_qc_side_edit_invalidates_and_replay_never_reactivates(imaging):
    api = imaging
    case = api.create(api.doctor)
    binary, key = dicom_bytes(), uuid.uuid4().hex
    prepared = upload(api, case, data=binary, key=key)
    seal(api, prepared)
    current = prep_get(api, prepared.json()['id'])
    accepted, _ = review(api, current)
    assert accepted.status_code == 201
    url = f"/api/v1/evaluations/{case.json()['id']}"
    before = api.doctor.get(url)
    changed = api.doctor.patch(url, headers=request_headers(api, before.headers['etag']), json={'clinical': {'age': 64}})
    assert changed.json()['preparationId'] == current.json()['id'] and changed.json()['reviewId'] == accepted.json()['id']
    assert prep_get(api, current.json()['id']).json()['activeReview']['id'] == accepted.json()['id']
    side = api.doctor.patch(url, headers=request_headers(api, changed.headers['etag']), json={'clinical': {'side': 'left'}})
    assert side.json()['preparationId'] is None and side.json()['reviewId'] is None
    replay = upload(api, case, data=binary, key=key)
    assert replay.status_code == 200 and replay.json()['isCurrent'] is False
    assert api.doctor.get(url).json()['preparationId'] is None
    assert review(api, prep_get(api, current.json()['id']))[0].status_code == 409
    back = api.doctor.patch(url, headers=request_headers(api, side.headers['etag']), json={'clinical': {'side': 'right'}})
    reselected = upload(api, case, data=binary, version=back.headers['etag'])
    assert reselected.status_code == 200 and reselected.json()['reused'] is True and reselected.json()['isCurrent'] is True
    assert reselected.json()['sourceReference']['id'] == current.json()['sourceReference']['id']
    assert api.processor.calls.count('source') == 1 and api.processor.calls.count('crop') == 1


def test_other_account_and_anonymous_cannot_fetch_original_or_crop_or_mutate(imaging):
    api = imaging
    case, current = ready(api)
    other = api.client()
    api.login(other, SECOND_EMAIL)
    anonymous = api.client()
    for client, status in [(other, 404), (anonymous, 401)]:
        assert client.get(f"/api/v1/preparations/{current.json()['id']}").status_code == status
        for url in [*current.json()['sourceContext']['views'].values(), *current.json()['reviewContext']['views'].values()]:
            response = client.get(url)
            assert response.status_code == status and 'image/' not in response.headers.get('content-type', '')
    assert upload(api, case, client=other, version='"fictional"').status_code == 404
    assert api.doctor.post(f"/api/v1/preparations/{current.json()['id']}/retry", json={}).status_code == 403
    assert api.doctor.get(f"/api/v1/preparations/{current.json()['id']}/views/dicom?revision={current.json()['preparationRevisionId']}").status_code == 404


def test_upload_replay_conflict_etag_and_invalid_file_are_safely_bounded(imaging):
    api = imaging
    case = api.create(api.doctor)
    binary, key = dicom_bytes(), uuid.uuid4().hex
    first = upload(api, case, data=binary, key=key)
    replay = upload(api, case, data=binary, key=key, version=case.headers['etag'])
    assert first.status_code == 202 and replay.status_code == 200 and replay.json()['id'] == first.json()['id']
    assert upload(api, case, data=binary+b'changed', key=key).status_code == 409
    assert upload(api, case, version=case.headers['etag']).status_code == 412
    invalid = upload(api, case, data=b'NOT-A-DICOM')
    assert invalid.status_code == 422 and 'PRIVATE_PATIENT_NAME' not in invalid.text
    assert prep_get(api, invalid.headers['location'].split('/')[-1]).json()['status'] == 'failed'
    assert not list(api.app.state.private_files.staging.iterdir())


@pytest.mark.parametrize('kind', ['file', 'chunked_body', 'multiple', 'empty', 'mime'])
def test_multipart_limits_enforce_actual_bytes_and_single_file(imaging, kind):
    from dataclasses import replace
    from knee_app.api import create_app
    api = imaging
    if kind in {'file', 'chunked_body'}:
        app = create_app(replace(api.settings, max_request_bytes=1500, max_dicom_bytes=1000), clock=api.clock, processor=api.processor)
        client = api.client(app)
        api.login(client)
    else:
        client = api.doctor
    case = api.create(client)
    url = f"/api/v1/evaluations/{case.json()['id']}/preparations"
    headers = request_headers(api, case.headers['etag'], client=client)
    if kind == 'chunked_body':
        headers['Content-Type'] = 'multipart/form-data; boundary=synth'
        body = b'--synth\r\nContent-Disposition: form-data; name="file"; filename="x.dcm"\r\nContent-Type: application/dicom\r\n\r\n' + b'x'*2000 + b'\r\n--synth--\r\n'
        response = client.post(url, headers=headers, content=iter([body[:1000], body[1000:]]))
    else:
        files = [('file', ('x.dcm', b'x'*1100 if kind == 'file' else (b'' if kind == 'empty' else b'fixture'), 'text/html' if kind == 'mime' else 'application/dicom'))]
        if kind == 'multiple': files.append(('file', ('y.dcm', b'fixture', 'application/dicom')))
        response = client.post(url, headers=headers, files=files)
    assert response.status_code == (413 if kind in {'file','chunked_body'} else 415 if kind == 'mime' else 422), response.text
    with closing(api.app.state.db.connect()) as connection:
        assert connection.execute('SELECT COUNT(*) FROM preparations').fetchone()[0] == 0


def test_failed_crop_retries_same_reference_without_replacing_original(imaging):
    api = imaging
    prepared = upload(api)
    api.processor.fail_crop = True
    failed, _ = seal(api, prepared)
    assert failed.status_code == 503
    current = prep_get(api, prepared.json()['id'])
    reference = current.json()['sourceReference']
    assert current.json()['status'] == 'failed' and reference is not None
    api.processor.fail_crop = False
    url = f"/api/v1/preparations/{current.json()['id']}/retry"
    headers = request_headers(api, current.headers['etag'])
    retried = api.doctor.post(url, headers=headers, json={})
    assert retried.status_code == 202 and retried.json()['status'] == 'ready'
    assert retried.json()['sourceReference'] == reference
    replay = api.doctor.post(url, headers=headers, json={})
    assert replay.status_code == 200 and api.processor.calls.count('crop') == 2
    assert api.processor.calls.count('source') == 1 and api.processor.calls.count('reference') == 1


def test_process_interruption_is_recorded_and_can_retry_without_data_loss(imaging):
    from knee_app.api import create_app
    api = imaging
    prepared = upload(api)
    with api.app.state.db.transaction() as connection:
        connection.execute("UPDATE preparations SET status='processing' WHERE id=?", (prepared.json()['id'],))
    restarted_app = create_app(api.settings, clock=api.clock, processor=api.processor)
    restarted = api.client(restarted_app)
    api.login(restarted)
    result = prep_get(api, prepared.json()['id'], restarted)
    assert result.json()['status'] == 'failed' and result.json()['error']['code'] == 'PROCESSING_INTERRUPTED'
    assert restarted.get(result.json()['sourceContext']['views']['half_a']).status_code == 200


def test_failed_commit_leaves_no_case_reference_to_missing_file_and_can_retry(imaging, monkeypatch):
    api = imaging
    case = api.create(api.doctor)
    original = api.app.state.preparations.record_files
    def fail_after_record(*args):
        original(*args)
        raise sqlite3.OperationalError('PRIVATE INTERNAL FAILURE')
    with monkeypatch.context() as local:
        local.setattr(api.app.state.preparations, 'record_files', fail_after_record)
        response = upload(api, case)
        assert response.status_code in {500,503} and 'PRIVATE INTERNAL FAILURE' not in response.text
    after = api.doctor.get(f"/api/v1/evaluations/{case.json()['id']}")
    assert after.json()['preparationId'] is None
    with closing(api.app.state.db.connect()) as connection:
        assert connection.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0] == 0
    assert upload(api, case).status_code == 202


def test_complete_backup_restore_verifies_every_file_and_revokes_sessions(imaging, tmp_path):
    from knee_app.api import create_app
    from knee_app.db import backup_database, backup_package, restore_package
    api = imaging
    case, current = ready(api)
    accepted, _ = review(api, current)
    assert accepted.status_code == 201
    package, restored_dir = tmp_path/'package', tmp_path/'restored'
    with pytest.raises(ValueError): backup_database(api.path, tmp_path/'incomplete.sqlite3')
    backup_package(api.path, package, api.settings.private_files_path)
    manifest = json.loads((package/'manifest.json').read_text())
    assert len(manifest['objects']) >= 10
    for item in manifest['objects']:
        data = (package/'private_objects'/item['key']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    restore_package(package, restored_dir)
    app = create_app(api.settings_for(restored_dir/'application.sqlite3'), clock=api.clock, processor=api.processor)
    restored = api.client(app)
    restored.cookies.set(COOKIE, api.doctor.cookies.get(COOKIE), path='/api/v1')
    assert restored.get('/api/v1/evaluations').status_code == 401
    api.login(restored)
    restored_preparation = prep_get(api, current.json()['id'], restored).json()
    assert restored_preparation['activeReview']['id'] == accepted.json()['id']
    assert restored.get(restored_preparation['reviewContext']['views']['crop_a']).status_code == 200
    victim = package/'private_objects'/manifest['objects'][0]['key']
    victim.write_bytes(b'corrupted synthetic fixture')
    with pytest.raises(ValueError): restore_package(package, tmp_path/'corrupt-restore')
    assert not (tmp_path/'corrupt-restore').exists()


def test_v1_upgrade_preserves_existing_basal_users_and_revisions(imaging):
    from knee_app.api import create_app
    from knee_app.db import BASE_COLUMNS, EXPECTED_COLUMNS
    api = imaging
    case = api.create(api.doctor)
    with closing(sqlite3.connect(api.path)) as connection:
        connection.execute('PRAGMA foreign_keys=OFF')
        for table in EXPECTED_COLUMNS.keys() - BASE_COLUMNS.keys():
            connection.execute(f'DROP TABLE {table}')
        connection.execute('ALTER TABLE evaluations DROP COLUMN preparation_id')
        connection.execute('ALTER TABLE evaluations DROP COLUMN review_id')
        connection.execute('PRAGMA user_version=1')
        connection.commit()
    restarted = api.client(create_app(api.settings, clock=api.clock, processor=api.processor))
    api.login(restarted)
    result = restarted.get(f"/api/v1/evaluations/{case.json()['id']}")
    assert result.json() == case.json() and result.headers['etag'] == case.headers['etag']
    assert api.revision_count(case.json()['id']) == 1
    with closing(api.app.state.db.connect()) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 2
