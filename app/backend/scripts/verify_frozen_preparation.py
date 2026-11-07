"""Optional integration check using ONLY generated pixels and the approved local preparer.

Requires private scientific assets already present. Never reads research images,
labels, manifests of subjects, or a reserved test. Does not train or predict progression.
"""
from pathlib import Path
import argparse
import json
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / 'tests'))

from fastapi.testclient import TestClient
from knee_app.api import create_app
from knee_app.config import Settings
from knee_app.db import create_user
from synthetic_dicoms import dicom_bytes, source_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(database_path=args.output/'application.sqlite3', secret_key=b'synthetic-integration-key-only-32',
        environment='test', allowed_origins=('http://testserver',), frozen_checkout=args.checkout,
        frozen_python=args.python, frozen_timeout_seconds=180)
    app = create_app(settings)
    create_user(app.state.db, email='synthetic@kneeoai.invalid', password='Synthetic-Only-2026!', display_name='Synthetic Verification')
    summary = {'syntheticOnly': True, 'progressionPredictionExecuted': False, 'trainingExecuted': False}
    start = time.monotonic()
    with TestClient(app) as client:
        bootstrap = client.get('/api/v1/session').json()
        response = client.post('/api/v1/session', headers={'Origin':'http://testserver','X-CSRF-Token':bootstrap['csrfToken']},
            json={'email':'synthetic@kneeoai.invalid','password':'Synthetic-Only-2026!'})
        assert response.status_code == 200
        headers = {'Origin':'http://testserver','X-CSRF-Token':response.json()['csrfToken']}
        case = client.post('/api/v1/evaluations', headers=headers, json={'clinical':{'age':62,'sex':'F','bmi':None,'side':'right'}})
        assert case.status_code == 201
        upload = client.post(f"/api/v1/evaluations/{case.json()['id']}/preparations",
            headers={**headers,'If-Match':case.headers['etag'],'Idempotency-Key':'synthetic-source-verification'},
            files={'file':('generated-only.dcm',dicom_bytes(),'application/dicom')})
        summary['uploadStatus'] = upload.status_code
        if upload.status_code != 202:
            summary['errorCode'] = upload.json()['error']['code']
        else:
            source = upload.json()
            summary['sourceStatus'] = source['status']
            summary['nativeHalfShape'] = source['nativeHalfShape']
            summary['sourcePngVerified'] = sum(client.get(url).content.startswith(b'\x89PNG') for url in source['sourceContext']['views'].values())
            reference = {**source_reference(), 'preparationRevisionId':source['preparationRevisionId']}
            seal = client.post(f"/api/v1/preparations/{source['id']}/reference",
                headers={**headers,'If-Match':upload.headers['etag'],'Idempotency-Key':'synthetic-reference-verification'}, json=reference)
            summary['referenceRequestStatus'] = seal.status_code
            if seal.status_code != 201:
                summary['cropErrorCode'] = seal.json()['error']['code']
            result = client.get(f"/api/v1/preparations/{source['id']}").json()
            summary['referencePreserved'] = result['sourceReference'] is not None
            summary['finalPreparationStatus'] = result['status']
            summary['cropGeometry'] = result['cropGeometry']
            summary['qualityReviewRecorded'] = result['activeReview'] is not None
            summary['cropPngVerified'] = sum(client.get(url).content.startswith(b'\x89PNG') for url in (result['reviewContext'] or {}).get('views',{}).values())
            assert summary['sourcePngVerified'] == 4 and summary['referencePreserved']
            summary['expectedCropOutcome'] = seal.status_code == 201 or (seal.status_code == 422 and summary.get('cropErrorCode') == 'CROP_ABSTAIN')
        engine = client.get('/api/v1/engine-status').json()
        assert engine['status'] == 'not_configured'
    summary['seconds'] = round(time.monotonic()-start,2)
    (args.output/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))
    return 0 if summary['uploadStatus']==202 and summary.get('referencePreserved') and summary.get('expectedCropOutcome') else 1


if __name__ == '__main__':
    raise SystemExit(main())
