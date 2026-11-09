"""Closed neural-worker boundaries with fake artifacts and fake model calls."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from knee_app import neural_probe_worker as worker


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle_fixture(tmp_path):
    checkout, root, scratch = tmp_path/'checkout', tmp_path/'bundle', tmp_path/'scratch'/'job'
    checkout.mkdir()
    root.mkdir()
    scratch.mkdir(parents=True)
    sources, recipes = {}, {}
    for names, target in ((worker.MINIMUM_SOURCES, sources), (worker.MINIMUM_RECIPES, recipes)):
        for name in names:
            path = checkout/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# Synthetic source only\n' if name.endswith('.py') else b'{}')
            target[name] = checksum(path)
    external = []
    for name in worker.EXTERNAL_NAMES:
        path = root/'external'/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'fake_external_not_a_model')
        external.append({'path':path.relative_to(root).as_posix(),'name':name,
                         'bytes':path.stat().st_size,'sha256':checksum(path)})
    jobs = []
    for slot, (model, stage) in worker.SLOTS.items():
        for fold in range(5):
            weight = root/'weights'/slot/f'fold_{fold}.safetensors'
            state = root/'states'/f'fold_{fold}.json'
            weight_payload = b'fake_trained_not_a_model'
            state_payload = json.dumps({'scenario':'clinical_main','validation_fold':fold}).encode()
            # Only one job has actually been acquired. Other nine may be absent.
            if slot == 'radiographic' and fold == 0:
                weight.parent.mkdir(parents=True)
                weight.write_bytes(weight_payload)
                state.parent.mkdir(exist_ok=True)
                state.write_bytes(state_payload)
            request = {'model_id':model,'stage':stage,'image_encoder_id':'image_skelex',
                'clinical_scenario':'clinical_main','validation_fold':fold,'seed':2026,
                'operation':'development_fold','image_micro_batch_size':4}
            jobs.append({'slot':slot,'fold':fold,'request':request,'source_record_sha256':'b'*64,
                'weights':{'path':weight.relative_to(root).as_posix(),
                           'sha256':hashlib.sha256(weight_payload).hexdigest(),'bytes':len(weight_payload)},
                'clinical_state':{'path':state.relative_to(root).as_posix(),
                                  'sha256':hashlib.sha256(state_payload).hexdigest()}})
    manifest = {'format':worker.FORMAT,'purpose':worker.PURPOSE,'clinical_release_ready':False,
        'runtime':dict(worker.RUNTIME),'scientific_sources':sources,'recipe_files':recipes,
        'external':external,'jobs':jobs}
    payload = {'operation':'preflight','checkout':str(checkout),'bundle':str(root),
               'slot':'radiographic','fold':0,'scratch':str(scratch)}
    def save():
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest),encoding='utf-8')
        payload['manifestSha256'] = checksum(path)
    save()
    return root, checkout, scratch, manifest, payload, save


def test_authenticates_selected_job_without_opening_other_missing_weights(bundle_fixture):
    root, _, scratch, _, payload, _ = bundle_fixture
    bundle = worker.load_bundle(payload)
    assert bundle.job.slot == 'radiographic' and bundle.job.fold == 0
    assert bundle.scratch == scratch
    assert len(list((root/'weights').rglob('*.safetensors'))) == 1
    # Streaming verification does not allow materializing a checkpoint as JSON/source bytes.
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_ARTIFACT_MISMATCH'):
        bundle.job.weights.read()


@pytest.mark.parametrize('change', [
    {'patient':{'age':62}}, {'operation':'train'}, {'operation':[]},
    {'slot':'best_model'}, {'slot':[]}, {'fold':True}, {'fold':5},
    {'manifestSha256':'invalid'}, {'threshold':0.5},
])
def test_protocol_rejects_patient_options_training_or_ambiguous_job(bundle_fixture,change):
    payload = {**bundle_fixture[4],**change}
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_REQUEST_INVALID'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('change',['clinical','runtime','extra','duplicate','missing','wrong_model',
                                    'wrong_stage','wrong_seed','wrong_scenario','bool_batch'])
def test_manifest_cannot_promote_release_or_change_registered_job_matrix(bundle_fixture,change):
    _, _, _, manifest, payload, save = bundle_fixture
    if change == 'clinical': manifest['clinical_release_ready'] = True
    elif change == 'runtime': manifest['runtime']['torch'] = 'unverified'
    elif change == 'extra': manifest['calibrator'] = 'not_permitted'
    elif change == 'duplicate': manifest['jobs'][-1] = deepcopy(manifest['jobs'][-2])
    elif change == 'missing': manifest['jobs'].pop()
    else:
        key, value = {'wrong_model':('model_id','image_vit_b16'),
            'wrong_stage':('stage','stage_1'),'wrong_seed':('seed',42),
            'wrong_scenario':('clinical_scenario','clinical_plus_baseline_kl'),
            'bool_batch':('image_micro_batch_size',True)}[change]
        manifest['jobs'][0]['request'][key] = value
    save()
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_BUNDLE_INVALID'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('part',['manifest','source','recipe','external','weights','state'])
def test_rejects_any_selected_input_hash_change(bundle_fixture,part):
    root, checkout, _, manifest, payload, _ = bundle_fixture
    paths = {'manifest':root/'manifest.json',
        'source':checkout/next(iter(manifest['scientific_sources'])),
        'recipe':checkout/next(iter(manifest['recipe_files'])),
        'external':root/manifest['external'][0]['path'],
        'weights':root/manifest['jobs'][0]['weights']['path'],
        'state':root/manifest['jobs'][0]['clinical_state']['path']}
    path = paths[part]
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_(ARTIFACT|SOURCE)_MISMATCH'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('part',['weights','external','state'])
def test_missing_target_assets_fail_before_loading_any_runtime(bundle_fixture,part,monkeypatch):
    root, _, _, manifest, payload, _ = bundle_fixture
    path = (root/manifest['external'][0]['path'] if part == 'external'
            else root/manifest['jobs'][0]['weights' if part == 'weights' else 'clinical_state']['path'])
    path.unlink()
    monkeypatch.setattr(worker,'scientific_runtime',lambda _:pytest.fail('Missing assets must not load a model'))
    with pytest.raises(worker.WorkerError,match='ASSET_MISSING'):
        worker.run(payload)


@pytest.mark.parametrize('name',['../outside.safetensors','/outside.safetensors',
    'C:/outside.safetensors','folder\\outside.safetensors','a/../outside.safetensors'])
def test_paths_cannot_escape_the_private_bundle(bundle_fixture,name):
    _, _, _, manifest, payload, save = bundle_fixture
    manifest['jobs'][0]['weights']['path'] = name
    save()
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_BUNDLE_INVALID'):
        worker.load_bundle(payload)


def test_scratch_must_be_empty_disjoint_and_not_a_read_only_input(bundle_fixture):
    root, checkout, scratch, _, payload, _ = bundle_fixture
    for forbidden in (root, root.parent, checkout):
        with pytest.raises(worker.WorkerError):
            worker.load_bundle({**payload,'scratch':str(forbidden)})
    (scratch/'existing').write_text('must_not_be_touched')
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_REQUEST_INVALID'):
        worker.load_bundle(payload)
    assert (scratch/'existing').read_text() == 'must_not_be_touched'


def test_scratch_and_artifact_ancestors_cannot_be_links(bundle_fixture,tmp_path):
    root, _, scratch, _, payload, _ = bundle_fixture
    link = root.parent/'linked_scratch'
    junction = False
    try:
        link.symlink_to(scratch,target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            pytest.skip('Directory links unavailable')
        result = subprocess.run(['cmd.exe','/c','mklink','/J',str(link),str(scratch)],
                                capture_output=True,text=True,timeout=10)
        assert result.returncode == 0 and link.is_junction(),result.stderr
        junction = True
    try:
        with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_BUNDLE_INVALID'):
            worker.load_bundle({**payload,'scratch':str(link)})
    finally:
        if junction:
            assert link.parent == root.parent and link.resolve() == scratch
            link.rmdir()


def fake_runtime(bundle,monkeypatch,operation,*,mutate=False):
    calls = []
    np = SimpleNamespace(float32='float32',empty=lambda shape,dtype:('empty',shape,dtype))
    def release(checkout):
        assert checkout == bundle[1]
        specs = [{'name':p['name'],'sha256':p['sha256'],'bytes':p['bytes']}
                 for p in bundle[3]['external']]
        return None,{'models':{'image_skelex':{'files':specs}}},'pinned-plan','pinned-registry'
    def validate(request,plan,registry):
        calls.append(('validate_original_request',request,plan,registry))
    def transform(rows,state):
        assert rows == list(worker.FIXTURES) and state['validation_fold'] == 0
        calls.append(('transform_original',len(rows)))
        return ('frozen-clinical',2,3)
    def inference(job,clinical,images,plan,catalog,external,device):
        assert device == 'cpu' and plan == 'pinned-plan'
        assert job['receipt']['request'] == bundle[3]['jobs'][0]['request']
        assert job['weight_sha256'] == bundle[3]['jobs'][0]['weights']['sha256']
        assert len(external) == 3
        if operation == 'preflight':
            assert clinical == ('empty',(0,3),'float32') and images == []
        else:
            assert clinical == ('frozen-clinical',2,3) and images == ['closed-gray-a','closed-gray-b']
        calls.append(('original_neural_logits',device))
        if mutate:
            job['weight'].write_bytes(b'changed-after-load')
        return [] if operation == 'preflight' else [0.25,-0.5]
    monkeypatch.setattr(worker,'scientific_runtime',lambda _:(np,release,validate,lambda _:None,inference,transform))
    monkeypatch.setattr(worker,'fixed_images',lambda _:['closed-gray-a','closed-gray-b'])
    return calls


@pytest.mark.parametrize('operation',['preflight','diagnostic'])
def test_dispatch_reuses_original_functions_with_cpu_and_closed_fixtures(bundle_fixture,monkeypatch,operation):
    payload = {**bundle_fixture[4],'operation':operation}
    calls = fake_runtime(bundle_fixture,monkeypatch,operation)
    result = worker.run(payload)
    assert result['scope'] == worker.PURPOSE and result['clinicalReleaseReady'] is False
    assert result['weightsVerified'] is True and result['operation'] == operation
    assert result['runtime'] == worker.RUNTIME and result['elapsedSeconds'] >= 0
    assert result['logits'] == ([] if operation == 'preflight' else [0.25,-0.5])
    assert calls[-1] == ('original_neural_logits','cpu')
    assert sum(call[0] == 'transform_original' for call in calls) == (operation == 'diagnostic')


def test_post_call_integrity_still_rejects_a_changed_checkpoint(bundle_fixture,monkeypatch):
    fake_runtime(bundle_fixture,monkeypatch,'preflight',mutate=True)
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_ARTIFACT_MISMATCH'):
        worker.run(bundle_fixture[4])


def test_audit_policy_allows_only_scratch_mutations_and_exact_reads(tmp_path):
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    input_ = tmp_path/'allowed.json'
    input_.write_bytes(b'{}')
    pin = worker.Pin(input_,checksum(input_),100)
    policy = worker.ScratchPolicy([pin],[tmp_path/'python'],scratch)
    policy('open',(str(input_),'r',os.O_RDONLY))
    owned = scratch/'owned'
    policy('os.mkdir',(str(owned),0o700,-1))
    policy('open',(str(owned/'file'),'w',os.O_CREAT))
    policy('os.remove',(str(owned/'file'),-1))
    policy('os.rmdir',(str(owned),-1))
    for event,args in [('open',(str(input_),'w',os.O_WRONLY)),
        ('open',(str(tmp_path/'cohort.csv'),'r',os.O_RDONLY)),
        ('socket.connect',(None,('example.invalid',443))),('subprocess.Popen',()),
        ('os.rmdir',(str(scratch),-1)),
        ('os.rename',(str(owned),str(tmp_path/'outside'),-1,-1)),
        ('shutil.rmtree',(str(tmp_path),None))]:
        with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_SAFETY_VIOLATION'):
            policy(event,args)


def test_real_child_guard_allows_temporary_cleanup_but_denies_outside_write_and_network(tmp_path):
    scratch = tmp_path/'private-scratch'
    scratch.mkdir()
    outside = tmp_path/'must-not-exist'
    code = '''
import json,socket,sys,tempfile
from pathlib import Path
from knee_app.neural_probe_worker import ScratchPolicy,WorkerError
scratch=Path(sys.argv[1]); tempfile.tempdir=str(scratch)
sys.addaudithook(ScratchPolicy([], {sys.prefix,sys.base_prefix},scratch))
with tempfile.TemporaryDirectory(prefix='original_helper_',dir=scratch) as temporary:
 p=Path(temporary)/'copy';p.write_bytes(b'fictional');assert p.read_bytes()==b'fictional'
out=['scratch_created_and_cleaned']
for operation in (lambda:Path(sys.argv[2]).write_text('forbidden'),lambda:socket.socket()):
 try:operation()
 except WorkerError as error:out.append(error.code)
print(json.dumps(out))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(worker.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable,'-B','-c',code,str(scratch),str(outside)],
                            env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout) == ['scratch_created_and_cleaned','NEURAL_PROBE_SAFETY_VIOLATION','NEURAL_PROBE_SAFETY_VIOLATION']
    assert not outside.exists() and list(scratch.iterdir()) == []


def test_local_hostname_metadata_is_allowed_but_resolution_and_connections_are_not(tmp_path):
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    policy = worker.ScratchPolicy([],[],scratch)
    policy('socket.gethostname',())
    for event,args in [('socket.getaddrinfo',('example.invalid',443,0,0,0)),
                       ('socket.connect',(None,('127.0.0.1',443))),
                       ('socket.__new__',(None,2,1,0))]:
        with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_SAFETY_VIOLATION'):
            policy(event,args)


def test_trusted_implementation_reads_are_exact_and_changes_are_detected(tmp_path,monkeypatch):
    own, common = tmp_path/'neural_probe_worker.py',tmp_path/'predictor_worker.py'
    own.write_text('# Fake trusted implementation A\n',encoding='utf-8')
    common.write_text('# Fake trusted implementation B\n',encoding='utf-8')
    monkeypatch.setattr(worker,'__file__',str(own))
    monkeypatch.setattr(worker.common,'__file__',str(common))
    pins = worker.trusted_implementation_pins()
    assert {pin.path for pin in pins} == {own,common}
    policy = worker.ScratchPolicy(pins,[],tmp_path/'scratch')
    policy('open',(str(own),'r',os.O_RDONLY))
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_SAFETY_VIOLATION'):
        policy('open',(str(tmp_path/'other_app_file.py'),'r',os.O_RDONLY))
    own.write_text('# Modified after import\n',encoding='utf-8')
    with pytest.raises(worker.WorkerError,match='NEURAL_PROBE_SOURCE_MISMATCH'):
        pins[0].verify()


@pytest.mark.parametrize('raw',[b'{"operation":"preflight","operation":"diagnostic"}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"private":"do_not_echo"}',b'x'*(worker.MAX_REQUEST+1)])
def test_cli_errors_are_bounded_and_never_echo_private_input(raw):
    result = subprocess.run([sys.executable,'-B',worker.__file__],input=raw,capture_output=True,timeout=10)
    assert result.returncode == 1 and result.stderr == b''
    assert len(result.stdout) <= worker.MAX_RESPONSE
    assert json.loads(result.stdout) == {'ok':False,'error':{'code':'NEURAL_PROBE_REQUEST_INVALID'}}
    assert b'do_not_echo' not in result.stdout
