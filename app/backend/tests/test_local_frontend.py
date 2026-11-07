"""Public build copies must never ingest runtime or scientific data."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('local_frontend', Path(__file__).resolve().parents[1]/'scripts'/'prepare_local_frontend.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def public_build(tmp_path):
    dist = tmp_path/'dist'
    dist.mkdir()
    for name in ('index.html','sw.js','manifest.webmanifest','pwa-version.json'):
        (dist/name).write_text('public synthetic application fixture',encoding='utf-8')
    (dist/'assets').mkdir()
    (dist/'assets'/'index-test.js').write_text('public application code',encoding='utf-8')
    return dist


def test_immutable_snapshot_reuses_same_content_and_preserves_previous_version(tmp_path):
    dist, root = public_build(tmp_path), tmp_path/'versions'
    first = module.snapshot(dist,root)
    assert module.snapshot(dist,root) == first
    assert (Path(first['snapshotPath'])/'assets'/'index-test.js').read_text() == 'public application code'
    (dist/'assets'/'index-test.js').write_text('new public application version')
    second = module.snapshot(dist,root)
    assert first != second
    assert (Path(first['snapshotPath'])/'assets'/'index-test.js').read_text() == 'public application code'


def test_changed_existing_snapshot_is_preserved_and_never_silently_used(tmp_path):
    dist, root = public_build(tmp_path), tmp_path/'versions'
    published = module.snapshot(dist,root)
    changed = Path(published['snapshotPath'])/'index.html'
    changed.write_text('changed fixture')
    with pytest.raises(ValueError): module.snapshot(dist,root)
    assert changed.read_text() == 'changed fixture'


@pytest.mark.parametrize('name',['patient.dcm','application.sqlite3','session.key','.env','images/patient.png','assets/weights.pt'])
def test_snapshot_rejects_files_outside_public_distribution(tmp_path,name):
    dist = public_build(tmp_path)
    extra = dist/name
    extra.parent.mkdir(parents=True,exist_ok=True)
    extra.write_text('synthetic sentinel that must never be published')
    with pytest.raises(ValueError): module.snapshot(dist,tmp_path/'versions')


def test_incomplete_pwa_build_is_never_published(tmp_path):
    dist = public_build(tmp_path)
    (dist/'sw.js').unlink()
    with pytest.raises(ValueError): module.snapshot(dist,tmp_path/'versions')


def test_fingerprint_tracks_application_sources_but_excludes_runtime_and_dependencies(tmp_path):
    source = tmp_path/'frontend'
    (source/'src').mkdir(parents=True)
    (source/'src'/'App.vue').write_text('synthetic public source')
    before = module.source_fingerprint(source)
    for name in ('dist','runtime','node_modules'):
        (source/name).mkdir()
        (source/name/'sentinel.txt').write_text('ignored synthetic local state')
    assert module.source_fingerprint(source) == before
    (source/'src'/'App.vue').write_text('changed public source')
    assert module.source_fingerprint(source) != before


def test_copy_failure_does_not_publish_partial_or_remove_other_versions(tmp_path,monkeypatch):
    dist, root = public_build(tmp_path), tmp_path/'versions'
    first = module.snapshot(dist,root)
    (dist/'sw.js').write_text('new public worker')
    def failed(*args,**kwargs): raise OSError('synthetic full disk')
    monkeypatch.setattr(module.shutil,'copyfile',failed)
    with pytest.raises(OSError): module.snapshot(dist,root)
    assert sorted(path.name for path in root.iterdir()) == [first['digest']]
