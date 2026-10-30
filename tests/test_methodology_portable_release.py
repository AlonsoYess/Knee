"""Pinned text remains identical across Windows/Git without relaxing content."""
from pathlib import Path
import shutil
import tempfile
import unittest

from knee import methodology_release as release
from knee.pretrained_verification import SOURCE_PINS

ROOT=Path(__file__).resolve().parents[1]


class PortableReleaseTests(unittest.TestCase):
    def fixture(self, root, endings):
        destination=Path(root)/"snapshot"
        destination.mkdir()
        for name in release.PINS:
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        from knee.final_model_release import PINS
        for name in {*PINS, *SOURCE_PINS, "configs/pretrained_weights_v1.json", "configs/training_plan_v1.json"}:
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            raw=(ROOT/name).read_bytes().replace(b"\r\n",b"\n")
            target.write_bytes(raw.replace(b"\n",b"\r\n") if endings=="CRLF" else raw)
        return destination

    def test_windows_and_git_sources_load_same_complete_release(self):
        expected=release.load_release(ROOT)
        for endings in ("LF","CRLF"):
            with self.subTest(endings=endings),tempfile.TemporaryDirectory() as directory:
                self.assertEqual(release.load_release(self.fixture(directory,endings)),expected)

    def test_added_code_or_space_rejected_even_when_newline_portability_enabled(self):
        for suffix in (b"\n# additional code comment\n",b" "):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as directory:
                target=self.fixture(directory,"LF")
                source=target/"src/knee/clinical_fold_inputs.py"
                source.write_bytes(source.read_bytes()+suffix)
                with self.assertRaises(ValueError): release.load_release(target)

    def test_catalog_content_and_recipe_edits_still_rejected(self):
        import json
        for name in ("configs/pretrained_weights_v1.json","configs/training_plan_v1_2.json"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                target=self.fixture(directory,"LF")
                source=target/name
                data=json.loads(source.read_text())
                data["unreviewed_override"]=True
                source.write_text(json.dumps(data),encoding="utf-8")
                with self.assertRaises(ValueError): release.load_release(target)


if __name__=="__main__": unittest.main()
