"""Artificial stage-one records; no GPU, private participants or weight reuse."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knee import gradual_selection as subject
from knee import methodology_campaign as core
from knee import methodology_campaign_colab as previous
from knee import methodology_release as release
from knee import image_campaign_recovery as recovery
from knee.aligned_training import engine, runner
from knee.colab_data import sha_file
from test_methodology_campaign import synthetic_preflight, synthetic_neural_job

ROOT = Path(__file__).resolve().parents[1]


def fake_model_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(128, 128, bias=False)
        model.task_head = torch.nn.Linear(2, 1)
    return {name:t.detach().clone() for name,t in model.state_dict().items()}


def fake_job(target, request, rows, registry):
    target.mkdir()
    directory = target / "job"
    job = synthetic_neural_job(directory, request, rows, registry, fake_model_state(), "cuda:0")
    model = request["model_id"]
    predictions = runner.read_json(directory / "best_validation_predictions.json")
    scale = {"image_skelex":2., "image_mambaout_tiny":1.5,
        "image_densenet121":.8, "image_vit_b16":.7}.get(model,0.)
    predictions["logits"] = [scale if y else -scale for y in predictions["targets"]]
    metrics = engine.binary_metrics(predictions["targets"],predictions["logits"])
    record = runner.read_json(directory / "training_record.json")
    record["model_id"] = model
    record["best_validation_ap"] = metrics["average_precision_score"]
    for item in record["history"]:item["validation"] = metrics
    for name,value in (("best_validation_predictions.json",predictions),("training_record.json",record)):
        (directory/name).unlink()
        runner.save_json(directory/name,value)
        job["files_sha256"][name] = sha_file(directory/name)
    job["metrics"] = metrics
    job["record_sha256"] = core.digest({k:v for k,v in job.items() if k!="record_sha256"})
    (directory/runner.JOB_FILENAME).unlink()
    runner.save_json(directory/runner.JOB_FILENAME,job)
    receipt = {"version":previous.VERSION,"key":core.job_key(request),
        "files_sha256":{**job["files_sha256"],runner.JOB_FILENAME:sha_file(directory/runner.JOB_FILENAME)},
        "job_record_sha256":job["record_sha256"]}
    runner.save_json(target/previous.JOB_RECEIPT,receipt)
    return directory,job,predictions,receipt


def screening_sources(root, value, plan, registry):
    common = {"scope":"development_only","seed":2026,"stage":"stage_1","folds_completed":list(range(5)),
        "code_revision":"a"*40,"colab_runtime_executed":True,"local_rehearsal":False,
        "result_files_saved_and_readback_verified":True,"training_plan_sha256":release.PINS[release.PLAN_PATH],
        "modeling_release_sha256":release.PINS[release.RELEASE_PATH],"development_package_sha256":value["package_sha256"],
        "model_input_reference_sha256":sha_file(value["reference_path"]),"development_knees":len(value["rows"]),
        "development_participants":len({r["src_subject_id"] for r in value["rows"]}),"cpu_tests":{"passed":True,"skipped":0}}
    for flag in ("reserved_test_accessed","reserved_test_predictive_evaluation_opened","originals_modified","partitions_modified",
            "encoder_unfreezing_executed","calibration_performed","threshold_selected","final_refit_executed",
            "full_encoder_training_authorized","predictive_performance_claimed","unbiased_final_performance_claimed",
            "historical_v1_1_jobs_reused","fusion_training_executed","completed_checkpoint_payloads_copied_on_recovery",
            "checkpoint_payloads_rehashed_on_recovery","pretrained_weights_loaded_for_completed_only_models"):
        common[flag] = False
    summaries, roots = {}, {}
    for number,campaign in (("40","densenet_frozen"),("41","remaining_encoders_frozen")):
        campaign_dir=root/(previous.CAMPAIGN_PREFIX+campaign+"_artificial")
        campaign_dir.mkdir()
        roots[number]=campaign_dir
        identity={key:common[key] for key in ("seed","stage","code_revision","training_plan_sha256","modeling_release_sha256",
            "development_package_sha256","model_input_reference_sha256")}
        identity["campaign"]=campaign
        runner.save_json(campaign_dir/previous.CAMPAIGN_FILENAME,{"version":previous.VERSION,"identity":identity})
        summary={**common,"campaign":campaign,"persistent_campaign_folder":campaign_dir.name,
            "job_results":[],"model_results":{},"private_job_locations":{},"artifact_files_sha256":{}}
        groups={}
        for index,request in enumerate(core.requests(campaign,value["package_sha256"],"0"*64,plan,registry)):
            directory,job,predictions,receipt=fake_job(campaign_dir/("job_attempt_"+str(index)),request,value["rows"],registry)
            key=core.job_key(request)
            summary["job_results"].append({"key":key,"job_record_sha256":job["record_sha256"],"metrics":job["metrics"]})
            summary["private_job_locations"][key]=directory.relative_to(campaign_dir).as_posix()
            summary["artifact_files_sha256"].update({key+"/"+name:checksum for name,checksum in receipt["files_sha256"].items()})
            groups.setdefault(request["model_id"],[]).append((request,job,predictions))
        summary["job_count"]=len(summary["job_results"])
        for model,group in groups.items():
            metrics=[{"validation_fold":r["validation_fold"],**j["metrics"]} for r,j,p in group]
            targets=[y for r,j,p in group for y in p["targets"]]
            logits=[x for r,j,p in group for x in p["logits"]]
            summary["model_results"]["clinical_main__"+model]={"fivefold_selection_metrics":engine.aggregate_fold_metrics(metrics),
                "pooled_oof_metrics_descriptive_only":engine.binary_metrics(targets,logits),"out_of_fold_predictions":len(targets)}
        if number=="40":
            summary.update(version=core.VERSION,status=core.STATUS,training_executed=True)
            runner.save_json(campaign_dir/core.FILENAME,summary)
            summary={**summary,"core_record_sha256":sha_file(campaign_dir/core.FILENAME)}
            runner.save_json(campaign_dir/previous.SUMMARY_FILENAME,summary)
        else:
            summary.update(version=recovery.VERSION,status=recovery.STATUS,completed_jobs_preserved=15,jobs_executed_this_invocation=10,
                checkpoint_rehash_required_before_weight_reuse=True,legacy_campaign_identity_preserved=True,
                legacy_campaign_identity_sha256=sha_file(campaign_dir/previous.CAMPAIGN_FILENAME))
            completion=campaign_dir/"recovery_completion_artificial"
            completion.mkdir()
            runner.save_json(completion/recovery.SUMMARY_FILENAME,summary)
        summaries[number]=summary
    authority=subject.screening_authority(summaries["40"],summaries["41"])
    runner.save_json(root/subject.AUTHORITY_FILENAME,authority)
    return roots,summaries


class GradualSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
        _,_,cls.plan,cls.registry=release.load_release(ROOT)
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.value=synthetic_preflight(cls.root)
        cls.sources,cls.summaries=screening_sources(cls.root,cls.value,cls.plan,cls.registry)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        subject._VALIDATED.clear()

    def prepare(self,name):
        path=self.root/name
        value=subject.prepare_selection(self.sources["40"],self.sources["41"],path,self.plan,self.registry,self.value["rows"],
            authority_path=self.root/subject.AUTHORITY_FILENAME)
        return path,value

    def request(self,path,model="image_skelex"):
        request=next(r for r in core.requests("remaining_encoders_frozen",self.value["package_sha256"],"0"*64,self.plan,self.registry)
            if r["model_id"]==model and r["validation_fold"]==0)
        request.update(stage="stage_2",selection_evidence_sha256=sha_file(path))
        return request

    def discovery_records(self, root):
        """Only small campaign records; discovery must never need fitted weights."""
        for number, source in self.sources.items():
            target = root / source.name
            target.mkdir()
            names = [previous.CAMPAIGN_FILENAME]
            if number == "40":
                names += [core.FILENAME, previous.SUMMARY_FILENAME]
            else:
                (target / "recovery_completion_artificial").mkdir()
                names.append("recovery_completion_artificial/" + recovery.SUMMARY_FILENAME)
            for name in names:
                (target / name).write_bytes((source / name).read_bytes())
        (root / subject.AUTHORITY_FILENAME).write_bytes((self.root / subject.AUTHORITY_FILENAME).read_bytes())

    def test_discovery_ignores_incomplete_and_unapproved_complete_attempts_in_both_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.discovery_records(root)
            for number, campaign in (("40", "densenet_frozen"), ("41", "remaining_encoders_frozen")):
                authorized = root / self.sources[number].name
                incomplete = root / (previous.CAMPAIGN_PREFIX + campaign + "_old_incomplete")
                incomplete.mkdir()
                (incomplete / previous.CAMPAIGN_FILENAME).write_bytes((authorized / previous.CAMPAIGN_FILENAME).read_bytes())
                wrong = root / (previous.CAMPAIGN_PREFIX + campaign + "_other_complete")
                wrong.mkdir()
                (wrong / previous.CAMPAIGN_FILENAME).write_bytes((authorized / previous.CAMPAIGN_FILENAME).read_bytes())
                summary = {**self.summaries[number], "persistent_campaign_folder": wrong.name,
                    "executed_at_utc": "unapproved_other_completion"}
                if number == "40":
                    runner.save_json(wrong / previous.SUMMARY_FILENAME, summary)
                else:
                    completion = wrong / "recovery_completion_other"
                    completion.mkdir()
                    runner.save_json(completion / recovery.SUMMARY_FILENAME, summary)
                    # A normal second close has a different digest and is not selected.
                    newer = authorized / "recovery_completion_newer"
                    newer.mkdir()
                    runner.save_json(newer / recovery.SUMMARY_FILENAME,
                        {**self.summaries[number], "executed_at_utc": "different_later_completion"})
                self.assertEqual(subject.find_authorized_campaign(root, campaign, self.value["package_sha256"]), authorized)

    def test_discovery_missing_approved_source_or_copied_approved_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.discovery_records(root)
            source = root / self.sources["40"].name
            path = source / previous.SUMMARY_FILENAME
            raw = path.read_bytes()
            value = json.loads(raw)
            path.write_text(json.dumps({**value, "unapproved": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one_privately_authorized_completed"):
                subject.find_authorized_campaign(root, "densenet_frozen", self.value["package_sha256"])
            path.write_bytes(raw)
            duplicate = root / (previous.CAMPAIGN_PREFIX + "densenet_frozen_duplicate")
            duplicate.mkdir()
            (duplicate / previous.SUMMARY_FILENAME).write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "one_privately_authorized_completed"):
                subject.find_authorized_campaign(root, "densenet_frozen", self.value["package_sha256"])

    def test_discovery_duplicate_identical_41_closures_are_rejected_not_ordered_by_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.discovery_records(root)
            source = root / self.sources["41"].name
            duplicate = source / "recovery_completion_identical_copy"
            duplicate.mkdir()
            (duplicate / recovery.SUMMARY_FILENAME).write_bytes(
                (source / "recovery_completion_artificial" / recovery.SUMMARY_FILENAME).read_bytes())
            with self.assertRaisesRegex(ValueError, "one_exact_privately_authorized_screening_summary"):
                subject.find_authorized_campaign(root, "remaining_encoders_frozen", self.value["package_sha256"])

    def test_discovery_authenticates_package_family_manifest_and_original_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.discovery_records(root)
            with self.assertRaisesRegex(ValueError, "family_or_package_mismatch"):
                subject.find_authorized_campaign(root, "densenet_frozen", "f" * 64)
            with self.assertRaisesRegex(ValueError, "registered_screening_source_family"):
                subject.find_authorized_campaign(root, "clinical", self.value["package_sha256"])
            source = root / self.sources["40"].name
            manifest = source / previous.CAMPAIGN_FILENAME
            raw = manifest.read_bytes()
            value = json.loads(raw)
            value["identity"]["campaign"] = "remaining_encoders_frozen"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "campaign_manifest_binding"):
                subject.find_authorized_campaign(root, "densenet_frozen", self.value["package_sha256"])
            manifest.write_bytes(raw)
            moved = root / (previous.CAMPAIGN_PREFIX + "densenet_frozen_renamed")
            source.rename(moved)
            with self.assertRaisesRegex(ValueError, "must_name_its_campaign_folder"):
                subject.find_authorized_campaign(root, "densenet_frozen", self.value["package_sha256"])

    def test_exact_30_job_selection_is_deterministic_and_does_not_open_weights(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            if path.suffix==".safetensors":raise AssertionError("No stage-one weight payload reads")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",new=guard),patch("safetensors.torch.load_file",side_effect=AssertionError("No weights")):
            first,value=self.prepare("selection1.json")
            second,other=self.prepare("selection2.json")
            validated=subject.validate_selection_evidence(first,self.request(first),self.plan,self.registry,self.value["rows"])
        self.assertEqual(first.read_bytes(),second.read_bytes())
        self.assertEqual(value,other)
        self.assertEqual(value["model_order"],list(subject.MODEL_ORDER))
        self.assertEqual(value["promoted_modern_models"],["image_skelex","image_mambaout_tiny"])
        self.assertEqual(value["excluded_modern_models"],subject.EXCLUDED)
        self.assertEqual(len(value["screening_jobs"]),30)
        self.assertFalse(value["stage_one_weights_reused"])
        self.assertIsNone(validated["selected_image_encoder_id"])

    def test_cached_fold_validation_reopens_only_evidence_and_private_authority(self):
        path,_=self.prepare("cached.json")
        first=subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry,self.value["rows"])
        with patch.object(subject,"_build",side_effect=AssertionError("No repeated30jobread after authenticatedproof")):
            next_request=self.request(path)
            next_request["validation_fold"]=1
            second=subject.validate_selection_evidence(path,next_request,self.plan,self.registry,self.value["rows"])
        self.assertEqual(first,second)
        first["retained_image_models"].clear()
        third=subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry,self.value["rows"])
        self.assertEqual(third["retained_image_models"],list(subject.MODEL_ORDER))

    def test_predictor_or_receipt_mutation_is_rejected_before_selection(self):
        root=self.sources["41"]/"job_attempt_0"
        for path in (root/"job/best_validation_predictions.json",root/previous.JOB_RECEIPT):
            raw=path.read_bytes()
            try:
                value=json.loads(raw)
                if path.name==previous.JOB_RECEIPT:value["job_record_sha256"]="f"*64
                else:value["targets"][0]=1-value["targets"][0]
                path.write_text(json.dumps(value),encoding="utf-8")
                with self.assertRaises(ValueError):self.prepare("bad_source.json")
            finally:path.write_bytes(raw)

    def test_exclusion_promotion_and_row_mutations_cannot_change_authorized_selection(self):
        path,value=self.prepare("trusted.json")
        for key,replacement in (("excluded_modern_models",{}),("promoted_modern_models",["image_convnextv2_tiny","image_skelex"]),
                ("model_order",list(reversed(subject.MODEL_ORDER)))):
            changed={**value,key:replacement}
            target=self.root/(key+".json")
            runner.save_json(target,changed)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"artifact_must_equal"):
                subject.validate_selection_evidence(target,self.request(target),self.plan,self.registry,self.value["rows"])
        rows=deepcopy(self.value["rows"])
        rows[0]["PROGRESION_48M"]=1-rows[0]["PROGRESION_48M"]
        with self.assertRaisesRegex(ValueError,"development_binding"):
            subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry,rows)

    def test_fusion_other_model_and_missing_development_rows_are_rejected(self):
        path,_=self.prepare("bounded.json")
        for model in ("image_convnextv2_tiny","image_biomedclip"):
            with self.assertRaisesRegex(ValueError,"only_authorizes"):
                subject.validate_selection_evidence(path,self.request(path,model),self.plan,self.registry,self.value["rows"])
        fusion=self.request(path)
        fusion.update(model_id="fusion_intermediate_concat",stage="fusion",fold_image_job_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"only_authorizes"):
            subject.validate_selection_evidence(path,fusion,self.plan,self.registry,self.value["rows"])
        with self.assertRaisesRegex(ValueError,"manifest_rows"):
            subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry)

    def test_private_authority_cannot_be_changed_after_cached_validation(self):
        path,_=self.prepare("authority_bound.json")
        subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry,self.value["rows"])
        authority_path=self.root/subject.AUTHORITY_FILENAME
        raw=authority_path.read_bytes()
        try:
            authority=json.loads(raw)
            authority["source41_summary_digest"]="f"*64
            authority_path.write_text(json.dumps(authority),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"private_authority_changed"):
                subject.validate_selection_evidence(path,self.request(path),self.plan,self.registry,self.value["rows"])
        finally:authority_path.write_bytes(raw)


class ActualGradualArchitectureMappingTests(unittest.TestCase):
    def test_complete_registered_architectures_map_all_three_phases_without_weights(self):
        """Real constructors, no pretrained download, forward, optimizer or fit.

        Large architectures use shape-only tensors. MambaOut's constructor
        materializes its drop-path schedule, so that small model stays on CPU.
        """
        from knee.aligned_training import models
        from torchvision.models import densenet121, vit_b_16
        from transformers import ViTMAEConfig, ViTMAEModel
        import timm
        _, _, plan, _ = release.load_release(ROOT)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                for model_id in subject.MODEL_ORDER:
                    device = "cpu" if model_id == "image_mambaout_tiny" else "meta"
                    with torch.device(device):
                        if model_id == "image_densenet121":
                            complete = densenet121(weights=None)
                            encoder = torch.nn.Sequential(complete.features, torch.nn.ReLU(inplace=False),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
                            del complete
                        elif model_id == "image_vit_b16":
                            encoder = vit_b_16(weights=None)
                            encoder.heads = torch.nn.Identity()
                        elif model_id == "image_skelex":
                            config = ViTMAEConfig(hidden_size=1024, num_hidden_layers=24,
                                num_attention_heads=16, intermediate_size=4096, image_size=224,
                                patch_size=16, num_channels=3, mask_ratio=0)
                            config._attn_implementation = "eager"
                            encoder = models.SkelexTrainingFeatures(ViTMAEModel(config))
                        else:
                            encoder = timm.create_model("mambaout_tiny", pretrained=False, num_classes=1000)
                            encoder.reset_classifier(0)
                        model = models.RegisteredNeuralModel(model_id, encoder, model_id)
                        model.train()
                        counts = []
                        for phase in ("head_only", "last_block", "full"):
                            models.set_phase(model, phase)
                            trainable = [p for p in model.parameters() if p.requires_grad]
                            counts.append(sum(p.numel() for p in trainable))
                            groups = models.optimizer_groups(model, plan, phase, stage="stage_2")
                            grouped = [id(p) for g in groups for p in g["params"]]
                            self.assertEqual(set(grouped), {id(p) for p in trainable})
                            self.assertEqual(len(grouped), len(set(grouped)))
                            self.assertTrue(all(not m.training for m in encoder.modules()
                                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)))
                        self.assertLess(counts[0], counts[1])
                        self.assertLess(counts[1], counts[2])
                        if model_id == "image_skelex":
                            self.assertFalse(encoder.encoder.embeddings.position_embeddings.requires_grad)
                        del model, encoder
        finally:
            torch.set_num_threads(threads)


if __name__=="__main__":unittest.main()
