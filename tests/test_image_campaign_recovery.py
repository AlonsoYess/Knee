"""Artificial recovery evidence; no participant training, GPU or network."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knee import image_campaign_recovery as subject
from knee import methodology_campaign as core
from knee import methodology_campaign_colab as previous
from knee import methodology_release as release
from knee.aligned_training import engine,models,runner
from knee.aligned_training.authority import require_permit
from knee.colab_data import sha_file
from test_methodology_campaign import synthetic_preflight,synthetic_neural_job
from test_methodology_image_campaign import tiny_image_model,synthetic_evidence

ROOT=Path(__file__).resolve().parents[1]


def large_artificial_model(*args,**kwargs):
    model,adoption=tiny_image_model(*args,**kwargs)
    # Enough tensors for the production minimum size, without any GPU compute.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model.encoder=torch.nn.Linear(128,128,bias=False)
    model.encoder.requires_grad_(False)
    return model,adoption


def synthetic_completed(root,request,rows,registry):
    model,_=large_artificial_model(request["model_id"])
    initial={key:value.detach().clone() for key,value in model.state_dict().items()}
    jobroot=root/"job"
    record=synthetic_neural_job(jobroot,request,rows,registry,initial,"cuda:0")
    path=jobroot/"training_record.json"
    record_engine=runner.read_json(path)
    record_engine["model_id"]=request["model_id"]
    path.unlink()
    runner.save_json(path,record_engine)
    record["files_sha256"]["training_record.json"]=sha_file(path)
    record["record_sha256"]=core.digest({k:v for k,v in record.items() if k!="record_sha256"})
    (jobroot/runner.JOB_FILENAME).unlink()
    runner.save_json(jobroot/runner.JOB_FILENAME,record)
    receipt={"version":previous.VERSION,"key":core.job_key(request),
        "files_sha256":{**record["files_sha256"],runner.JOB_FILENAME:sha_file(jobroot/runner.JOB_FILENAME)},
        "job_record_sha256":record["record_sha256"]}
    runner.save_json(root/previous.JOB_RECEIPT,receipt)
    return jobroot,receipt,record


class ImageCampaignRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
        _,cls.catalog,cls.plan,cls.registry=release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_compatibility_rejects_other_recipe_membership_or_implementation(self):
        compatibility=runner.read_json(ROOT/subject.COMPATIBILITY_PATH)
        sources={name:allowed[0] for name,allowed in compatibility["legacy_source_files_sha256"].items()}
        legacy={"code_revision":compatibility["legacy_code_revision"],"campaign":subject.CAMPAIGN,
            "training_plan_sha256":release.PINS[release.PLAN_PATH],"development_package_sha256":"a"*64,
            "source_files_sha256":sources}
        current={**legacy,"code_revision":"f"*40,"source_files_sha256":dict(sources)}
        current["source_files_sha256"]["src/knee/aligned_training/engine.py"]="f"*64
        self.assertTrue(subject.compatible_identity(legacy,current,compatibility))
        for key,value in (("code_revision","b"*40),("development_package_sha256","b"*64)):
            changed={**legacy,key:value}
            with self.assertRaises(ValueError):subject.compatible_identity(changed,current,compatibility)
        changed=deepcopy(current)
        changed["source_files_sha256"]["src/knee/aligned_training/models.py"]="e"*64
        with self.assertRaisesRegex(ValueError,"non_recovery_source"):
            subject.compatible_identity(legacy,changed,compatibility)
        changed=deepcopy(legacy)
        changed["source_files_sha256"]["src/knee/aligned_training/engine.py"]="e"*64
        with self.assertRaisesRegex(ValueError,"legacy_campaign_implementation"):
            subject.compatible_identity(changed,current,compatibility)

    def test_small_json_rejects_duplicates_nonfinite_and_unbounded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"value.json"
            for raw in ('{"a":1,"a":2}','{"value":NaN}','[]'):
                path.write_text(raw,encoding="utf-8")
                with self.assertRaises(ValueError):subject._small_json(path)
            path.write_text('{"a":1}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"hash_mismatch"):
                subject._small_json(path,"f"*64)
            with path.open("wb") as handle:handle.truncate(subject.MAX_JSON_BYTES+1)
            with self.assertRaisesRegex(ValueError,"bounded_plain"):
                subject._small_json(path)

    def test_complete_job_recovery_never_opens_checkpoint_payloads_and_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            value=synthetic_preflight(root)
            request=core.requests(subject.CAMPAIGN,value["package_sha256"],"b"*64,self.plan,self.registry)[0]
            attempt=root/"job_attempt_x"
            attempt.mkdir()
            jobroot,receipt,_=synthetic_completed(attempt,request,value["rows"],self.registry)
            original_open=Path.open
            def no_checkpoint(path,*args,**kwargs):
                if path.suffix==".safetensors":raise AssertionError("Recovered payload must never be opened")
                return original_open(path,*args,**kwargs)
            with patch.object(Path,"open",new=no_checkpoint),patch("safetensors.torch.load_file",side_effect=AssertionError("No load")):
                item=subject.read_job_metadata(jobroot,receipt,request,value["rows"],self.plan,self.registry)
            self.assertFalse(item["checkpoint_payloads_rehashed_this_run"])
            self.assertEqual(item["record"]["request"],request)
            prediction=jobroot/"best_validation_predictions.json"
            prediction.write_text(prediction.read_text()+" ",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"json_hash_mismatch"):
                subject.read_job_metadata(jobroot,receipt,request,value["rows"],self.plan,self.registry)
            prediction.write_text(prediction.read_text()[:-1],encoding="utf-8")
            (jobroot/"best_model.safetensors").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError,"bounded_existing"):
                subject.read_job_metadata(jobroot,receipt,request,value["rows"],self.plan,self.registry)

    def test_15_preserved_10_pending_real_runner_publication_then_25_skip_without_gpu(self):
        from knee import training_gpu_runtime as gpu_runtime
        from knee import final_models_preflight
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory)
            value=synthetic_preflight(root)
            persistent=root/"persistent"
            persistent.mkdir()
            weights=root/"weights"
            weights.mkdir()
            current=previous.campaign_identity(ROOT,value["preflight"],value["reference_path"],"a"*40,subject.CAMPAIGN)
            compatibility=runner.read_json(ROOT/subject.COMPATIBILITY_PATH)
            legacy={**current,"code_revision":compatibility["legacy_code_revision"],
                "source_files_sha256":{name:allowed[0] for name,allowed in compatibility["legacy_source_files_sha256"].items()}}
            campaign=persistent/(previous.CAMPAIGN_PREFIX+subject.CAMPAIGN+"_synthetic")
            campaign.mkdir()
            runner.save_json(campaign/previous.CAMPAIGN_FILENAME,{"version":previous.VERSION,"identity":legacy})
            identity_before=(campaign/previous.CAMPAIGN_FILENAME).read_bytes()
            requests=core.requests(subject.CAMPAIGN,value["package_sha256"],"b"*64,self.plan,self.registry)
            protected=[]
            for index,request in enumerate(requests[:15]):
                target=campaign/("job_attempt_synthetic_"+str(index))
                target.mkdir()
                path,_,_=synthetic_completed(target,request,value["rows"],self.registry)
                protected.append(path)
            records={r["model_id"]:r for r in requests if r["validation_fold"]==0}
            evidence=synthetic_evidence(value,self.catalog)
            probes=[]
            def pending_probe(repository,proof,package,reference,cache,output,private,campaign_dir,pending_models):
                probes.append(tuple(pending_models))
                return {"model_eligibility":{m:evidence["model_eligibility"][m] for m in pending_models},
                    "current_gpu_probed_models":list(pending_models)}
            initial_models=[]
            original_initial=core._initial_image_state
            def initial(model,*args):
                initial_models.append(model)
                return original_initial(model,*args)
            request_map={(r["model_id"],r["validation_fold"]):r for r in requests}
            calls=[]
            def artificial_engine(model,training,validation,plan,*,device,seed,positive_weight,stage,output_dir,
                    maximum_steps,smoke,execution_authority,epoch_checkpoint=None):
                permit=execution_authority
                require_permit(permit,model_id=model.model_id,stage=stage,smoke=False)
                engine._validate_loader_bindings(training,validation,permit,positive_weight)
                self.assertIsNotNone(epoch_checkpoint)
                self.assertFalse(smoke)
                self.assertIsNone(maximum_steps)
                request=request_map[(model.model_id,permit.validation_fold)]
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                output=Path(output_dir)
                synthetic_neural_job(output,request,value["rows"],self.registry,initial,device)
                record=runner.read_json(output/"training_record.json")
                record.update(model_id=model.model_id,resume_supported=True,optimizer_state_saved=True,
                    epoch_checkpoint_version="frozen_image_epoch_resume_v1.0",resumed_from_epoch=0)
                (output/"training_record.json").unlink()
                runner.save_json(output/"training_record.json",record)
                for name in (runner.JOB_FILENAME,"clinical_state.json"):(output/name).unlink()
                calls.append((model.model_id,permit.validation_fold))
                return record
            original_open=Path.open
            def prevent_old_payload(path,*args,**kwargs):
                if path.suffix==".safetensors" and any(path.is_relative_to(p) for p in protected):
                    raise AssertionError("Complete Drive checkpoint must not be opened")
                return original_open(path,*args,**kwargs)
            stack.enter_context(patch.object(Path,"open",new=prevent_old_payload))
            for name in ("torch.optim.AdamW","torch.Tensor.backward","torch.autograd.backward",
                    "urllib.request.OpenerDirector.open","requests.sessions.Session.send","shutil.copytree"):
                stack.enter_context(patch(name,side_effect=AssertionError("No real fitting, download or whole-job copying")))
            stack.enter_context(patch.object(subject,"_pending_gpu_evidence",side_effect=pending_probe))
            stack.enter_context(patch.object(final_models_preflight,"checkpoint_files",return_value=[]))
            stack.enter_context(patch.object(models,"build_model",side_effect=large_artificial_model))
            stack.enter_context(patch.object(core,"_initial_image_state",side_effect=initial))
            stack.enter_context(patch.object(engine,"train_neural_fold",side_effect=artificial_engine))
            tests={"passed":True,"skipped":0,"tests_run":1,"scope":list(subject.test_modules()),
                "test_runtime":"gpu_build_cpu_compute_no_training"}
            runtime={"version":gpu_runtime.VERSION,"training_executed":False,"host_python_modified":False,
                "pretrained_weights_downloaded":False,"cpu_only":False,"python_version":"3.12.14",
                "source_sha256":{name:sha_file(ROOT/name) for name in gpu_runtime.SOURCE_FILES}}
            result=subject.train(ROOT,value["preflight"],value["package"],value["reference_path"],root/"output",persistent,
                campaign=subject.CAMPAIGN,approval_statement=subject.APPROVAL_STATEMENTS[subject.CAMPAIGN],
                code_revision="a"*40,cpu_tests=tests,runtime=runtime,weights_root=weights)
            self.assertEqual(result["completed_jobs_preserved"],15)
            self.assertEqual(result["jobs_executed_this_invocation"],10)
            self.assertEqual(calls,[(m,f) for m in core.REMAINING_IMAGE_IDS[3:] for f in range(5)])
            self.assertEqual(initial_models,list(core.REMAINING_IMAGE_IDS[3:]))
            self.assertEqual(probes,[core.REMAINING_IMAGE_IDS[3:]])
            self.assertEqual((campaign/previous.CAMPAIGN_FILENAME).read_bytes(),identity_before)
            self.assertEqual(len(result["artifact_files_sha256"]),150)
            protected.extend(item.parent/"job" for item in campaign.glob("job_attempt_*/"+previous.JOB_RECEIPT)
                if item.parent/"job" not in protected)
            with patch.object(runner,"execute",side_effect=AssertionError("No completed job retraining")), \
                    patch.object(core,"_initial_image_state",side_effect=AssertionError("No weights when all complete")):
                second=subject.train(ROOT,value["preflight"],value["package"],value["reference_path"],root/"second",persistent,
                    campaign=subject.CAMPAIGN,approval_statement=subject.APPROVAL_STATEMENTS[subject.CAMPAIGN],
                    code_revision="a"*40,cpu_tests=tests,runtime=runtime,weights_root=None)
            self.assertEqual(second["completed_jobs_preserved"],25)
            self.assertEqual(second["jobs_executed_this_invocation"],0)
            self.assertEqual(second["model_results"],result["model_results"])
            self.assertEqual(probes[-1],())
            self.assertFalse(second["training_executed_this_invocation"])
            self.assertTrue(second["checkpoint_rehash_required_before_weight_reuse"])


if __name__=="__main__":unittest.main()
