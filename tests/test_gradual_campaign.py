"""Artificial stage-two orchestration; no GPU, network or participant fits."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knee import gradual_campaign as subject
from knee import methodology_campaign as prior
from knee import methodology_release as release
from knee.aligned_training import engine, runner
from knee.colab_data import sha_file
from test_methodology_campaign import synthetic_preflight, synthetic_neural_job

ROOT = Path(__file__).resolve().parents[1]


def synthetic_gradual_job(output, request, rows, registry):
    """Invent metadata for fake execution only, never represent a real GPU."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        tiny = torch.nn.Linear(2,2)
    state = {"encoder.weight":tiny.weight.detach(),"encoder.bias":tiny.bias.detach(),
        "task_head.weight":torch.ones(1,2),"task_head.bias":torch.ones(1)}
    record = synthetic_neural_job(output,request,rows,registry,state,"cuda:0")
    training = runner.read_json(output / "training_record.json")
    row = deepcopy(training["history"][0])
    history = []
    for epoch in range(1,20):
        item=deepcopy(row)
        item.update(epoch=epoch,phase=engine.epoch_phase(request["model_id"],"stage_2",epoch),
            learning_rate_factor=engine.learning_rate_factor(epoch),selected_best=epoch==11)
        history.append(item)
    from knee.gradual_epoch_resume import VERSION
    training.update(model_id=request["model_id"],stage="stage_2",history=history,completed_epochs=19,
        best_epoch=11,optimizer_updates=row["optimizer_updates"]*19,resume_supported=True,optimizer_state_saved=True,
        epoch_checkpoint_version=VERSION,resumed_from_epoch=0,
        amp_scale_audit=[{"epoch":item["epoch"],**audit} for item in history for audit in item["amp_scale_audit"]])
    (output / "training_record.json").unlink()
    runner.save_json(output / "training_record.json",training)
    record["selection_evidence"]={"promoted_modern_models":["image_skelex","image_mambaout_tiny"],
        "selected_image_encoder_id":None}
    record["files_sha256"]["training_record.json"]=sha_file(output / "training_record.json")
    record["record_sha256"]=prior.digest({k:v for k,v in record.items() if k!="record_sha256"})
    (output / runner.JOB_FILENAME).unlink()
    runner.save_json(output / runner.JOB_FILENAME,record)
    return record


class GradualCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
        _,cls.catalog,cls.plan,cls.registry=release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_twenty_requests_no_fusion_or_seed_search(self):
        requests=[subject.make_request("a"*64,"b"*64,m,f,2,self.plan,self.registry)
            for m in subject.MODELS for f in range(5)]
        self.assertEqual(len({prior.job_key(r) for r in requests}),20)
        for request in requests:
            self.assertEqual(request["stage"],"stage_2")
            self.assertEqual(request["augmentation_policy"],"training_intensity_gamma_v1.0")
            self.assertEqual(request["seed"],2026)
        for model in ("image_biomedclip","fusion_intermediate_concat","image_convnextv2_tiny"):
            with self.assertRaises(ValueError):subject.make_request("a"*64,"b"*64,model,0,2,self.plan,self.registry)

    def test_unfreezing_authority_requires_actual_synthetic_model_probe(self):
        request=subject.make_request("a"*64,"b"*64,subject.MODELS[0],0,2,self.plan,self.registry)
        for probe in ({},{"model_id":subject.MODELS[0],"architecture_mapping_verified":True}):
            with self.assertRaises(ValueError):subject._authority(request,self.plan,self.registry,probe)
        probe={"model_id":subject.MODELS[0],"architecture_mapping_verified":True,
            "full_backward_and_optimizer_step_verified":True,"micro_batch_size":2}
        value=subject._authority(request,self.plan,self.registry,probe)
        self.assertTrue(value["full_encoder_training_authorized"])
        self.assertFalse(value["reserved_test_access_authorized"])

    def test_real_preparation_ignores_old_incomplete_campaign_and_seals_exact_selection(self):
        from test_gradual_selection import screening_sources
        from knee import gradual_selection
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory)
            value=synthetic_preflight(root)
            sources,_=screening_sources(value["records"],value,self.plan,self.registry)
            old=value["records"]/"methodology_v1_2_campaign_densenet_frozen_old_incomplete"
            old.mkdir()
            identity=runner.read_json(sources["40"]/"campaign_identity.json")
            runner.save_json(old/"campaign_identity.json",identity)
            for name in ("torch.optim.AdamW","torch.Tensor.backward","urllib.request.OpenerDirector.open"):
                stack.enter_context(patch(name,side_effect=AssertionError("Preparation must not train or download")))
            stack.enter_context(patch.object(runner,"execute",side_effect=AssertionError("Preparation must not execute jobs")))
            gradual_selection._VALIDATED.clear()
            first=subject.prepare(ROOT,value["records"],value["package"],value["reference_path"],root/"proof42a",code_revision="a"*40)
            second=subject.prepare(ROOT,value["records"],value["package"],value["reference_path"],root/"proof42b",code_revision="a"*40)
            self.assertEqual(first,second)
            self.assertEqual(first["model_order"],list(subject.MODELS))
            self.assertEqual(first["job_count"],20)
            self.assertFalse(first["training_executed"])
            self.assertFalse(first["optimizer_created"])
            self.assertFalse(first["reserved_test_accessed"])
            self.assertEqual(subject._find_stage_one(value["records"],"densenet_frozen",value["package_sha256"]),sources["40"])
            self.assertEqual((root/"proof42a"/subject.SELECTION_FILENAME).read_bytes(),
                (root/"proof42b"/subject.SELECTION_FILENAME).read_bytes())

    def test_source_change_rejected_before_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            proof=root/"proof";proof.mkdir()
            package=root/"package";package.mkdir()
            weights=root/"weights";weights.mkdir()
            records=root/"records";records.mkdir()
            reference=root/"reference";reference.mkdir()
            ref=reference/"input.json";runner.save_json(ref,{})
            runner.save_json(proof/subject.SELECTION_FILENAME,{})
            record={"version":subject.VERSION,"status":"gradual_campaign_prepared_no_training",
                "code_revision":"a"*40,"training_executed":False,"source_files_sha256":{},"model_order":list(subject.MODELS),
                "job_count":20,"selection_evidence_sha256":sha_file(proof/subject.SELECTION_FILENAME),
                "model_input_reference_sha256":sha_file(ref)}
            record["record_sha256"]=prior.digest(record)
            runner.save_json(proof/subject.PREPARATION_FILENAME,record)
            with patch.object(runner,"execute",side_effect=AssertionError("Must reject before training")), \
                self.assertRaisesRegex(ValueError,"unchanged_authenticated"):
                subject.run(ROOT,proof,package,ref,root/"output",weights_root=weights,persistent_root=records,code_revision="a"*40)

    def test_interrupted_campaign_keeps_seven_jobs_then_executes_thirteen_then_skips_all(self):
        from knee import gradual_selection
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory)
            value=synthetic_preflight(root)
            proof=root/"proof42";proof.mkdir()
            runner.save_json(proof/subject.SELECTION_FILENAME,{"artificial_fixture":True})
            package_hash=value["package_sha256"]
            prep={"version":subject.VERSION,"status":"gradual_campaign_prepared_no_training","code_revision":"a"*40,
                "training_executed":False,"source_files_sha256":subject._sources(ROOT),"model_order":list(subject.MODELS),
                "job_count":20,"development_package_sha256":package_hash,"development_participants":len({r['src_subject_id'] for r in value['rows']}),
                "selection_evidence_sha256":sha_file(proof/subject.SELECTION_FILENAME),
                "model_input_reference_sha256":sha_file(value["reference_path"])}
            prep["record_sha256"]=prior.digest(prep)
            runner.save_json(proof/subject.PREPARATION_FILENAME,prep)
            weights=root/"weights";weights.mkdir()
            records=value["records"]
            stack.enter_context(patch.object(gradual_selection,"validate_selection_evidence",return_value={
                "promoted_modern_models":["image_skelex","image_mambaout_tiny"],"selected_image_encoder_id":None}))
            stack.enter_context(patch("knee.training_smoke.verify_gpu_environment",return_value={}))
            for name in ("torch.optim.AdamW","torch.Tensor.backward","urllib.request.OpenerDirector.open"):
                stack.enter_context(patch(name,side_effect=AssertionError("No real fitting or network")))
            calls=[];probes=[];fail=[True]
            def probe(model_id,catalog,plan,cache,output,**kwargs):
                probes.append(model_id)
                return {"model_id":model_id,"micro_batch_size":2,"architecture_mapping_verified":True,
                    "full_backward_and_optimizer_step_verified":True}
            def execute(request,authority,**kwargs):
                if len(calls)==7 and fail[0]:raise RuntimeError("simulated_colab_disconnect")
                self.assertTrue(authority["full_encoder_training_authorized"])
                self.assertFalse(authority["reserved_test_access_authorized"])
                self.assertIsNotNone(kwargs["epoch_checkpoint"])
                calls.append(prior.job_key(request))
                return synthetic_gradual_job(Path(kwargs["output_dir"]),request,value["rows"],self.registry)
            stack.enter_context(patch.object(subject,"_probe",side_effect=probe))
            stack.enter_context(patch.object(runner,"execute",side_effect=execute))
            arguments=dict(weights_root=weights,persistent_root=records,code_revision="a"*40)
            with self.assertRaisesRegex(RuntimeError,"simulated_colab_disconnect"):
                subject.run(ROOT,proof,value["package"],value["reference_path"],root/"out1",**arguments)
            campaign=next(p for p in records.iterdir() if p.name.startswith("gradual_image_campaign_"))
            protected=[p/"job" for p in campaign.iterdir() if p.name.startswith("job_attempt_")]
            original_open=Path.open
            def forbid_completed_tensors(path,*args,**kwargs):
                if path.suffix==".safetensors" and any(path.is_relative_to(p) for p in protected):
                    raise AssertionError("Completed tensor must not be reread")
                return original_open(path,*args,**kwargs)
            stack.enter_context(patch.object(Path,"open",new=forbid_completed_tensors))
            fail[0]=False
            probes.clear()
            result=subject.run(ROOT,proof,value["package"],value["reference_path"],root/"out2",**arguments)
            self.assertEqual(result["completed_jobs_preserved"],7)
            self.assertEqual(result["jobs_executed_this_invocation"],13)
            self.assertEqual(len(calls),20)
            self.assertEqual(probes,list(subject.MODELS[1:]))
            calls.clear();probes.clear()
            result=subject.run(ROOT,proof,value["package"],value["reference_path"],root/"out3",**arguments)
            self.assertEqual(result["completed_jobs_preserved"],20)
            self.assertEqual(result["jobs_executed_this_invocation"],0)
            self.assertEqual((calls,probes),([],[]))
            self.assertFalse(result["reserved_test_accessed"])
            for item in result["model_results"].values():self.assertEqual(item["out_of_fold_predictions"],len(value["rows"]))

    def test_completed_job_rejects_an_altered_prediction_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            value=synthetic_preflight(root)
            request=subject.make_request(value["package_sha256"],"a"*64,subject.MODELS[0],0,2,self.plan,self.registry)
            output=root/"job"
            record=synthetic_gradual_job(output,request,value["rows"],self.registry)
            receipt={"version":subject.VERSION,"key":prior.job_key(request),"job_record_sha256":record["record_sha256"],
                "files_sha256":{**record["files_sha256"],runner.JOB_FILENAME:sha_file(output/runner.JOB_FILENAME)},
                "checkpoint_sizes":{name:(output/name).stat().st_size for name in subject.TENSOR_FILES}}
            subject.read_completed_job(output,receipt,request,value["rows"],self.plan,self.registry)
            with (output/"best_validation_predictions.json").open("a") as stream:stream.write(" ")
            with self.assertRaisesRegex(ValueError,"json_hash_mismatch"):
                subject.read_completed_job(output,receipt,request,value["rows"],self.plan,self.registry)


if __name__ == "__main__":
    unittest.main()
