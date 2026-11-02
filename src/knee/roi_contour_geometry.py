"""Experimental landmark-guided gradient paths and native rectangular crops.

This is a deterministic edge hypothesis, not validated bone segmentation.
KNEEL landmarks are frozen; all resampling and filtering is for localization.
The external supervisor extracts only a slice of the original pixels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from knee.roi_external_candidate import load_predictions
from knee.roi_landmark_geometry import landmark_box

CONFIG = {
    "version": "kneel_guided_gradient_contours_v1",
    "working_spacing_mm": 0.5, "smoothing_sigma_mm": 0.6,
    "surface_search_mm": 3.0, "lateral_search_fraction": 0.10,
    "lateral_margin_mm": 5.0, "prior_cost": 0.20,
    "transition_cost_per_step_squared": 0.15, "max_step_displacement_mm": 2.0,
    "compact_aspect_ratio": 2.4,
    "compact_vertical_rule": "max(width_mm/2.4, 2*surface_y_span_mm+2*row_spacing_mm)",
    "source_note": "Compact ratio is a transfer hypothesis from KOACAD's 480x200 joint-space ROI; it is not validated for progression prediction.",
    "abstention": "invalid source, points, degenerate signal, or requested bounds outside original half; no clipping",
    "quality_decision": "none; path energy is not calibrated anatomical or artifact confidence",
}


def minimum_cost_path(cost, transition_weight=.15, max_step=4):
    """Exact dynamic programming, columns progress, rows are displacement states."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not cost.size or not np.isfinite(cost).all():
        raise ValueError("invalid_path_cost")
    n, length = cost.shape
    prev = cost[:, 0].copy()
    back = np.empty((n, length), dtype=np.int32)
    indices = np.arange(n)
    step = indices[:, None]-indices[None, :]
    penalty = transition_weight*step.astype(float)**2
    penalty[np.abs(step) > max_step] = np.inf
    for col in range(1, length):
        options = prev[:, None]+penalty
        back[:, col] = np.argmin(options, axis=0)
        prev = options[back[:, col], indices]+cost[:, col]
    path = np.empty(length, dtype=np.int32)
    path[-1] = np.argmin(prev)
    for col in range(length-1, 0, -1):
        path[col-1] = back[path[col], col]
    return path


def _trace(gradient, prior_xy_mm, axis, sign, search_mm, step_mm):
    count = math.ceil(search_mm/step_mm)
    offsets = np.arange(-count, count+1)*step_mm
    coords = np.broadcast_to(prior_xy_mm, (len(offsets), *prior_xy_mm.shape)).copy()
    coords[:, :, axis] += offsets[:, None]
    gx = (coords[:, :, 0]/step_mm).astype(np.float32)
    gy = (coords[:, :, 1]/step_mm).astype(np.float32)
    signal = sign*cv2.remap(gradient.astype(np.float32), gx, gy,
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    scale = float(np.percentile(np.abs(signal), 95))
    if scale <= np.finfo(float).eps:
        raise ValueError("no_gradient_evidence")
    score = np.clip(signal/scale, -1, 1)
    prior = CONFIG["prior_cost"]*(offsets/max(search_mm,step_mm))**2
    path = minimum_cost_path(-score+prior[:,None], CONFIG["transition_cost_per_step_squared"],
                             round(CONFIG["max_step_displacement_mm"]/step_mm))
    trace = coords[path,np.arange(len(prior_xy_mm))]
    return trace, {"median_signed_gradient_score":float(np.median(score[path,np.arange(len(path))])),
                   "maximum_displacement_mm":float(np.max(np.abs(offsets[path])))}


def contour_boxes(raw, landmarks, dr, dc, photometric):
    """Construct two boxes under a single fixed edge configuration."""
    base = landmark_box(landmarks, raw.shape, dr, dc)
    if base["status"] != "candidate":
        raise ValueError(base["error_code"])
    if photometric not in ("MONOCHROME1","MONOCHROME2"):
        raise ValueError("unsupported_photometric")
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("invalid_native_pixels")
    lo, hi = np.percentile(values,[1,99])
    if hi <= lo:
        raise ValueError("constant_signal")
    values=np.clip((values-lo)/(hi-lo),0,1)
    if photometric == "MONOCHROME1": values=1-values
    step=CONFIG["working_spacing_mm"]
    nr,nc=math.ceil((raw.shape[0]-1)*dr/step)+1,math.ceil((raw.shape[1]-1)*dc/step)+1
    # Physical positions are explicit; resize dimensions never stand in for spacing.
    xx,yy=np.meshgrid(np.arange(nc)*step/dc,np.arange(nr)*step/dr)
    working=cv2.remap(values.astype(np.float32),xx.astype(np.float32),yy.astype(np.float32),
                      cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    working=gaussian_filter(working,CONFIG["smoothing_sigma_mm"]/step)
    gy,gx=np.gradient(working,step,step)
    points=np.asarray(landmarks,dtype=float)*[dc,dr]
    surface_paths=[]
    traces={}
    scores={}
    for name,group,sign in (("tibia",points[:9],1),("femur",points[9:],-1)):
        # Coincident x coordinates are averaged only for the working interpolation;
        # every original landmark remains in the final enclosing geometry.
        xs=np.unique(group[:,0])
        if len(xs)<2: raise ValueError("collapsed_surface_x")
        ys=np.array([np.mean(group[group[:,0]==x,1]) for x in xs])
        samplex=np.linspace(xs[0],xs[-1],math.ceil((xs[-1]-xs[0])/step)+1)
        prior=np.column_stack((samplex,np.interp(samplex,xs,ys)))
        path,details=_trace(gy,prior,1,sign,CONFIG["surface_search_mm"],step)
        surface_paths.append(path)
        traces[name]=path
        scores[name]=details
    width_mm=float(np.ptp(points[:,0]))
    _,_,by0,by1=base['native_half_box']
    lateral=[]
    for name,group in (("femur",points[9:]),("tibia",points[:9])):
        for side,idx,sign in (("left",np.argmin(group[:,0]),1),("right",np.argmax(group[:,0]),-1)):
            x,y=group[idx]
            first,last=(by0*dr,y) if name=='femur' else (y,by1*dr-dr)
            if last<=first: raise ValueError("empty_lateral_search")
            sampley=np.linspace(first,last,math.ceil((last-first)/step)+1)
            prior=np.column_stack((np.full(len(sampley),x),sampley))
            path,details=_trace(gx,prior,0,sign,CONFIG['lateral_search_fraction']*width_mm,step)
            key=name+'_'+side
            traces[key]=path
            scores[key]=details
            lateral.append(path)
    all_x=np.concatenate([points[:,0],*(p[:,0] for p in lateral)])
    left=float(all_x.min()-CONFIG['lateral_margin_mm'])
    right=float(all_x.max()+CONFIG['lateral_margin_mm'])
    all_y=np.concatenate([points[:,1],*(p[:,1] for p in surface_paths)])
    sy0,sy1=float(all_y.min()),float(all_y.max())
    center=(sy0+sy1)/2
    height=max((right-left)/CONFIG['compact_aspect_ratio'],2*(sy1-sy0)+2*dr)
    proposed={'contour_full_height':[left/dc,right/dc,by0,by1],
              'contour_compact':[left/dc,right/dc,(center-height/2)/dr,(center+height/2)/dr]}
    results={}
    for name,box in proposed.items():
        a,b,c,d=math.floor(box[0]),math.ceil(box[1]),math.floor(box[2]),math.ceil(box[3])
        inside=0<=a<b<=raw.shape[1] and 0<=c<d<=raw.shape[0]
        results[name]=dict(status='candidate' if inside else 'abstain',
                          error_code='' if inside else 'contour_box_outside_frozen_half',
                          requested_native_half_box=box,native_half_box=[a,b,c,d],
                          surface_vertical_fraction=[(sy0/dr-c)/(d-c),(sy1/dr-c)/(d-c)],
                          crop_width_mm=(b-a)*dc,crop_height_mm=(d-c)*dr)
    return results,dict(gradient_path_metrics=scores,
        traces_native_half_xy={k:(v/[dc,dr]).tolist() for k,v in traces.items()},
        landmark_span_mm=width_mm,surface_envelope_y_mm=[sy0,sy1],
        coverage_certified=False,artifact_absence_certified=False)


def run(kneel_path,verified_dir,output):
    kneel_path,verified_dir,output=map(lambda p:Path(p).resolve(),(kneel_path,verified_dir,output))
    if output.exists(): raise FileExistsError('new_output_directory_required')
    if any(output==p or output in p.parents or p in output.parents for p in (kneel_path.parent,verified_dir)):
        raise ValueError('output_must_be_separate')
    evidence_bytes=(verified_dir/'results.json').read_bytes()
    evidence=json.loads(evidence_bytes)
    aliases=[r['knee_alias'] for r in evidence['rows']]
    if len(aliases)!=20 or len(set(aliases))!=20 or evidence['summary']['exact_baseline_replays']!=20:
        raise ValueError('verified_historical20_required')
    predictions,upstream,pred_sha=load_predictions(kneel_path,aliases)
    if pred_sha!=evidence['summary']['external_predictions_sha256']:
        raise ValueError('upstream_predictions_hash_mismatch')
    module_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(__file__).with_name('roi_landmark_geometry.py'))}
    output.mkdir(parents=True)
    config=dict(CONFIG,recorded_before_execution_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=module_hashes)
    (output/'configuration_before_execution.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    rows={'contour_full_height':[],'contour_compact':[]}
    traces=[]
    for reference in evidence['rows']:
        alias=reference['knee_alias']
        source=predictions[alias]
        raw=np.load(verified_dir/'halves'/(alias+'.npy'),allow_pickle=False)
        before=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        if before!=source['source_half_sha256']: raise ValueError('source_half_hash_mismatch')
        points=source['metadata']['landmarks_native_half_xy']
        try:
            boxes,trace=contour_boxes(raw,points,reference['row_spacing_mm'],reference['column_spacing_mm'],reference['photometric'])
        except ValueError as exc:
            boxes={k:dict(status='abstain',error_code=str(exc)) for k in rows}
            trace={}
        if before!=hashlib.sha256(raw.tobytes()).hexdigest(): raise ValueError('native_source_changed')
        traces.append(dict(knee_alias=alias,**trace))
        for mode in rows:
            value=boxes[mode]
            rows[mode].append(dict(knee_alias=alias,source_half_sha256=before,
                status=value['status'],error_code=value['error_code'],
                native_half_box=value.get('native_half_box'),metadata=dict(value,**trace)))
        print(alias,{k:boxes[k]['status'] for k in rows},flush=True)
    for p in (Path(__file__),Path(__file__).with_name('roi_landmark_geometry.py')):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=module_hashes[p.name]: raise RuntimeError('code_changed')
    if hashlib.sha256(kneel_path.read_bytes()).hexdigest()!=pred_sha or (verified_dir/'results.json').read_bytes()!=evidence_bytes:
        raise RuntimeError('source_changed')
    for mode,values in rows.items():
        manifest=dict(candidate=CONFIG['version']+'_'+mode,configuration=config,
                      upstream_manifest=upstream,frozen_predictions_sha256=pred_sha,
                      experimental_only=True,acceptability_assigned=False,
                      source_pixel_modification=False,model_training_executed=False,
                      outcome_data_loaded=False,step_4_status='open')
        (output/(mode+'_predictions.json')).write_text(json.dumps(dict(manifest=manifest,rows=values),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:sum(r['status']=='candidate' for r in v) for k,v in rows.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kneel-predictions',required=True,type=Path)
    p.add_argument('--verified-kneel',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    run(a.kneel_predictions,a.verified_kneel,a.output)
