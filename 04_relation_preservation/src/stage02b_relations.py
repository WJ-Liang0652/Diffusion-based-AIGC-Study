"""Independent 02B protocol/queue, reusing the unchanged stage02 GPU and bbox functions."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import stage02_relations as shared
from PIL import Image, ImageDraw

ROOT = shared.ROOT
CONFIG = ROOT/"04_relation_preservation/configs/stage02b.json"
PROTOCOL = ROOT/"04_relation_preservation/notes/stage02b_protocol.md"


def queue(cfg):
    return [(prompt, seed) for prompt in cfg["prompt_order"] for seed in cfg["seed_candidates"]]


def records(out):
    return json.loads((out/"screening.json").read_text()) if (out/"screening.json").exists() else []


def next_candidate(cfg, rows):
    if any(row["qualified"] for row in rows):
        return None
    candidates = queue(cfg)
    assert [(r["prompt_id"],r["seed"]) for r in rows] == candidates[:len(rows)], "Recorded queue order violated"
    return candidates[len(rows)] if len(rows)<len(candidates) else None


def effective(cfg, prompt_id):
    return {**cfg,"prompt_id":prompt_id,"prompt":cfg["prompts"][prompt_id],
            "output_dir":cfg["output_dir"]+"/"+prompt_id}


def fingerprints(config):
    paths=[config,PROTOCOL,Path(__file__).resolve(),Path(shared.__file__).resolve(),
           Path(__file__).with_name("stage01_sdxl.py"),Path(__file__).with_name("attention_processor.py")]
    return {str(p.relative_to(ROOT)):shared.sha(p) for p in paths}


def consumed(out):
    return sum(json.loads(p.read_text()).get("sampling_seconds",0) for p in out.rglob("run_metrics.json"))


def interpretations(annotation):
    if not annotation["objects_alive"]:
        return "D_object_missing_failure"
    if annotation["identity_or_instances_uncertain"] or annotation["invalid_truncated"] or annotation["uncertain"]:
        return "D_uncertain_or_invalid"
    if annotation["new_relation"]!="clear_success":
        return "A_new_target_not_repaired"
    if annotation["old_relation"]=="clear_success":
        return "B_joint_success"
    return "C_degradation_candidate"


def annotate(cfg,out,args):
    rows=records(out)
    if args.kind=="baseline":
        assert next_candidate(cfg,rows)==(args.prompt_id,args.seed), "Candidate order/early stop violated"
    else:
        selected=next(row for row in rows if row["qualified"])
        assert (args.prompt_id,args.seed)==(selected["prompt_id"],selected["seed"])
        metrics=json.loads((out/args.prompt_id/"intervention/run_metrics.json").read_text())
        assert metrics["completed"] and metrics["guided_completed"] and metrics["restoration"]["passed"]
    subout=out/args.prompt_id
    shared.annotate(effective(cfg,args.prompt_id),subout,args.annotation,args.kind,args.seed)
    if args.kind=="baseline":
        row=json.loads((subout/"screening.json").read_text())[-1]
        rows.append({**row,"prompt_id":args.prompt_id,"candidate_index":len(rows)})
        shared.dump_json(out/"screening.json",rows)
        if row["qualified"]:
            shared.dump_json(out/"selected_baseline.json",rows[-1])
    else:
        result=json.loads((subout/"relation_results.json").read_text())
        result["prompt_id"]=args.prompt_id
        result["interpretation"]=interpretations(result["guided"])
        shared.dump_json(out/"relation_results.json",result)
    # Draw all reliably marked instances without forcing a unique identity.
    folder=subout/f"seed{args.seed}" if args.kind=="baseline" else subout/"intervention"
    data=json.loads((folder/f"{args.kind}_annotation.json").read_text())
    image=Image.open(folder/f"{args.kind}_annotated.png").convert("RGB")
    draw=ImageDraw.Draw(image)
    for key,color in [("A","red"),("B","blue")]:
        for i,obj in enumerate(data["objects"][key].get("instances",[])):
            box=obj.get("bbox_pixels")
            if box:
                draw.rectangle(box,outline=color,width=4)
                draw.text((box[0]+5,box[1]+5),f"{key}{i+1}",fill=color,stroke_width=2,stroke_fill="white")
    image.save(folder/f"{args.kind}_annotated.png")


def summary(cfg,out):
    rows=records(out)
    runs=[json.loads(p.read_text()) for p in out.rglob("run_metrics.json")]
    selected=next((r for r in rows if r["qualified"]),None)
    result=json.loads((out/"relation_results.json").read_text()) if (out/"relation_results.json").exists() else None
    metrics={"stage":"02B","purpose":"diagnostic_start_preparation_and_pipeline_validation_not_statistical_evaluation",
             "baseline_count":len(rows),"maximum_baselines":16,"selected":selected,
             "early_stop":selected is not None and len(rows)<16,"screening":rows,
             "relation_results":result,"sampling_seconds":sum(r["sampling_seconds"] for r in runs),
             "summed_generation_invocation_seconds":sum(r["total_seconds"] for r in runs),
             "peak_allocated_mib":max((r["peak_allocated_mib"] for r in runs),default=0),
             "peak_reserved_mib":max((r["peak_reserved_mib"] for r in runs),default=0)}
    if selected:
        p=out/selected["prompt_id"]/"intervention/run_metrics.json"
        metrics["intervention_metrics"]=json.loads(p.read_text()) if p.exists() else None
    shared.dump_json(out/"stage02b_summary.json",metrics)
    annotations=[json.loads((ROOT/r["annotation"]).read_text()) for r in rows]
    columns=["prompt_id","seed","qualified","dx","dy","old_relation","new_relation","uncertain","objects_alive","invalid_truncated"]
    table=[{k:(r[k] if k in r else a[k]) for k in columns} for r,a in zip(rows,annotations)]
    shared.csv_rows(out/"screening.csv",table)
    if result:
        shared.csv_rows(out/"relations.csv",[{"branch":name,"dx":a["dx"],"dy":a["dy"],"old_relation":a["old_relation"],
                  "new_relation":a["new_relation"],"joint_success":a["joint_success"],"objects_alive":a["objects_alive"],
                  "uncertain":a["uncertain"],"invalid_truncated":a["invalid_truncated"]} for name,a in
                  [("baseline",result["baseline"]),("guided",result["guided"])]])
    if rows:
        cols=min(4,len(rows)); height=((len(rows)+cols-1)//cols)*544
        canvas=Image.new("RGB",(cols*512,height),"white");draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows):
            x=(i%cols)*512;y=(i//cols)*544
            im=Image.open(out/r["prompt_id"]/f"seed{r['seed']}"/"baseline_annotated.png")
            canvas.paste(im.resize((512,512),Image.Resampling.LANCZOS),(x,y+32))
            draw.text((x+8,y+8),f"{r['prompt_id']} seed {r['seed']} {'qualified' if r['qualified'] else 'rejected'}",fill="black")
        canvas.save(out/"screening_annotated.png")
    if result:
        subout=out/result["prompt_id"]
        for annotated in [False,True]:
            name="baseline_annotated.png" if annotated else "baseline.png"
            guided="guided_annotated.png" if annotated else "guided.png"
            canvas=Image.new("RGB",(2048,1056),"white");draw=ImageDraw.Draw(canvas)
            for i,(label,path) in enumerate([("baseline",subout/f"seed{result['seed']}"/name),("guided",subout/"intervention"/guided)]):
                canvas.paste(Image.open(path),(i*1024,32));draw.text((i*1024+8,8),label,fill="black")
            canvas.save(out/("baseline_guided_annotated.png" if annotated else "baseline_guided_original.png"))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ["screening","relation_results","intervention_metrics"]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=CONFIG)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ["baseline","annotate"]:
        s=sub.add_parser(mode);s.add_argument("--prompt-id",choices=["P1","P2"],required=True);s.add_argument("--seed",type=int,required=True)
        if mode=="annotate":
            s.add_argument("--kind",choices=["baseline","guided"],required=True);s.add_argument("--annotation",type=Path,required=True)
    sub.add_parser("intervene");sub.add_parser("summary")
    args=p.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg["output_dir"]
    hashes=fingerprints(args.config.resolve());shared.lock_protocol(out,hashes)
    if args.mode=="annotate":
        annotate(cfg,out,args);return
    if args.mode=="summary":
        summary(cfg,out);return
    if args.mode=="baseline":
        assert next_candidate(cfg,records(out))==(args.prompt_id,args.seed), "Queue order or early-stop gate failed"
        prompt_id=args.prompt_id
    else:
        selected=next(r for r in records(out) if r["qualified"]);prompt_id=selected["prompt_id"]
    runtime=effective(cfg,prompt_id);subout=out/prompt_id;subout.mkdir(exist_ok=True)
    already=consumed(out);assert already<cfg["gpu_sampling_stop_seconds"]
    start=time.perf_counter()
    metrics={"mode":args.mode,"prompt_id":prompt_id,"timings":{},"sampling_seconds":0.0,"sampling_seconds_before":already,
             "completed":False,"environment":shared.environment(),"fingerprints":hashes,"command_argv":sys.argv}
    budget=shared.SamplingBudget(runtime,metrics,already)
    try:
        if args.mode=="baseline":
            shared.baseline(runtime,subout,args.seed,hashes,metrics,budget)
        else:
            shared.intervene(runtime,subout,hashes,metrics,budget)
        metrics["completed"]=True
    except Exception:
        metrics["traceback"]=traceback.format_exc();raise
    finally:
        metrics["total_seconds"]=time.perf_counter()-start
        metrics["peak_allocated_mib"]=max((t["peak_allocated_mib"] for t in metrics["timings"].values()),default=0)
        metrics["peak_reserved_mib"]=max((t["peak_reserved_mib"] for t in metrics["timings"].values()),default=0)
        shared.dump_json(Path(metrics.get("artifact_dir",subout))/"run_metrics.json",metrics)


if __name__=="__main__":
    main()
