import argparse
import csv
import getpass
import io
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

from .context import ROOT, read_json, timestamp, write_json


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"


def run_id(campaign, encoder, model, site, expansion, k, seed):
    return f"{campaign}_{encoder}_{model}_{site}_e{expansion}_k{k}_s{seed}"


def jobs():
    config = read_json(CONFIG_PATH)
    grid = config["grid"]
    reference = grid["reference_seed"]
    seeds = [reference] + [seed for seed in grid["seeds"] if seed != reference]
    deployed = [("fmcib", "layer2"), ("vista", "stage2"), ("vista", "stage3")]
    order = [("trained", *site) for site in deployed] + [("trained", "fmcib", "layer1_context")] + [("random", "fmcib", "layer1_context")] + [("random", *site) for site in deployed]
    values = []
    for priority, seed in enumerate(seeds):
        for encoder, model, site in order:
            for k, expansion in itertools.product(grid["k"], grid["expansion"]):
                values.append({
                    "run_id": run_id(config["campaign"], encoder, model, site, expansion, k, seed),
                    "model": model, "site": site, "seed": seed, "k": k, "expansion": expansion, "encoder": encoder,
                    "matrix_directory": "matrices" if encoder == "trained" else "matrices_random",
                    "priority": [priority, order.index((encoder, model, site))],
                })
    diagnostic = config["budget_diagnostic"]
    for index, variant in enumerate(diagnostic["variants"]):
        for model, site, expansion, k in variant["configurations"]:
            for seed in diagnostic["seeds"]:
                values.append({
                    "run_id": f"{run_id(config['campaign'], 'trained', model, site, expansion, k, seed)}_{variant['name']}",
                    "model": model, "site": site, "seed": seed, "k": k, "expansion": expansion, "encoder": "trained",
                    "matrix_directory": "matrices", "priority": [0, 10 + index],
                    "variant": variant["name"], "training": variant["training"],
                })
    production = config["production"]
    trainings = {variant["name"]: variant["training"] for variant in diagnostic["variants"]}
    existing = {job["run_id"] for job in values}
    local = {tuple(item) for item in production["local"]}
    for index, (name, choice) in enumerate(production["sites"].items()):
        model, site = name.split(":")
        variant = choice["variant"]
        for encoder in ["trained", "random"]:
            for seed in seeds:
                identifier = f"{run_id(config['campaign'], encoder, model, site, choice['expansion'], choice['k'], seed)}_{variant}"
                if identifier in existing:
                    continue
                values.append({
                    "run_id": identifier, "model": model, "site": site, "seed": seed, "k": choice["k"], "expansion": choice["expansion"], "encoder": encoder,
                    "matrix_directory": "matrices" if encoder == "trained" else "matrices_random",
                    "priority": [0, (20 if encoder == "trained" else 30) + index],
                    "variant": variant, "training": trainings[variant], "production": True,
                    "host": "local" if (model, site, encoder) in local else "mayo",
                })
    return values


def queue_name(job):
    first = 0 if job.get("production") else int(job["encoder"] != "trained")
    return f"{first}{job['priority'][0]:02d}{job['priority'][1]:02d}_{job['run_id']}.json"


def write_jobs(root, models, variants_only=False, host=None):
    for job in jobs():
        if job["model"] not in models or (variants_only and "variant" not in job) or (host and job.get("host") != host):
            continue
        path = Path(root) / "jobs" / queue_name(job)
        if path.exists():
            assert read_json(path) == job, path
        else:
            write_json(path, job)


def foreign_processes(gpu):
    query = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    uuid, utilization, memory = [value.strip() for value in next(csv.reader(io.StringIO(query.stdout)))]
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    user = getpass.getuser()
    foreign = []
    for row in csv.reader(io.StringIO(processes.stdout)):
        if not row or row[0].strip() != uuid:
            continue
        owner = subprocess.run(["ps", "-o", "user=", "-p", row[1].strip()], capture_output=True, text=True).stdout.strip()
        if owner != user:
            foreign.append({"pid": int(row[1]), "user": owner})
    return foreign, {"utilization": float(utilization), "memory_used_mib": int(memory)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--write-jobs", action="store_true")
    parser.add_argument("--models", default="fmcib,vista")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--variants-only", action="store_true")
    parser.add_argument("--host", choices=["local", "mayo"], help="write only production jobs assigned to this host")
    args = parser.parse_args()
    root = args.root
    if args.write_jobs:
        write_jobs(root, args.models.split(","), args.variants_only, args.host)
        return
    state_path = root / "workers" / f"{args.worker}.json"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "SCLVMI_SAE_CAMPAIGN_CONFIG": "configs/sae_interpretation.json"}
    if args.local:
        os.environ.update(environment)
        from .sae_campaign_train import fit_job
    with FileLock(str(state_path.with_suffix(".lock")), timeout=0):
        while True:
            status = {"worker": args.worker, "gpu": args.gpu, "pid": os.getpid(), "updated_at": timestamp()}
            if (root / "pause").exists():
                write_json(state_path, {**status, "state": "paused"})
                time.sleep(5)
                continue
            foreign, telemetry = ([], {}) if args.local else foreign_processes(args.gpu)
            if foreign:
                write_json(state_path, {**status, **telemetry, "state": "waiting_gpu", "foreign": foreign})
                time.sleep(15)
                continue
            chosen, claim, pending = None, None, 0
            for path in sorted((root / "jobs").glob("*.json")):
                job = read_json(path)
                run = root / "runs" / job["run_id"]
                if (run / "result.json").exists() or (run / "failure.json").exists():
                    continue
                pending += 1
                if not (root / job["matrix_directory"] / job["model"] / job["site"] / "matrix.json").exists():
                    continue
                run.mkdir(parents=True, exist_ok=True)
                lock = FileLock(str(run / "assignment.lock"), timeout=0)
                try:
                    lock.acquire()
                except Timeout:
                    continue
                if (run / "result.json").exists():
                    lock.release()
                    continue
                chosen, claim = (path, job), lock
                break
            if chosen is None:
                if pending == 0 and (root / "jobs_closed").exists():
                    write_json(state_path, {**status, "state": "finished"})
                    return
                write_json(state_path, {**status, **telemetry, "state": "waiting_jobs", "pending": pending})
                time.sleep(10)
                continue
            path, job = chosen
            run = root / "runs" / job["run_id"]
            if args.local:
                write_json(state_path, {**status, "state": "running", "run_id": job["run_id"], "updated_at": timestamp()})
                with FileLock(str(run / "execution.lock"), timeout=0):
                    result = fit_job(root, job, args.worker)
                print("RESULT", job["run_id"], result.get("state"), flush=True)
                claim.release()
                continue
            log_path = run / f"training_{time.time_ns()}.log"
            invocation = [sys.executable, "-u", "-m", "sclvmi.sae_campaign_train", "--root", str(root), "--job", str(path), "--worker", args.worker]
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(invocation, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
                while process.poll() is None:
                    write_json(state_path, {**status, **telemetry, "state": "running", "run_id": job["run_id"], "child_pid": process.pid, "updated_at": timestamp()})
                    time.sleep(5)
            if process.returncode != 0:
                failure = {"run_id": job["run_id"], "worker": args.worker, "exit_code": process.returncode, "log": log_path.name, "created_at": timestamp()}
                write_json(run / "failure.json", failure)
            claim.release()


if __name__ == "__main__":
    main()
