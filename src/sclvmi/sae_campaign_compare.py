from pathlib import Path

from filelock import FileLock

from .context import read_json, timestamp, write_json
from .sae_campaign import campaign_root, configuration


def preferred(candidates, margin):
    qualified = [row for row in candidates if row["feature_checks_pass"]]
    available = qualified or candidates
    best_fvu = min(row["development"]["fvu"] for row in available)
    equivalent = [row for row in available if row["development"]["fvu"] <= best_fvu + margin]
    return min(equivalent, key=lambda row: (row["request"]["k"], row["request"]["dict_size"], row["development"]["fvu"]))


def expand_queue(root=None):
    root = campaign_root(root)
    config = configuration()
    with FileLock(str(root / "queue_selection.lock"), timeout=5):
        queue = read_json(root / "jobs.json", synchronized=True)
        added = []
        for model, sites in config["models"].items():
            for site in sites:
                if (root / "decisions" / f"{model}-{site}-coverage-adjustment.json").exists():
                    continue
                initial = [job for job in queue["jobs"] if job["model"] == model and job["site"] == site and job["phase"] == "initial"]
                assert len(initial) == len(config["search"]["initial_k"])
                if not all((root / "runs" / job["run_id"] / "result.json").exists() for job in initial):
                    continue
                results = [read_json(root / "runs" / job["run_id"] / "result.json") for job in initial]
                strongest = preferred(results, config["selection"]["fvu_equivalence_margin"])
                lower = any(result["feature_checks_pass"] for result in results)
                alternative_k = config["search"]["lower_k"][0] if lower else config["search"]["higher_k"][0]
                alternatives = [(config["search"]["comparison_expansion"], strongest["request"]["k"], "Test dictionary capacity at the preferred initial activity level"), (config["search"]["initial_expansion"], alternative_k, "Test a sparser representation after initial qualification" if lower else "Test additional active features after initial quality criteria were unmet")]
                for expansion, k, reason in alternatives:
                    run_id = f"20261009_{model}_{site}_e{expansion}_k{k}_s2025"
                    if any(job["run_id"] == run_id for job in queue["jobs"]):
                        continue
                    decision = {"created_at": timestamp(), "model": model, "site": site, "run_id": run_id, "reason": reason, "reference": strongest["run_id"], "initial_evidence": [{"run_id": result["run_id"], "metrics": result["development"], "checks": result["feature_checks"]} for result in results], "test_used": False}
                    write_json(root / "decisions" / f"{run_id}.json", decision)
                    job = {"run_id": run_id, "model": model, "site": site, "expansion": expansion, "k": k, "seed": 2025, "pool": initial[0]["pool"], "phase": "comparison", "reason": reason, "decision": f"decisions/{run_id}.json"}
                    queue["jobs"].append(job)
                    added.append(run_id)
        if added:
            queue["updated_at"] = timestamp()
            write_json(root / "jobs.json", queue)
        return added


def select_site_candidates(root=None):
    root = campaign_root(root)
    config = configuration()
    with FileLock(str(root / "queue_selection.lock"), timeout=5):
        queue = read_json(root / "jobs.json", synchronized=True)
        added = []
        for model, sites in config["models"].items():
            for site in sites:
                candidates = [job for job in queue["jobs"] if job["model"] == model and job["site"] == site and job["phase"] in ["initial", "comparison", "adjustment"] and job.get("enabled", True)]
                if len(candidates) < 4:
                    continue
                decision_path = root / "selection" / f"{model}-{site}.json"
                if decision_path.exists():
                    continue
                if not all((root / "runs" / job["run_id"] / "result.json").exists() for job in candidates):
                    continue
                evidence = []
                awaiting_evaluation = False
                for job in candidates:
                    run = root / "runs" / job["run_id"]
                    result = read_json(run / "result.json")
                    downstream = read_json(run / "downstream.json") if (run / "downstream.json").exists() else None
                    if result["feature_checks_pass"] and downstream is None:
                        awaiting_evaluation = True
                    evidence.append({**result, "downstream": downstream, "qualified": bool(result["feature_checks_pass"] and downstream and downstream["checks_pass"])})
                if awaiting_evaluation:
                    continue
                qualified = [row for row in evidence if row["qualified"]]
                if not qualified:
                    write_json(root / "decisions" / f"{model}-{site}-needs-review.json", {"state": "needs_method_review", "model": model, "site": site, "evidence": evidence, "test_used": False, "updated_at": timestamp()})
                    continue
                chosen = preferred(qualified, config["selection"]["fvu_equivalence_margin"])
                template = next(job for job in candidates if job["run_id"] == chosen["run_id"])
                replicas = []
                for seed in config["search"]["seeds"]:
                    run_id = template["run_id"].replace(f"_s{template['seed']}", f"_s{seed}")
                    replicas.append(run_id)
                    if any(job["run_id"] == run_id for job in queue["jobs"]):
                        continue
                    queue["jobs"].append({**template, "run_id": run_id, "seed": seed, "phase": "replication", "reason": "Repeat the qualified site configuration with an independent training seed", "decision": f"selection/{model}-{site}.json"})
                    added.append(run_id)
                queue["updated_at"] = timestamp()
                write_json(root / "jobs.json", queue)
                write_json(decision_path, {"state": "selected_for_replication", "model": model, "site": site, "selected_run": chosen["run_id"], "replicas": replicas, "rule": "Require feature and fixed-classifier checks, then prefer lower activity and smaller dictionaries within the configured FVU equivalence margin", "evidence": evidence, "created_at": timestamp(), "test_used": False})
        if added:
            queue["updated_at"] = timestamp()
            write_json(root / "jobs.json", queue)
        return added


def adjust_coverage_conflicts(root=None):
    root = campaign_root(root)
    config = configuration()
    with FileLock(str(root / "queue_selection.lock"), timeout=5):
        queue = read_json(root / "jobs.json", synchronized=True)
        added = []
        for model, sites in config["models"].items():
            for site in sites:
                initial = sorted([job for job in queue["jobs"] if job["model"] == model and job["site"] == site and job["phase"] == "initial"], key=lambda row: row["k"])
                if len(initial) != 2 or not all((root / "runs" / job["run_id"] / "result.json").exists() for job in initial):
                    continue
                results = [read_json(root / "runs" / job["run_id"] / "result.json") for job in initial]
                lower, higher = results
                if not (lower["feature_checks"]["inactive"] and not lower["feature_checks"]["cosine"] and higher["feature_checks"]["cosine"] and not higher["feature_checks"]["inactive"]):
                    continue
                decision_path = root / "decisions" / f"{model}-{site}-coverage-adjustment.json"
                if decision_path.exists():
                    changed = False
                    for job in queue["jobs"]:
                        if job["model"] == model and job["site"] == site and job["phase"] == "comparison" and job.get("enabled", True):
                            if not (root / "runs" / job["run_id"] / "status.json").exists():
                                job["enabled"] = False
                                job["superseded_by"] = decision_path.relative_to(root).as_posix()
                                changed = True
                    if changed:
                        write_json(root / "jobs.json", queue)
                    continue
                lower_k, higher_k = initial[0]["k"], initial[1]["k"]
                quarter = lower_k + (higher_k - lower_k) // 4
                middle = (lower_k + higher_k) // 2
                alternatives = [(4, quarter), (4, middle), (8, quarter)]
                decision = {"created_at": timestamp(), "model": model, "site": site, "reason": "Lower activity preserves feature coverage but misses cosine fidelity; higher activity meets fidelity with excessive inactive features. Test intermediate activity and capacity at that activity.", "initial_evidence": results, "alternatives": alternatives, "test_used": False}
                for job in queue["jobs"]:
                    if job["model"] == model and job["site"] == site and job["phase"] == "comparison":
                        status_path = root / "runs" / job["run_id"] / "status.json"
                        if not status_path.exists():
                            job["enabled"] = False
                            job["superseded_by"] = str(decision_path.relative_to(root))
                for expansion, k in alternatives:
                    run_id = f"20261009_{model}_{site}_e{expansion}_k{k}_s2025"
                    if not any(job["run_id"] == run_id for job in queue["jobs"]):
                        queue["jobs"].append({"run_id": run_id, "model": model, "site": site, "expansion": expansion, "k": k, "seed": 2025, "pool": initial[0]["pool"], "phase": "adjustment", "reason": decision["reason"], "decision": str(decision_path.relative_to(root))})
                        added.append(run_id)
                write_json(root / "jobs.json", queue)
                write_json(decision_path, decision)
        return added


if __name__ == "__main__":
    print("ADDED_COMPARISONS", expand_queue(), flush=True)
