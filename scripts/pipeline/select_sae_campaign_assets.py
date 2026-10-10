import argparse
from pathlib import Path

from sclvmi.context import read_json, sha256, timestamp, write_json
from sclvmi.sae_campaign import campaign_root, configuration


def select(root, sites, rationale):
    destination = root / "selection.json"
    assert not destination.exists()
    map_checks = [read_json(path) for path in (root / "viewer").glob("validation_*.json")]
    checked_maps = {row["run_id"] for receipt in map_checks if receipt["state"] == "passed" and "response_statistics_checked" in receipt and receipt["response_statistics_checked"] for row in receipt["records"]}
    raw_checks = [read_json(path) for path in (root / "smoke").glob("raw_prediction_*.json")]
    checked_raw = {row["run_id"] for receipt in raw_checks if receipt["state"] == "passed" for row in receipt["records"]}
    selected = []
    for specification in sites:
        parts = specification.split(":")
        assert len(parts) in [3, 4]
        model, site, role = parts[:3]
        decision_name = parts[3] if len(parts) == 4 else f"{model}-{site}"
        assert Path(decision_name).name == decision_name
        assert role in configuration()["selection"]["roles"]
        decision_path = root / "selection" / f"{decision_name}.json"
        decision = read_json(decision_path)
        assert decision["model"] == model and decision["site"] == site
        replicas = decision["replicas"]
        assert len(replicas) == 3
        diagnostic_path = root / "diagnostics" / decision_name / "result.json"
        diagnostic = read_json(diagnostic_path)
        assert diagnostic["state"] == "completed" and set(diagnostic["runs"]) == set(replicas)
        assert len(diagnostic["comparisons"]) == 3
        for run_id in replicas:
            run = root / "runs" / run_id
            result = read_json(run / "result.json")
            downstream = read_json(run / "downstream.json")
            responses = read_json(root / "viewer/responses" / run_id / "result.json")
            assert result["feature_checks_pass"] and downstream["checks_pass"]
            assert responses["state"] == "completed" and responses["cases"] == 128
            assert result["dictionary_sha256"] == downstream["dictionary_sha256"] == responses["dictionary_sha256"] == sha256(run / "dictionary.pt")
            assert run_id in checked_maps and run_id in checked_raw
        selected.append({"model": model, "site": site, "role": role, "selected_run": decision["selected_run"], "replicas": replicas, "site_selection": decision_path.relative_to(root).as_posix(), "diagnostics": diagnostic_path.relative_to(root).as_posix(), "site_selection_sha256": sha256(decision_path), "diagnostics_sha256": sha256(diagnostic_path)})
    assert len({(row["model"], row["role"]) for row in selected}) == len(selected)
    assert len(selected) <= 6
    result = {"state": "completed", "selected": selected, "rationale": rationale, "selection_scope": "Development-selected research assets; feature medical meaning requires image evidence and independent confirmation", "test_used": False, "created_at": timestamp()}
    write_json(destination, result)
    print({"state": result["state"], "configurations": len(selected), "dictionaries": len(selected) * 3}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args()
    select(campaign_root(args.root), args.sites, args.rationale)
