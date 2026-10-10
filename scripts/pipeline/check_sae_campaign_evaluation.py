from sclvmi.context import timestamp, write_json
from sclvmi.sae_campaign import campaign_root, configuration
from sclvmi.sae_campaign_evaluate import evaluate


def main():
    root = campaign_root() / "smoke"
    records = []
    for name, sites in configuration()["models"].items():
        for site in sites:
            run_id = f"smoke_{name}_{site}_complete_s2025"
            result = evaluate(root, run_id, smoke=True)
            assert result["state"] == "completed" and len(result["raw_checks"]) == 2
            records.append({"model": name, "site": site, "cases": result["metrics"]["cases"], "raw_checks": result["raw_checks"]})
            print("EVALUATION_WITNESS", name, site, result["metrics"], flush=True)
    write_json(root / "evaluation_witness.json", {"state": "passed", "records": records, "completed_at": timestamp(), "input": "real development crops and raw CT hook replacement"})


if __name__ == "__main__":
    main()
