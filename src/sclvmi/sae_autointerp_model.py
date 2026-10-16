import argparse
import base64
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .context import ROOT, context, read_json, sha256, timestamp, write_json


CONFIG_PATH = ROOT / "configs/sae_autointerp.json"
SOURCES = {"fmcib:layer2": "fmcib", "vista:stage2": "vista"}
ROUTES = {"describe_summarize": "summarize", "direct": "direct"}
STAGES = ["describe", "summarize", "direct", "validate", "specificity", "analyze"]


class ExampleDescription(BaseModel):
    nodule: str
    surroundings: str
    overlay_location: str
    overlay_content: str
    uncertainties: str


class FeatureExplanation(BaseModel):
    no_clear_pattern: bool
    candidate_label: str
    description: str
    supporting_example_ids: list[str]
    contradicting_example_ids: list[str]
    uncertainties: str


class MatchScore(BaseModel):
    score: int
    reason: str


class Ledger:
    def __init__(self, output, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.spent = sum(read_json(path)["cost_usd"] for path in (output / "calls").rglob("*.json")) if (output / "calls").exists() else 0.0
        self.reserved = 0.0

    def cost(self, usage):
        prices = self.settings["price_per_million_usd"]
        cached = usage["input_tokens_details"]["cached_tokens"]
        return ((usage["input_tokens"] - cached) * prices["input"] + cached * prices["cached_input"] + usage["output_tokens"] * prices["output"]) / 1e6

    def reserve(self, stage):
        prices = self.settings["price_per_million_usd"]
        amount = (self.settings["reserve_input_tokens"][stage] * prices["input"] + self.settings["max_output_tokens"][stage] * prices["output"]) / 1e6
        with self.lock:
            assert self.spent + self.reserved + amount <= self.settings["budget_usd"], f"budget {self.settings['budget_usd']} USD would be exceeded (spent {self.spent:.2f})"
            self.reserved += amount
        return amount

    def settle(self, amount, cost):
        with self.lock:
            self.reserved -= amount
            self.spent += cost


def image_part(path, detail):
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(), "detail": detail}


def call_model(client, ledger, stage, schema, content, destination, metadata):
    settings = ledger.settings
    amount = ledger.reserve(stage)
    started = time.monotonic()
    response = client.responses.parse(model=settings["name"], input=[{"role": "user", "content": content}], text_format=schema,
                                      reasoning={"effort": settings["reasoning"][stage]}, max_output_tokens=settings["max_output_tokens"][stage])
    assert response.status == "completed", f"{destination}: status {response.status} {response.incomplete_details}"
    refusals = [part.refusal for item in response.output if item.type == "message" for part in item.content if part.type == "refusal"]
    assert not refusals, f"{destination}: refusal {refusals}"
    usage = response.usage.model_dump()
    cost = ledger.cost(usage)
    ledger.settle(amount, cost)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {**metadata, "stage": stage, "model": response.model, "response_id": response.id, "parsed": response.output_parsed.model_dump(), "usage": usage, "cost_usd": cost,
                             "seconds": time.monotonic() - started, "prompt_text": [part["text"] for part in content if part["type"] == "input_text"], "raw": response.model_dump(mode="json"), "created_at": timestamp()})


def run_calls(config, output, stage, jobs):
    settings = config["model"]
    ledger = Ledger(output, settings)
    pending = [job for job in jobs if not job["destination"].exists()]
    print(stage.upper(), len(jobs), "calls,", len(jobs) - len(pending), "already saved, spent so far", round(ledger.spent, 4), "USD", flush=True)
    if not pending:
        return
    client = OpenAI(api_key=os.environ[settings["api_key_environment"]], max_retries=settings["max_retries"], timeout=settings["timeout_seconds"])
    started, done = time.monotonic(), 0
    with ThreadPoolExecutor(settings["concurrency"]) as pool:
        futures = [pool.submit(call_model, client, ledger, job["stage"], job["schema"], job["content"], job["destination"], job["metadata"]) for job in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0 or done == len(pending):
                elapsed = time.monotonic() - started
                print(f"{stage.upper()} {done}/{len(pending)} {elapsed:.0f} s, eta {elapsed / done * (len(pending) - done):.0f} s, spent {ledger.spent:.3f} USD", flush=True)


def layout(config, output, source, overlay):
    window_mm = read_json(output / "render.json")["window_mm"][source]
    return config["prompts"]["layout"].format(window_mm=f"{window_mm:.0f}", overlay_row=config["prompts"]["overlay_row"] if overlay else "")


def tables(output):
    features = {item["key"]: item for item in read_json(output / "features.json")["features"]}
    return features, pd.read_csv(output / "examples.csv", dtype={"PatientID": str})


def describe_jobs(config, output, keys):
    features, examples = tables(output)
    jobs = []
    for row in examples[examples.stage == "generation"].itertuples():
        if keys and row.key not in keys:
            continue
        prompt = config["prompts"]["describe"].format(layout=layout(config, output, SOURCES[features[row.key]["site"]], True))
        content = [{"type": "input_text", "text": prompt}, image_part(output / "images_generation" / row.key / f"{row.example_id}.png", config["model"]["detail"])]
        jobs.append({"stage": "describe", "schema": ExampleDescription, "content": content, "destination": output / "calls" / "describe" / row.key / f"{row.example_id}.json",
                     "metadata": {"key": row.key, "example_id": row.example_id}})
    return jobs


def explanation_jobs(config, output, stage, keys):
    features, examples = tables(output)
    jobs = []
    for key, group in examples[examples.stage == "generation"].groupby("key", sort=False):
        if keys and key not in keys:
            continue
        if stage == "summarize":
            lines = []
            for row in group.itertuples():
                text = read_json(output / "calls" / "describe" / key / f"{row.example_id}.json")["parsed"]
                lines.append(f"[{row.example_id}] ({row.group}) nodule: {text['nodule']} | surroundings: {text['surroundings']} | overlay location: {text['overlay_location']} | overlay content: {text['overlay_content']} | uncertainties: {text['uncertainties']}")
            content = [{"type": "input_text", "text": config["prompts"]["summarize"].format(examples="\n".join(lines))}]
        else:
            content = [{"type": "input_text", "text": config["prompts"]["direct"].format(layout=layout(config, output, SOURCES[features[key]["site"]], True))}]
            for row in group.itertuples():
                content += [{"type": "input_text", "text": f"Image {row.example_id}:"}, image_part(output / "images_generation" / key / f"{row.example_id}.png", config["model"]["detail"])]
        jobs.append({"stage": stage, "schema": FeatureExplanation, "content": content, "destination": output / "calls" / stage / f"{key}.json", "metadata": {"key": key}})
    return jobs


def validate_jobs(config, output, keys):
    features, examples = tables(output)
    jobs = []
    for key, group in examples[examples.stage == "validation"].groupby("key", sort=False):
        if keys and key not in keys:
            continue
        source = SOURCES[features[key]["site"]]
        for route, stage in ROUTES.items():
            explanation = read_json(output / "calls" / stage / f"{key}.json")["parsed"]
            if explanation["no_clear_pattern"]:
                continue
            prompt = config["prompts"]["score"].format(layout=layout(config, output, source, False), label=explanation["candidate_label"], description=explanation["description"])
            for row in group.itertuples():
                content = [{"type": "input_text", "text": prompt}, image_part(output / "images_validation" / f"{source}_{row.identifier}.png", config["model"]["detail"])]
                jobs.append({"stage": "score", "schema": MatchScore, "content": content, "destination": output / "calls" / "score" / route / key / f"{row.example_id}.json",
                             "metadata": {"key": key, "route": route, "example_id": row.example_id}})
    return jobs


def ordered(identifier):
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")


def panel(config, features, examples, site):
    keys = [key for key, item in features.items() if item["site"] == site]
    pool = examples[examples.key.isin(keys) & (examples.stage == "validation")].drop_duplicates("identifier")
    return pool.assign(order=pool.identifier.map(ordered)).sort_values("order").head(config["specificity"]["panel_per_site"])


def specificity_jobs(config, output, keys):
    features, examples = tables(output)
    stage = ROUTES[config["specificity"]["route"]]
    jobs = []
    for site in SOURCES:
        source = SOURCES[site]
        patients = panel(config, features, examples, site)
        for key, feature in features.items():
            if feature["site"] != site or feature["role"] == "shuffled" or (keys and key not in keys):
                continue
            explanation = read_json(output / "calls" / stage / f"{key}.json")["parsed"]
            if explanation["no_clear_pattern"]:
                continue
            prompt = config["prompts"]["score"].format(layout=layout(config, output, source, False), label=explanation["candidate_label"], description=explanation["description"])
            for row in patients.itertuples():
                content = [{"type": "input_text", "text": prompt}, image_part(output / "images_validation" / f"{source}_{row.identifier}.png", config["model"]["detail"])]
                jobs.append({"stage": "score", "schema": MatchScore, "content": content, "destination": output / "calls" / "specificity" / key / f"{row.identifier}.json",
                             "metadata": {"key": key, "route": config["specificity"]["route"], "identifier": row.identifier}})
    return jobs


def specificity(config, output, features, examples):
    candidates = Path(context()[0]["runs"]) / config["candidates_run"]
    rows = []
    for site in SOURCES:
        patients = panel(config, features, examples, site)
        with np.load(candidates / site.replace(":", "_") / "summaries.npz") as saved:
            values = {encoder: saved[encoder][patients.row.to_numpy()] for encoder in ["trained", "random"]}
        site_keys = [key for key, item in features.items() if item["site"] == site and item["role"] != "shuffled"]
        targets = {key: values[features[key]["encoder"]][:, features[key]["feature"]] for key in site_keys}
        for key in site_keys:
            if read_json(output / "calls" / ROUTES[config["specificity"]["route"]] / f"{key}.json")["parsed"]["no_clear_pattern"]:
                continue
            scores = np.asarray([read_json(output / "calls" / "specificity" / key / f"{identifier}.json")["parsed"]["score"] for identifier in patients.identifier], dtype=np.float64)
            for target, value in targets.items():
                rows.append({"site": site, "label_key": key, "target_key": target, "own": key == target, "spearman": spearmanr(scores, value).statistic})
    frame = pd.DataFrame(rows)
    ranks = frame.assign(rank=frame.groupby("label_key").spearman.rank(ascending=False)).query("own")[["label_key", "spearman", "rank"]]
    labels = frame.label_key.nunique()
    summary = {"labels": int(labels), "targets_per_site": {site: int(group.target_key.nunique()) for site, group in frame.groupby("site")},
               "own_rank_1": int((ranks["rank"] == 1).sum()), "median_own_rank": float(ranks["rank"].median()), "mean_own_spearman": float(frame[frame.own].spearman.mean()),
               "mean_other_spearman": float(frame[~frame.own].spearman.mean())}
    return frame, ranks, summary


def bootstrap_auroc(labels, scores, count, generator):
    values = []
    for _ in range(count):
        sample = generator.integers(0, len(labels), len(labels))
        if labels[sample].min() != labels[sample].max():
            values.append(roc_auc_score(labels[sample], scores[sample]))
    return np.percentile(values, [2.5, 97.5])


def analyze(config, output):
    features, examples = tables(output)
    measured = pd.read_csv(output / "measurement.csv").set_index(["key", "example_id"]).measurement_score
    generator = np.random.default_rng(20261016)
    rows, scored = [], []
    for key, feature in features.items():
        validation = examples[(examples.key == key) & (examples.stage == "validation")].reset_index(drop=True)
        labels = (validation.group == "high").to_numpy().astype(int)
        baseline = measured.loc[[(key, example) for example in validation.example_id]].to_numpy()
        usable = np.isfinite(baseline)
        measurement_auroc = roc_auc_score(labels[usable], baseline[usable])
        for route, stage in ROUTES.items():
            explanation = read_json(output / "calls" / stage / f"{key}.json")["parsed"]
            row = {"key": key, "site": feature["site"], "encoder": feature["encoder"], "role": feature["role"], "feature": feature["feature"], "route": route,
                   "no_clear_pattern": explanation["no_clear_pattern"], "candidate_label": explanation["candidate_label"], "description": explanation["description"],
                   "measurement_auroc": measurement_auroc, "nodule_enrichment": feature["nodule_enrichment"], "delta_auroc_volume": feature["delta_auroc"]}
            if not explanation["no_clear_pattern"]:
                scores = np.asarray([read_json(output / "calls" / "score" / route / key / f"{example}.json")["parsed"]["score"] for example in validation.example_id], dtype=np.float64)
                assert ((scores >= 0) & (scores <= 10)).all(), key
                low, high = bootstrap_auroc(labels, scores, 2000, generator)
                row.update({"auroc": roc_auc_score(labels, scores), "auroc_low": low, "auroc_high": high, "spearman": spearmanr(scores, validation.value).statistic,
                            "mean_score_high": scores[labels == 1].mean(), "mean_score_low": scores[labels == 0].mean()})
                scored += [{"key": key, "route": route, "example_id": example, "group": group, "value": value, "score": score} for example, group, value, score in zip(validation.example_id, validation.group, validation.value, scores)]
            rows.append(row)
    table = pd.DataFrame(rows)
    destination = output / "results"
    destination.mkdir(exist_ok=True)
    table.to_csv(destination / "validation.csv", index=False)
    pd.DataFrame(scored).to_csv(destination / "scores.csv", index=False)
    calls = [read_json(path) for path in (output / "calls").rglob("*.json")]
    costs = pd.DataFrame([{"stage": call["stage"], "cost_usd": call["cost_usd"], "input_tokens": call["usage"]["input_tokens"], "output_tokens": call["usage"]["output_tokens"]} for call in calls])
    summary = table.assign(passed=lambda frame: frame.auroc_low > 0.5).groupby(["role", "route"]).agg(features=("key", "count"), labelled=("no_clear_pattern", lambda values: int((~values).sum())), passed=("passed", "sum"),
                                                                                                   median_auroc=("auroc", "median"), median_measurement_auroc=("measurement_auroc", "median")).reset_index()
    summary.to_csv(destination / "summary_by_role.csv", index=False)
    pairs, own, specific = specificity(config, output, features, examples)
    pairs.to_csv(destination / "specificity_pairs.csv", index=False)
    own.to_csv(destination / "specificity_own.csv", index=False)
    write_json(destination / "summary.json", {"by_role": summary.to_dict(orient="records"), "specificity": specific, "cost_by_stage": costs.groupby("stage").sum().to_dict(orient="index"),
                                              "total_cost_usd": float(costs.cost_usd.sum()), "calls": int(len(costs)),
                                              "rule": "a label passes when the lower bound of its 95% patient-bootstrap AUROC (score, high versus size-matched low validation patients) exceeds 0.5",
                                              "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    figure(table, destination)
    pd.set_option("display.width", 250)
    print(table[["key", "route", "candidate_label", "auroc", "auroc_low", "auroc_high", "measurement_auroc", "spearman"]].round(3).to_string(index=False), flush=True)
    print(summary.round(3).to_string(index=False), flush=True)
    print(own.round(3).to_string(index=False), flush=True)
    print("SPECIFICITY", specific, flush=True)
    print("COST", round(float(costs.cost_usd.sum()), 3), "USD over", len(costs), "calls", flush=True)


def figure(table, destination):
    colors = {"describe_summarize": "#2a78d6", "direct": "#eb6834"}
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    order = ["candidate", "nodule", "random", "shuffled"]
    keys = table.assign(order=table.role.map(order.index)).sort_values(["order", "key"]).key.drop_duplicates().tolist()
    fig, axis = plt.subplots(figsize=(9, 0.32 * len(keys) + 1.6), facecolor=surface)
    positions = {key: index for index, key in enumerate(reversed(keys))}
    for route, offset in [("describe_summarize", 0.15), ("direct", -0.15)]:
        part = table[(table.route == route) & table.auroc.notna()]
        y = part.key.map(positions) + offset
        axis.hlines(y, part.auroc_low, part.auroc_high, color=colors[route], linewidth=2)
        axis.plot(part.auroc, y, "o", color=colors[route], markersize=6, markeredgecolor=surface, markeredgewidth=1.5, label="describe then summarise" if route == "describe_summarize" else "direct from images")
    measured = table.drop_duplicates("key")
    axis.plot(measured.measurement_auroc, measured.key.map(positions), "|", color=muted, markersize=12, markeredgewidth=2, label="measured nodule properties")
    axis.axvline(0.5, color=muted, linewidth=0.8)
    roles = table.drop_duplicates("key").set_index("key").role
    axis.set_yticks(list(positions.values()), [f"{key} ({roles[key]})" for key in positions])
    axis.set_xlabel("validation AUROC: high versus size-matched low development patients (95% bootstrap interval)")
    axis.set_xlim(0, 1)
    axis.set_facecolor(surface)
    axis.grid(axis="x", color=grid, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"validation.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--keys", default="", help="comma-separated feature keys")
    parser.add_argument("--limit", type=int, default=0, help="number of calls for a smoke run")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    output = Path(context()[0]["runs"]) / config["run_id"]
    keys = [value for value in args.keys.split(",") if value]
    if args.stage == "analyze":
        analyze(config, output)
        return
    if args.stage == "describe":
        jobs = describe_jobs(config, output, keys)
    elif args.stage == "validate":
        jobs = validate_jobs(config, output, keys)
    elif args.stage == "specificity":
        jobs = specificity_jobs(config, output, keys)
    else:
        jobs = explanation_jobs(config, output, args.stage, keys)
    run_calls(config, output, args.stage, jobs[:args.limit] if args.limit else jobs)


if __name__ == "__main__":
    main()
