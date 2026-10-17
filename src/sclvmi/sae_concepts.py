import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_sites import SpatialEncoder


CONFIG_PATH = ROOT / "configs/sae_radiomics.json"
CAMPAIGN_CONFIG = ROOT / "configs/sae_interpretation.json"
KEYS = {("fmcib", "layer2"): (("previous", "layer2"), ("new", "layer2_random"), "layer2_distance"), ("vista", "stage2"): (("previous", "stage2"), ("new", "stage2_random"), "stage2_distance")}


def paths():
    config, campaign = read_json(CONFIG_PATH), read_json(CAMPAIGN_CONFIG)
    storage = context()[0]
    runs = Path(storage["runs"])
    return config, campaign, runs / campaign["campaign"], Path(storage["activations"]) / campaign["previous_suite"], runs / config["run_id"]


def members(root):
    frame = pd.read_csv(root / "profiles" / "fmcib_layer2" / "members.csv", dtype={"PatientID": str})
    assert frame.PatientID.is_unique and set(frame.split) == {"train", "development"}
    return frame


def dictionaries(root, model, site, seed):
    index = read_json(root / "viewer/index.json")["dictionaries"]
    rows = {(row["encoder"], row["seed"]): row for row in index if row["model"] == model and row["site"] == site}
    return rows[("trained", seed)], rows[("random", seed)], {key[1]: row for key, row in rows.items() if key[0] == "trained" and key[1] != seed}


def stable_mask(root, group, campaign):
    with np.load(root / "stability" / group / "features.npz") as saved:
        reappearance, live, matched, seeds = saved["reappearance"][0], saved["development_live"][0], saved["matched"], saved["seeds"]
    return (reappearance >= campaign["stability"]["stable_minimum_reappearance"]) & live, matched, seeds


def load_arrays(root, previous, model, site, identifier):
    (trained_location, trained_key), (random_location, random_key), distance_key = KEYS[(model, site)]
    with np.load(root / "cases" / model / f"{identifier}.npz") as new, np.load(previous / model / "cases" / f"{identifier}.npz") as old:
        source = {"new": new, "previous": old}
        return source[trained_location][trained_key], source[random_location][random_key], new[distance_key]


def region_statistics(codes, distance):
    within, near = distance <= 4, (distance > 4) & (distance <= 10)
    assert within.any() and near.any()
    return torch.stack([codes[within].mean(0), codes[near].mean(0), codes[within].amax(0)], -1)


@torch.inference_mode()
def activations(config, campaign, root, previous, output, device):
    frame = members(root)
    output.mkdir(parents=True, exist_ok=True)
    for name in config["sites"]:
        model, site = name.split(":")
        destination = output / f"activations_{model}_{site}.npz"
        if destination.exists():
            continue
        trained, random, others = dictionaries(root, model, site, config["reference_seed"])
        stable, matched, seeds = stable_mask(root, trained["stability_group"], campaign)
        assert seeds[0] == config["reference_seed"]
        load = lambda row: SpatialDictionary(root / "runs" / row["run_id"] / "dictionary.pt", device=device)
        trained_sae, random_sae = load(trained), load(random)
        stable_index = np.flatnonzero(stable)
        other_saes = {int(seed): (load(others[int(seed)]), torch.from_numpy(matched[0, position, stable_index].astype(np.int64)).to(device)) for position, seed in enumerate(seeds) if position > 0}
        result = {"trained": [], "random": [], "matched": {seed: [] for seed in other_saes}}
        started = time.monotonic()
        for count, identifier in enumerate(frame.identifier):
            trained_values, random_values, distance = load_arrays(root, previous, model, site, identifier)
            # Only positions within 10 mm of the nodule surface enter the region statistics.
            keep = np.flatnonzero(distance.reshape(-1) <= 10)
            distance = torch.from_numpy(distance.reshape(-1)[keep]).to(device)
            tokens = torch.from_numpy(trained_values.reshape(trained_values.shape[0], -1).T[keep]).to(device)
            result["trained"].append(region_statistics(trained_sae.encode(tokens), distance).cpu().numpy())
            random_tokens = torch.from_numpy(random_values.reshape(random_values.shape[0], -1).T[keep]).to(device)
            result["random"].append(region_statistics(random_sae.encode(random_tokens), distance).cpu().numpy())
            for seed, (sae, columns) in other_saes.items():
                result["matched"][seed].append(region_statistics(sae.encode(tokens)[:, columns], distance).cpu().numpy())
            if (count + 1) % 200 == 0 or count + 1 == len(frame):
                elapsed = time.monotonic() - started
                print(f"ACTIVATIONS {name} {count + 1}/{len(frame)} {elapsed:.0f} s, eta {elapsed / (count + 1) * (len(frame) - count - 1):.0f} s", flush=True)
        np.savez_compressed(destination, identifiers=frame.identifier.to_numpy(dtype=str), trained=np.stack(result["trained"]).astype(np.float32), random=np.stack(result["random"]).astype(np.float32),
                            matched=np.stack([np.stack(result["matched"][seed]) for seed in sorted(result["matched"])]).astype(np.float32), matched_seeds=np.asarray(sorted(result["matched"])),
                            stable=stable, random_stable=stable_mask(root, random["stability_group"], campaign)[0], statistics=np.asarray(config["activation_statistics"]))
        write_json(destination.with_suffix(".json"), {"site": name, "trained_run": trained["run_id"], "random_run": random["run_id"], "matched_runs": {int(seed): others[int(seed)]["run_id"] for seed in other_saes},
                                                     "stable_features": int(stable.sum()), "patients": len(frame), "rule": config["activation_rule"], "created_at": timestamp(), "test_used": False})


def effects(config, campaign, root, previous, output, device):
    settings = config["effects"]
    frame = members(root)
    development = frame[frame.split == "development"].reset_index(drop=True)
    for name in config["sites"]:
        model, site = name.split(":")
        destination = output / f"effects_{model}_{site}.npz"
        if destination.exists():
            continue
        trained, _, _ = dictionaries(root, model, site, config["reference_seed"])
        features = np.flatnonzero(stable_mask(root, trained["stability_group"], campaign)[0])
        sae = SpatialDictionary(root / "runs" / trained["run_id"] / "dictionary.pt", device=device)
        directions = sae.weights["W_dec"][torch.from_numpy(features).to(device)] * sae.scale
        encoder = SpatialEncoder(model, device=device)
        if isinstance(encoder.head, torch.nn.Module):
            encoder.head.eval()
        contributions, logits, checks, cache = [], [], [], []
        started = time.monotonic()
        for count, identifier in enumerate(development.identifier):
            with np.load(previous / model / "cases" / f"{identifier}.npz") as saved:
                activation, cached = torch.from_numpy(saved[site]).to(device), float(saved["probability"][0])
            x = activation[None].clone().requires_grad_(True)
            logit = encoder.logit(encoder.forward_from(site, x))
            gradient = torch.autograd.grad(logit.sum(), x)[0][0]
            with torch.no_grad():
                codes = sae.encode(activation.flatten(1).T)[:, torch.from_numpy(features).to(device)].clone()
                # First-order contribution of each feature to the malignancy logit: sum over positions of activation x (decoder direction . gradient).
                contributions.append((codes * (gradient.flatten(1).T @ directions.T)).sum(0).cpu().numpy())
                logits.append(float(logit))
                checks.append({"identifier": identifier, "cached_probability": cached, "torch_probability": float(torch.sigmoid(logit)), "classifier_probability": float(encoder.probability(encoder.continue_from(site, activation[None]))[0])})
                cache.append((activation, codes))
            if (count + 1) % 50 == 0 or count + 1 == len(development):
                elapsed = time.monotonic() - started
                print(f"EFFECTS {name} gradient {count + 1}/{len(development)} {elapsed:.0f} s", flush=True)
        contributions = np.stack(contributions)
        top = np.argsort(-np.abs(contributions).mean(0))[:settings["top"]]
        exact = np.zeros((len(development), len(top)), dtype=np.float32)
        top_directions = directions[torch.from_numpy(top).to(device)]
        with torch.no_grad():
            for row, (activation, codes) in enumerate(cache):
                tokens = activation.flatten(1).T
                for first in range(0, len(top), settings["batch_size"]):
                    chosen = top[first:first + settings["batch_size"]]
                    removed = tokens[None] - codes[:, torch.from_numpy(chosen).to(device)].T[:, :, None] * top_directions[first:first + len(chosen)][:, None, :]
                    removed = removed.transpose(1, 2).reshape(len(chosen), *activation.shape)
                    exact[row, first:first + len(chosen)] = (encoder.logit(encoder.forward_from(site, removed)) - logits[row]).cpu().numpy()
                if (row + 1) % 50 == 0 or row + 1 == len(cache):
                    print(f"EFFECTS {name} exact removal {row + 1}/{len(cache)} {time.monotonic() - started:.0f} s", flush=True)
        checks = pd.DataFrame(checks)
        agreement = float(np.corrcoef(-contributions[:, top].ravel(), exact.ravel())[0, 1])
        np.savez_compressed(destination, identifiers=development.identifier.to_numpy(dtype=str), features=features, contributions=contributions.astype(np.float32), logits=np.asarray(logits, dtype=np.float32),
                            top=features[top], exact_logit_change=exact)
        write_json(destination.with_suffix(".json"), {"site": name, "dictionary": trained["run_id"], "classifier": read_json(Path(context()[0]["runs"]) / "20260926_material_preparation/materials" / model / "material.json")["run"],
                                                     "patients": len(development), "stable_features": int(len(features)), "top": int(len(top)),
                                                     "max_probability_difference_torch_vs_classifier": float((checks.torch_probability - checks.classifier_probability).abs().max()),
                                                     "max_probability_difference_cache_vs_classifier": float((checks.cached_probability - checks.classifier_probability).abs().max()),
                                                     "first_order_vs_exact_correlation": agreement, "rule": settings["rule"], "created_at": timestamp(), "test_used": False})
        print("EFFECTS", name, "first-order vs exact removal correlation", round(agreement, 3), "max probability check", float((checks.torch_probability - checks.classifier_probability).abs().max()), flush=True)


def ranks(values):
    return rankdata(values, axis=0)


def spearman_matrix(left, right):
    left, right = ranks(left), ranks(right)
    left = (left - left.mean(0)) / np.where(left.std(0) > 0, left.std(0), np.nan)
    right = (right - right.mean(0)) / np.where(right.std(0) > 0, right.std(0), np.nan)
    return left.T @ right / len(left)


def residual(values, covariate):
    design = np.column_stack([np.ones(len(values)), ranks(covariate)])
    ranked = ranks(values)
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def clusters(radiomics, threshold):
    correlation = np.nan_to_num(np.abs(spearman_matrix(radiomics, radiomics)))
    np.fill_diagonal(correlation, 1)
    labels = fcluster(linkage(squareform(1 - correlation, checks=False), method="average"), t=1 - threshold, criterion="distance")
    representatives = []
    for label in np.unique(labels):
        members_index = np.flatnonzero(labels == label)
        representatives.append(members_index[np.argmax(correlation[np.ix_(members_index, members_index)].mean(1))])
    return labels, np.asarray(representatives)


def normal_scores(train, development):
    transform = QuantileTransformer(n_quantiles=min(1000, len(train)), output_distribution="normal", random_state=0).fit(train)
    return transform.transform(train), transform.transform(development)


def forward_selection(target_train, target_development, predictors_train, predictors_development, settings):
    chosen, design_train, design_development = [], np.ones((len(target_train), 1)), np.ones((len(target_development), 1))
    best_train = 0.0
    total = np.square(target_train - target_train.mean()).sum()
    for _ in range(settings["maximum_selected"]):
        coefficients = np.linalg.lstsq(design_train, target_train, rcond=None)[0]
        remainder = target_train - design_train @ coefficients
        projection = predictors_train - design_train @ np.linalg.lstsq(design_train, predictors_train, rcond=None)[0]
        norms = np.square(projection).sum(0)
        gains = np.where(norms > 1e-9, np.square(projection.T @ remainder) / np.where(norms > 1e-9, norms, 1), 0) / total
        gains[chosen] = 0
        pick = int(np.argmax(gains))
        if gains[pick] < settings["minimum_gain"]:
            break
        chosen.append(pick)
        design_train = np.column_stack([design_train, predictors_train[:, pick]])
        design_development = np.column_stack([design_development, predictors_development[:, pick]])
        best_train += gains[pick]
    coefficients = np.linalg.lstsq(design_train, target_train, rcond=None)[0]
    predicted = design_development @ coefficients
    development_total = np.square(target_development - target_development.mean()).sum()
    development_r2 = 1 - np.square(target_development - predicted).sum() / development_total if development_total > 0 else np.nan
    return chosen, coefficients[1:], best_train, development_r2


def explain(values, peak, radiomics_train, radiomics_development, train, development, representatives, settings, generator, permutations):
    rows, null = [], []
    for feature in range(values.shape[1]):
        target = values[:, feature, peak[feature]]
        if target[train].std() == 0 or target[development].std() == 0:
            rows.append({"chosen": [], "coefficients": [], "train_r2": np.nan, "development_r2": np.nan, "active_development": int((target[development] > 0).sum())})
            continue
        target_train, target_development = normal_scores(target[train, None], target[development, None])
        chosen, coefficients, train_r2, development_r2 = forward_selection(target_train[:, 0], target_development[:, 0], radiomics_train[:, representatives], radiomics_development[:, representatives], settings)
        rows.append({"chosen": [int(representatives[index]) for index in chosen], "coefficients": coefficients.tolist(), "train_r2": train_r2, "development_r2": development_r2,
                     "active_development": int((target[development] > 0).sum())})
        for _ in range(permutations):
            shuffled = target.copy()
            shuffled[train] = generator.permutation(shuffled[train])
            shuffled[development] = generator.permutation(shuffled[development])
            permuted_train, permuted_development = normal_scores(shuffled[train, None], shuffled[development, None])
            null.append(forward_selection(permuted_train[:, 0], permuted_development[:, 0], radiomics_train[:, representatives], radiomics_development[:, representatives], settings)[3])
    return rows, np.asarray(null)


def analyze(config, root, output):
    settings = config["analysis"]
    frame = members(root)
    radiomics = pd.read_csv(output / "radiomics_features.csv").set_index("identifier").loc[frame.identifier]
    radiomics = radiomics.drop(columns=["seconds"])
    radiomics = radiomics.loc[:, radiomics.notna().all() & (radiomics.std() > 0)]
    names = radiomics.columns.to_numpy()
    train, development = (frame.split == "train").to_numpy(), (frame.split == "development").to_numpy()
    values_radiomics = radiomics.to_numpy(dtype=np.float64)
    labels, representatives = clusters(values_radiomics[train], settings["cluster_correlation"])
    radiomics_train, radiomics_development = normal_scores(values_radiomics[train], values_radiomics[development])
    size = values_radiomics[:, list(names).index(settings["size_variable"])]
    generator = np.random.default_rng(20261017)
    tables, nulls, structured = [], {}, []
    for name in config["sites"]:
        model, site = name.split(":")
        with np.load(output / f"activations_{model}_{site}.npz") as saved:
            assert (saved["identifiers"] == frame.identifier.to_numpy()).all()
            arrays = {key: saved[key] for key in saved.files}
        for encoder, mask_key in [("trained", "stable"), ("random", "random_stable")]:
            features = np.flatnonzero(arrays[mask_key])
            # Targets come from the two region means; the within-4 mm maximum is kept for the review list.
            values = arrays[encoder][:, features, :2]
            peak = values[train].mean(0).argmax(1)
            rows, null = explain(values, peak, radiomics_train, radiomics_development, train, development, representatives, settings, generator, settings["permutations"])
            threshold = float(np.nanpercentile(null, 99))
            nulls[f"{name}:{encoder}"] = {"threshold": threshold, "values": null.tolist()}
            targets = np.stack([values[:, index, peak[index]] for index in range(len(features))], 1)
            raw = spearman_matrix(targets[train], values_radiomics[train])
            adjusted = spearman_matrix(residual(targets[train], size[train]), residual(values_radiomics[train], size[train]))
            agreement = np.full(len(features), np.nan)
            if encoder == "trained":
                profiles = [spearman_matrix(arrays["matched"][index][:, :, :2][np.arange(len(frame))[:, None], np.arange(len(features))[None, :], peak[None, :]][train], values_radiomics[train]) for index in range(len(arrays["matched_seeds"]))]
                agreement = np.nanmedian(np.stack([[np.corrcoef(np.nan_to_num(raw[row]), np.nan_to_num(profile[row]))[0, 1] for row in range(len(features))] for profile in profiles]), 0)
            for index, feature in enumerate(features):
                row = rows[index]
                top = np.argsort(-np.nan_to_num(np.abs(adjusted[index])))[:3]
                if row["active_development"] < settings["minimum_active_development"]:
                    category = "inactive"
                elif row["development_r2"] >= max(settings["explained_r2"], threshold) and (encoder == "random" or agreement[index] >= settings["cross_seed_agreement"]):
                    category = "explained"
                elif row["development_r2"] >= threshold:
                    category = "partial"
                else:
                    category = "unexplained"
                terms = {"explanation": [{"name": names[choice], "sign": 1 if coefficient > 0 else -1} for choice, coefficient in zip(row["chosen"], row["coefficients"])],
                         "top_raw": [{"name": names[choice], "rho": float(raw[index, choice])} for choice in np.argsort(-np.nan_to_num(np.abs(raw[index])))[:3]],
                         "top_size_adjusted": [{"name": names[choice], "rho": float(adjusted[index, choice])} for choice in top]}
                structured.append({"site": name, "encoder": encoder, "feature": int(feature), **terms})
                tables.append({"site": name, "encoder": encoder, "feature": int(feature), "peak_region": config["activation_statistics"][peak[index]], "train_r2": row["train_r2"], "development_r2": row["development_r2"],
                               "null_threshold": threshold, "active_development": row["active_development"], "cross_seed_agreement": agreement[index], "category": category,
                               "explanation": "; ".join(f"{term['name']} ({'+' if term['sign'] > 0 else '-'})" for term in terms["explanation"]),
                               "top_raw": "; ".join(f"{term['name']} {term['rho']:+.2f}" for term in terms["top_raw"]),
                               "top_size_adjusted": "; ".join(f"{term['name']} {term['rho']:+.2f}" for term in terms["top_size_adjusted"])})
            print("ANALYZED", name, encoder, len(features), "features, null threshold", round(threshold, 3), flush=True)
    table = pd.DataFrame(tables)
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    table.to_csv(destination / "explanations.csv", index=False)
    write_json(destination / "explanation_terms.json", structured)
    pd.DataFrame({"feature": names, "group": labels, "representative": np.isin(np.arange(len(names)), representatives)}).to_csv(destination / "radiomics_groups.csv", index=False)
    summary = table.groupby(["site", "encoder", "category"]).size().unstack(fill_value=0)
    summary.to_csv(destination / "categories.csv")
    write_json(destination / "summary.json", {"radiomics_features": int(len(names)), "radiomics_groups": int(len(representatives)), "null": {key: value["threshold"] for key, value in nulls.items()},
                                              "categories": {f"{site}:{encoder}": row.to_dict() for (site, encoder), row in summary.iterrows()},
                                              "median_development_r2": {f"{site}:{encoder}": value for (site, encoder), value in table.groupby(["site", "encoder"]).development_r2.median().items()},
                                              "settings": settings, "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    write_json(destination / "null.json", nulls)
    figure(table, nulls, destination)
    print(summary.to_string(), flush=True)


def review(config, output):
    runs = Path(context()[0]["runs"])
    explanations = pd.read_csv(output / "analysis" / "explanations.csv")
    tables = []
    for name in config["sites"]:
        model, site = name.split(":")
        with np.load(output / f"effects_{model}_{site}.npz") as saved:
            features, contributions, top, exact = saved["features"], saved["contributions"], saved["top"], saved["exact_logit_change"]
        effect = pd.DataFrame({"feature": features, "contribution_mean": contributions.mean(0), "contribution_mean_abs": np.abs(contributions).mean(0)})
        effect["effect_rank"] = effect.contribution_mean_abs.rank(ascending=False, method="first").astype(int)
        effect = effect.merge(pd.DataFrame({"feature": top, "removal_logit_change_mean": exact.mean(0), "removal_logit_change_mean_abs": np.abs(exact).mean(0)}), on="feature", how="left")
        candidates = pd.read_csv(runs / config["candidate_run"] / f"{model}_{site}" / "candidates.csv")
        candidates = candidates[candidates.confirmed][["feature", "partial_confirmation", "q_confirmation", "beyond_morphology", "delta_auroc", "delta_auroc_low", "delta_auroc_high", "family"]]
        table = explanations[(explanations.site == name) & (explanations.encoder == "trained")].merge(effect, on="feature", how="left").merge(candidates, on="feature", how="left")
        table["malignancy_confirmed"] = table.partial_confirmation.notna()
        tables.append(table)
    table = pd.concat(tables)
    rule = config["review"]
    table["review"] = table.malignancy_confirmed & (table.effect_rank <= rule["effect_rank"]) & table.category.isin(rule["categories"])
    destination = output / "analysis"
    table.sort_values(["site", "effect_rank"]).to_csv(destination / "trained_features.csv", index=False)
    listed = table[table.review].sort_values(["site", "category", "effect_rank"], ascending=[True, False, True])
    listed.to_csv(destination / "review_list.csv", index=False)
    write_json(destination / "review.json", {"rule": rule, "counts": {site: {"stable": int((table.site == site).sum()), "malignancy_confirmed": int(((table.site == site) & table.malignancy_confirmed).sum()),
                                                                              "top_effect": int(((table.site == site) & (table.effect_rank <= rule["effect_rank"])).sum()), "review": int(((table.site == site) & table.review).sum())} for site in config["sites"]},
                                             "created_at": timestamp(), "test_used": False})
    review_figure(table, destination)
    print(listed[["site", "feature", "category", "development_r2", "effect_rank", "delta_auroc", "explanation"]].to_string(index=False), flush=True)


def review_figure(table, destination):
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    sites = list(dict.fromkeys(table.site))
    fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 4.4), facecolor=surface, squeeze=False)
    for axis, site in zip(axes[0], sites):
        rows = table[(table.site == site) & (table.category != "inactive")]
        groups = [(rows[~rows.malignancy_confirmed], "#b9b8b3", "other analysable stable features"), (rows[rows.malignancy_confirmed & ~rows.review], "#2a78d6", "malignancy association confirmed"),
                  (rows[rows.review], "#eb6834", "review list")]
        for subset, color, label in groups:
            axis.scatter(subset.development_r2, subset.contribution_mean_abs, s=26 if color == "#b9b8b3" else 46, color=color, edgecolor=surface, linewidth=1, label=f"{label} ({len(subset)})", zorder=3 if color != "#b9b8b3" else 2)
        for row in rows[rows.review].itertuples():
            axis.annotate(str(row.feature), (row.development_r2, row.contribution_mean_abs), xytext=(5, 3), textcoords="offset points", fontsize=8, color=ink)
        axis.set_yscale("log")
        axis.axvline(0.5, color=muted, linestyle=":", linewidth=1)
        axis.set_title(site, loc="left")
        axis.set_xlabel("development R2 of the radiomics explanation")
        axis.set_ylabel("mean |first-order contribution| to the classifier logit")
        axis.set_facecolor(surface)
        axis.grid(color=grid, linewidth=0.6)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"review.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def figure(table, nulls, destination):
    colors = {"trained": "#2a78d6", "random": "#eb6834"}
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    sites = list(dict.fromkeys(table.site))
    fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 4.2), facecolor=surface, squeeze=False)
    bins = np.linspace(-0.2, 1, 49)
    for axis, site in zip(axes[0], sites):
        for encoder in ["trained", "random"]:
            values = table[(table.site == site) & (table.encoder == encoder) & (table.category != "inactive")].development_r2.dropna()
            axis.hist(values, bins=bins, color=colors[encoder], histtype="step", linewidth=2, label=f"{encoder} encoder ({len(values)} analysable stable features)", density=True)
        axis.axvline(nulls[f"{site}:trained"]["threshold"], color=muted, linestyle="--", linewidth=1, label="shuffled-target 99th percentile")
        axis.axvline(0.5, color=muted, linestyle=":", linewidth=1, label="R2 0.5")
        axis.set_title(site, loc="left")
        axis.set_xlabel("development R2 of the radiomics explanation")
        axis.set_facecolor(surface)
        axis.grid(axis="y", color=grid, linewidth=0.6)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"explained_variance.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["activations", "effects", "analyze", "review"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config, campaign, root, previous, output = paths()
    torch.set_num_threads(8)
    if args.stage == "activations":
        activations(config, campaign, root, previous, output, args.device)
    elif args.stage == "effects":
        effects(config, campaign, root, previous, output, args.device)
    elif args.stage == "analyze":
        analyze(config, root, output)
    else:
        review(config, output)


if __name__ == "__main__":
    main()
