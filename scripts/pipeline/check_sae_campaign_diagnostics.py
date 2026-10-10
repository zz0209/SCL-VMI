import numpy as np
import torch

from sclvmi.context import read_json, timestamp, write_json
from sclvmi.sae_campaign import campaign_root
from sclvmi.sae_campaign_diagnostics import compare_seeds, pca_reference, weights_from_offsets
from sclvmi.sae_campaign_train import fit_job


def main():
    root = campaign_root() / "smoke"
    matrix = root / "smoke_matrices/vista/stage2"
    run_ids = []
    for seed in [2025, 2026, 2027]:
        run_id = f"smoke_vista_stage2_diagnostics_s{seed}"
        result = fit_job(root, {"run_id": run_id, "model": "vista", "site": "stage2", "seed": seed, "k": 64, "expansion": 4, "matrix_directory": "smoke_matrices", "training": {"steps": 48, "batch_size": 64, "validate_every": 24, "checkpoint_every": 24}, "inference": {"calibration_train_tokens": 256}}, "local")
        assert result["state"] == "completed"
        run_ids.append(run_id)
    pca = pca_reference(root, run_ids[0], matrix)
    source = np.load(matrix / "train.npy").astype(np.float64)
    weights = weights_from_offsets(np.load(matrix / "train_offsets.npy"))
    with np.load(matrix / "normalization.npz") as saved:
        mean, scale = saved["mean"], float(saved["scale"])
    normalized = (source - mean) / scale
    _, _, right = np.linalg.svd(normalized * np.sqrt(weights[:, None]), full_matrices=False)
    expected = right[:64].T @ right[:64]
    with np.load(root / "runs" / run_ids[0] / "pca_reference/basis.npz") as saved:
        basis = saved["basis"].astype(np.float64)
    np.testing.assert_allclose(basis @ basis.T, expected, atol=2e-6, rtol=2e-5)
    destination = root / "diagnostics/three_seed_witness"
    comparison = compare_seeds(root, run_ids, destination, matrix)
    sample = np.load(destination / "sample.npz")
    np.testing.assert_allclose(sample["weights"].sum(), 1.0, atol=1e-12)
    first = np.load(destination / f"{run_ids[0]}_activations.npy")
    second = np.load(destination / f"{run_ids[1]}_activations.npy")
    direct = first.astype(np.float64).T @ second.astype(np.float64)
    with np.load(destination / "s2025_s2026.npz") as saved:
        np.testing.assert_allclose(saved["correlation"], direct.max(axis=1), atol=2e-6, rtol=2e-5)
        selected = direct[np.arange(len(direct)), saved["target_index"]]
        np.testing.assert_allclose(selected, direct.max(axis=1), atol=2e-6, rtol=0)
        top_two = np.partition(direct, -2, axis=1)[:, -2:]
        unique = np.abs(top_two[:, 1] - top_two[:, 0]) > 4e-6
        assert np.array_equal(saved["target_index"][unique], direct.argmax(axis=1)[unique])
        assert np.isfinite(saved["decoder_cosine"]).all()
    assert len(comparison["comparisons"]) == 3
    assert read_json(destination / "result.json")["state"] == "completed"
    receipt = {"state": "passed", "input": "actual VISTA stage2 training and development vectors", "runs": run_ids, "pca_svd_projector_agreement": True, "weighted_matching_direct_agreement": True, "matching_absolute_tolerance": 2e-6, "numerically_separated_argmax_agreement": True, "seed_pairs": 3, "pca_development_fvu": pca["development_fvu"], "created_at": timestamp()}
    write_json(root / "diagnostics_witness.json", receipt)
    print("DIAGNOSTICS_WITNESS", receipt, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3)
    main()
