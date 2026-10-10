import numpy as np
import torch

from sclvmi.context import timestamp, write_json
from sclvmi.sae_campaign import campaign_root
from sclvmi.sae_campaign_diagnostics import compare_seeds, pca_reference, weights_from_offsets
from sclvmi.sae_campaign_train import fit_job


def main():
    smoke = campaign_root() / "smoke"
    root = smoke / "defined_statistics"
    matrix = smoke / "smoke_matrices/vista/stage2"
    run_ids = []
    for seed in [2025, 2026, 2027]:
        run_id = f"smoke_vista_stage2_defined_s{seed}"
        result = fit_job(root, {"run_id": run_id, "model": "vista", "site": "stage2", "seed": seed, "k": 64, "expansion": 4, "matrix_directory": str(matrix.parents[1]), "training": {"steps": 48, "batch_size": 64, "validate_every": 24, "checkpoint_every": 24}, "inference": {"calibration_train_tokens": 256}}, "local")
        assert result["state"] == "completed"
        run_ids.append(run_id)
    pca = pca_reference(root, run_ids[0], matrix)
    train = np.load(matrix / "train.npy").astype(np.float64)
    weights = weights_from_offsets(np.load(matrix / "train_offsets.npy"))
    with np.load(matrix / "normalization.npz") as saved:
        normalized = (train - saved["mean"]) / float(saved["scale"])
    _, _, right = np.linalg.svd(normalized * np.sqrt(weights[:, None]), full_matrices=False)
    with np.load(root / "runs" / run_ids[0] / "pca_reference/basis.npz") as saved:
        basis = saved["basis"].astype(np.float64)
    np.testing.assert_allclose(basis @ basis.T, right[:64].T @ right[:64], atol=2e-6, rtol=2e-5)
    destination = root / "diagnostics"
    comparison = compare_seeds(root, run_ids, destination, matrix)
    checked = []
    for pair in comparison["comparisons"]:
        first = np.load(destination / f"{pair['source_run']}_activations.npy").astype(np.float64)
        second = np.load(destination / f"{pair['target_run']}_activations.npy").astype(np.float64)
        valid_source = np.sum(first ** 2, axis=0) > 0
        valid_target = np.sum(second ** 2, axis=0) > 0
        direct = first[:, valid_source].T @ second[:, valid_target]
        target_indices = np.flatnonzero(valid_target)
        with np.load(destination / f"{pair['pair']}.npz") as saved:
            np.testing.assert_array_equal(saved["source_nonconstant"], valid_source)
            np.testing.assert_array_equal(saved["target_nonconstant"], valid_target)
            assert np.isnan(saved["correlation"][~valid_source]).all()
            assert np.all(saved["target_index"][~valid_source] == -1)
            np.testing.assert_allclose(saved["correlation"][valid_source], direct.max(1), rtol=2e-5, atol=2e-6)
            reverse = {int(index): column for column, index in enumerate(target_indices)}
            chosen = np.array([reverse[int(index)] for index in saved["target_index"][valid_source]])
            np.testing.assert_allclose(direct[np.arange(len(direct)), chosen], direct.max(1), rtol=0, atol=2e-6)
            top_two = np.sort(direct, axis=1)[:, -2:]
            separated = top_two[:, 1] - top_two[:, 0] > 4e-6
            np.testing.assert_array_equal(chosen[separated], direct.argmax(1)[separated])
            assert np.isfinite(saved["decoder_cosine"][valid_source]).all()
            assert np.isnan(saved["decoder_cosine"][~valid_source]).all()
        checked.append({"pair": pair["pair"], "eligible_source": int(valid_source.sum()), "undefined_source": int((~valid_source).sum())})
    assert any(row["undefined_source"] > 0 for row in checked)
    receipt = {"state": "passed", "input": "actual VISTA stage2 training and development vectors", "pairs": checked, "undefined_correlations_retained_as_nan": True, "pca_svd_projector_agreement": True, "weighted_matching_direct_agreement": True, "pca_development_fvu": pca["development_fvu"], "created_at": timestamp()}
    write_json(smoke / "diagnostics_defined_witness.json", receipt)
    print("DEFINED_DIAGNOSTICS_WITNESS", receipt, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
