from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    storage, _, _ = context()
    root = Path(storage['runs']) / '20261008_feature_validation/fmcib'
    checks = pd.read_csv(root / 'feature_patient_checks.csv').set_index('feature_id')
    members = pd.read_csv(root / 'image_sample.csv', dtype={'PatientID': str})
    assert members.PatientID.is_unique
    dictionaries = [SpatialDictionary(Path(storage['runs']) / '20261008_spatial_sae' / run / 'dictionary.pt', device='cpu') for run in read_json(root / 'screen.json')['runs']]
    with np.load(root / 'discovery_interior_position_means.npz') as saved:
        means = {key: saved[key].copy() for key in saved.files}
    features = checks.index[checks.qualified_numeric_feature].to_numpy()
    compared = 0
    maximum_error = 0.
    for split in ['discovery', 'confirmation']:
        selected = members[members.feature_split == split].reset_index(drop=True)
        saved_pairs = {}
        for mode in ['raw', 'residual']:
            for seed in [2026, 2027]:
                prefix = f'{split}_patient_{mode}_s{seed}'
                with np.load(root / f'{prefix}_values.npz') as saved:
                    values, valid = saved['correlations'], saved['valid']
                    saved_pairs[(mode, seed)] = values.copy()
                assert np.all(values[~valid] == 0)
                count = valid.sum(0)
                estimate = np.divide(values.sum(0), count, out=np.full(2048, np.nan), where=count > 0)
                np.testing.assert_allclose(checks[f'{prefix}_mean'], estimate, atol=1e-7, equal_nan=True)
                np.testing.assert_array_equal(checks[f'{prefix}_patients'], count)
        for index in [0, 9, 18, 27, 36, 45, 54, 63]:
            row = selected.iloc[index]
            with np.load(Path(storage['activations']) / '20261008_spatial_sae/fmcib/cases' / f'{row.identifier}.npz') as saved:
                native = torch.from_numpy(saved['layer1'][:, 2:-2, 2:-2, 2:-2].copy()).flatten(1).T
            activities = [dictionary.encode(native).numpy() for dictionary in dictionaries]
            for mode in ['raw', 'residual']:
                for seed_index, seed in [(1, 2026), (2, 2027)]:
                    for feature in features:
                        target = int(checks.loc[feature, f'patient_{mode}_counterpart_s{seed}'])
                        first = activities[0][:, feature].astype(float)
                        second = activities[seed_index][:, target].astype(float)
                        if mode == 'residual':
                            first -= means['s2025'][:, feature]
                            second -= means[f's{seed}'][:, target]
                        value = pearsonr(first, second).statistic
                        expected = saved_pairs[(mode, seed)][index, feature]
                        maximum_error = max(maximum_error, abs(value - expected))
                        np.testing.assert_allclose(value, expected, atol=2e-5)
                        compared += 1
            print(f'Checked {split} patient {index + 1}/64', flush=True)
    result = {'status': 'passed', 'completed_at': timestamp(), 'scipy_real_patient_feature_correlations': compared,
              'maximum_absolute_error': maximum_error, 'all_saved_means_and_patient_counts_recomputed': True,
              'source_sha256': sha256(Path(__file__))}
    write_json(root / 'patient_independent_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
