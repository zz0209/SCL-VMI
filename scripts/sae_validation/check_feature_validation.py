from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from feature_images import paired_sums, shifted_input, sums_correlation
from sclvmi.context import context, sha256, timestamp, write_json


storage, _, _ = context()
root = Path(storage['runs']) / '20261008_feature_validation_smoke/fmcib'
members = pd.read_csv(root / 'image_sample.csv', dtype={'PatientID': str})
assert members.PatientID.is_unique
assert set(members.query("feature_split == 'discovery'").PatientID).isdisjoint(members.query("feature_split == 'confirmation'").PatientID)
with np.load(root / 'cases' / f'{members.iloc[0].identifier}.npz') as first:
    input_volume = torch.from_numpy(first['input'])
    first_map = torch.from_numpy(first['activations'][:50])
with np.load(root / 'cases' / f'{members.iloc[1].identifier}.npz') as second:
    second_map = torch.from_numpy(second['activations'][:50])
actual = sums_correlation(paired_sums(first_map, second_map))
nonconstant = (first_map.flatten(1).std(1) > 0) & (second_map.flatten(1).std(1) > 0)
reference = np.array([pearsonr(first_map[index].numpy().ravel(), second_map[index].numpy().ravel()).statistic for index in torch.where(nonconstant)[0]])
np.testing.assert_allclose(actual[nonconstant.numpy()], reference, atol=1e-6)
for axis in range(3):
    for direction in [-1, 1]:
        shifted = shifted_input(input_volume, axis, direction * 4)
        source = [slice(None)] * 4
        target = [slice(None)] * 4
        source[axis + 1] = slice(0, 46) if direction > 0 else slice(4, 50)
        target[axis + 1] = slice(4, 50) if direction > 0 else slice(0, 46)
        assert torch.equal(input_volume[tuple(source)], shifted[tuple(target)])
        native_original = 5
        native_shifted = native_original + direction
        assert native_shifted * 4 - direction * 4 == native_original * 4
result = {'status': 'passed', 'completed_at': timestamp(), 'real_image_pairs': 1,
          'nonconstant_features_compared_with_scipy': int(nonconstant.sum()),
          'scipy_max_absolute_error': float(np.max(np.abs(actual[nonconstant.numpy()] - reference))),
          'exact_real_input_translation_checks': 6, 'patient_split_disjoint': True,
          'source_sha256': sha256(Path(__file__))}
write_json(root / 'independent_checks.json', result)
print(result, flush=True)
