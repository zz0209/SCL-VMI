import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary


def normalize_patient(values, active, mean=None):
    residual = values if mean is None else values - mean
    centered = residual - residual.mean(1, keepdim=True)
    norms = centered.norm(dim=1)
    valid = active & (norms > 1e-8)
    normalized = centered / norms[:, None, :].clamp_min(1e-8)
    return normalized * valid[:, None, :], valid


def patient_correlation(first, first_valid, second, second_valid):
    counts = first_valid.float().T @ second_valid.float()
    totals = first.flatten(0, 1).T @ second.flatten(0, 1)
    return totals / counts.clamp_min(1), counts


def paired_values(first, first_valid, second, second_valid, targets):
    valid = first_valid & second_valid[:, targets]
    values = (first * second[:, :, targets]).sum(1) * valid
    return values.cpu().numpy(), valid.cpu().numpy()


def bootstrap(values, valid):
    generator = np.random.default_rng(20261009)
    weights = generator.multinomial(len(values), np.full(len(values), 1 / len(values)), size=1000)
    counts = weights @ valid.astype(float)
    totals = weights @ values
    estimates = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    intervals = np.full((2, values.shape[1]), np.nan)
    eligible = valid.sum(0) >= 20
    intervals[:, eligible] = np.nanquantile(estimates[:, eligible], [.025, .975], axis=0)
    return intervals


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / 'fmcib'
    assert not (destination / 'patient_repeatability.json').exists()
    screen = read_json(destination / 'screen.json')
    dictionaries = [SpatialDictionary(Path(storage['runs']) / '20261008_spatial_sae' / run / 'dictionary.pt') for run in screen['runs']]
    members = pd.read_csv(destination / 'image_sample.csv', dtype={'PatientID': str})
    source = Path(storage['activations']) / '20261008_spatial_sae/fmcib/cases'
    groups = {}
    output = pd.read_csv(destination / 'feature_interior_checks.csv')
    for split in ['discovery', 'confirmation']:
        selected = members[members.feature_split == split]
        values = [torch.empty((len(selected), 729, 2048), device='cuda') for _ in dictionaries]
        for patient_index, row in enumerate(tqdm(selected.itertuples(), total=len(selected), desc=f'{split} patient patterns', mininterval=3)):
            with np.load(source / f'{row.identifier}.npz') as saved:
                features = torch.tensor(saved['layer1'][:, 2:-2, 2:-2, 2:-2].copy(), device='cuda').flatten(1).T
            for seed_index, dictionary in enumerate(dictionaries):
                values[seed_index][patient_index] = dictionary.encode(features)
        groups[split] = values
    means = [values.mean(0) for values in groups['discovery']]
    mean_columns = []
    lower_columns = []
    count_columns = []
    for mode in ['raw', 'residual']:
        normalized = {}
        for split in ['discovery', 'confirmation']:
            normalized[split] = [normalize_patient(values, (values > 0).any(1), None if mode == 'raw' else means[index]) for index, values in enumerate(groups[split])]
        for seed_index in [1, 2]:
            seed = dictionaries[seed_index].request['seed']
            first, first_valid = normalized['discovery'][0]
            second, second_valid = normalized['discovery'][seed_index]
            correlations, counts = patient_correlation(first, first_valid, second, second_valid)
            eligible = counts >= 20
            selected = correlations.masked_fill(~eligible, -2).argmax(1)
            output[f'patient_{mode}_counterpart_s{seed}'] = selected.cpu().numpy()
            for split in ['discovery', 'confirmation']:
                first, first_valid = normalized[split][0]
                second, second_valid = normalized[split][seed_index]
                values, valid = paired_values(first, first_valid, second, second_valid, selected)
                count = valid.sum(0)
                mean = np.divide(values.sum(0), count, out=np.full(2048, np.nan), where=count > 0)
                prefix = f'{split}_patient_{mode}_s{seed}'
                output[f'{prefix}_mean'] = mean
                output[f'{prefix}_patients'] = count
                mean_columns.append(f'{prefix}_mean')
                count_columns.append(f'{prefix}_patients')
                np.savez(destination / f'{prefix}_values.npz', correlations=values, valid=valid, counterparts=selected.cpu().numpy())
                if split == 'confirmation':
                    intervals = bootstrap(values, valid)
                    output[f'{prefix}_lower95'] = intervals[0]
                    output[f'{prefix}_upper95'] = intervals[1]
                    lower_columns.append(f'{prefix}_lower95')
            print(f'Completed {mode} patient correspondence with seed {seed}', flush=True)
    output['patient_mean_minimum'] = output[mean_columns].min(axis=1, skipna=False)
    output['patient_confirmation_lower95_minimum'] = output[lower_columns].min(axis=1, skipna=False)
    output['paired_patient_minimum'] = output[count_columns].min(axis=1)
    output['patient_repeatability_pass'] = ((output.patient_mean_minimum >= .6) & (output.patient_confirmation_lower95_minimum >= .5) & (output.paired_patient_minimum >= 20))
    output['qualified_numeric_feature'] = output.interior_screen_pass & output.patient_repeatability_pass
    output.to_csv(destination / 'feature_patient_checks.csv', index=False)
    summary = {'status': 'completed', 'completed_at': timestamp(), 'patient_repeatability_pass': int(output.patient_repeatability_pass.sum()),
               'qualified_numeric_features': int(output.qualified_numeric_feature.sum()), 'features': len(output),
               'method': 'Mean within-patient spatial Pearson correlation among coactive nonconstant patient pairs; discovery-selected counterparts; 1000 paired-patient bootstrap replicates',
               'source_sha256': sha256(Path(__file__)), 'test_used': False}
    write_json(destination / 'patient_repeatability.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
