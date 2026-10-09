import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from feature_screen import correlation
from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / 'fmcib'
    screen = read_json(destination / 'screen.json')
    assert not (destination / 'interior_checks.json').exists()
    dictionaries = [SpatialDictionary(Path(storage['runs']) / '20261008_spatial_sae' / run / 'dictionary.pt') for run in screen['runs']]
    members = pd.read_csv(destination / 'image_sample.csv', dtype={'PatientID': str})
    source = Path(storage['activations']) / '20261008_spatial_sae/fmcib/cases'
    groups = {}
    result = pd.read_csv(destination / 'feature_checks.csv')
    spatial_means = []
    for split in ['discovery', 'confirmation']:
        selected = members[members.feature_split == split]
        values = [torch.empty((len(selected), 729, 2048), device='cuda') for _ in dictionaries]
        for patient_index, row in enumerate(tqdm(selected.itertuples(), total=len(selected), desc=f'{split} interior encoding', mininterval=3)):
            with np.load(source / f'{row.identifier}.npz') as saved:
                features = torch.tensor(saved['layer1'][:, 2:-2, 2:-2, 2:-2].copy(), device='cuda').flatten(1).T
            assert features.shape == (729, 512)
            for seed_index, dictionary in enumerate(dictionaries):
                values[seed_index][patient_index] = dictionary.encode(features)
        result[f'{split}_interior_active_patients'] = (values[0] > 0).any(1).sum(0).cpu().numpy()
        result[f'{split}_interior_frequency'] = (values[0] > 0).float().mean((0, 1)).cpu().numpy()
        if split == 'discovery':
            spatial_means = [value.mean(0) for value in values]
        groups[split] = values
    targets = {}
    for mode in ['raw', 'position_residual']:
        for seed_index in [1, 2]:
            seed = dictionaries[seed_index].request['seed']
            discovery_values = groups['discovery']
            confirmation_values = groups['confirmation']
            if mode == 'position_residual':
                first = (discovery_values[0] - spatial_means[0]).flatten(0, 1)
                second = (discovery_values[seed_index] - spatial_means[seed_index]).flatten(0, 1)
            else:
                first = discovery_values[0].flatten(0, 1)
                second = discovery_values[seed_index].flatten(0, 1)
            matched = correlation(first, second)
            counterpart = matched.argmax(1)
            indices = torch.arange(2048, device='cuda')
            result[f'{mode}_counterpart_s{seed}'] = counterpart.cpu().numpy()
            result[f'discovery_{mode}_correlation_s{seed}'] = matched[indices, counterpart].cpu().numpy()
            if mode == 'position_residual':
                first = (confirmation_values[0] - spatial_means[0]).flatten(0, 1)
                second = (confirmation_values[seed_index] - spatial_means[seed_index]).flatten(0, 1)
            else:
                first = confirmation_values[0].flatten(0, 1)
                second = confirmation_values[seed_index].flatten(0, 1)
            confirmed = correlation(first, second)
            result[f'confirmation_{mode}_correlation_s{seed}'] = confirmed[indices, counterpart].cpu().numpy()
        columns = [column for column in result if f'_{mode}_correlation_' in column]
        result[f'{mode}_minimum_correlation'] = result[columns].min(axis=1)
    result['interior_screen_pass'] = ((result.discovery_interior_active_patients >= 20) &
                                      (result.confirmation_interior_active_patients >= 20) &
                                      (result.raw_minimum_correlation >= .6) &
                                      (result.position_residual_minimum_correlation >= .6) & result.translation_screen_pass)
    result.to_csv(destination / 'feature_interior_checks.csv', index=False)
    np.savez(destination / 'discovery_interior_position_means.npz', **{f's{dictionary.request["seed"]}': mean.cpu().numpy() for dictionary, mean in zip(dictionaries, spatial_means)})
    summary = {'status': 'completed', 'completed_at': timestamp(), 'features': len(result),
               'interior_screen_pass': int(result.interior_screen_pass.sum()),
               'interior_grid': [9, 9, 9], 'patients_per_half': 64,
               'mean_template_fitted_on': 'discovery only', 'source_sha256': sha256(Path(__file__)),
               'position_residual_minimum_correlation_quantiles': result.position_residual_minimum_correlation.quantile([0, .1, .5, .9, 1]).to_dict(),
               'test_used': False}
    write_json(destination / 'interior_checks.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
