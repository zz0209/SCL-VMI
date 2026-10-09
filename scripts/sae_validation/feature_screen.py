import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary


def correlation(first, second):
    first = first - first.mean(0)
    second = second - second.mean(0)
    first = first / first.norm(dim=0).clamp_min(1e-12)
    second = second / second.norm(dim=0).clamp_min(1e-12)
    return first.T @ second


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vista', 'fmcib'], default='fmcib')
    parser.add_argument('--name', default='20261008_feature_validation')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'results/20261008_sae_feature_validation/README.md')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    storage, _, _ = context()
    suite = Path(storage['runs']) / '20261008_spatial_sae'
    selection_name = 'fmcib_selection_candidate' if args.model == 'fmcib' else 'vista_selection_activity80'
    runs = read_json(suite / f'{selection_name}.json')['selected']['runs']
    dictionaries = [SpatialDictionary(suite / run / 'dictionary.pt') for run in runs]
    request = dictionaries[0].request
    matrix = Path(request['matrix'])
    members = pd.read_csv(matrix.parent / 'full_members.csv', dtype={'PatientID': str})
    members = members[members.split == 'development'].copy()
    patients = np.array(sorted(members.PatientID.unique()))
    assert len(patients) == 422
    generator = np.random.default_rng(20261009)
    order = generator.permutation(len(patients))
    discovery_indices = np.sort(order[:len(patients) // 2])
    confirmation_indices = np.sort(order[len(patients) // 2:])
    if args.smoke:
        discovery_indices = discovery_indices[:4]
        confirmation_indices = confirmation_indices[:4]
    destination = Path(storage['runs']) / args.name / args.model
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / 'screen.json'
    assert not receipt.exists(), 'Completed screening requires a new run name'
    source = np.load(matrix / 'development.npy', mmap_mode='r')
    assert len(source) == len(patients) * 256
    group_indices = [np.concatenate([np.arange(index * 256, (index + 1) * 256) for index in indices]) for indices in [discovery_indices, confirmation_indices]]
    split_names = ['discovery', 'confirmation']
    groups = {}
    for split, indices in zip(split_names, [discovery_indices, confirmation_indices]):
        for index in indices:
            groups[patients[index]] = split
    members = members[members.PatientID.isin(groups)].copy()
    members['feature_split'] = members.PatientID.map(groups)
    members.to_csv(destination / 'members.csv', index=False)
    output = {'feature_id': np.arange(request['dict_size'])}
    encoded = {}
    for split, indices in zip(split_names, group_indices):
        tokens = torch.tensor(np.array(source[indices], copy=True), device='cuda')
        encoded[split] = []
        for dictionary in tqdm(dictionaries, desc=f'{split} seed encoding', mininterval=1):
            values = torch.cat([dictionary.encode(batch) for batch in tokens.split(2048)])
            encoded[split].append(values)
        default = encoded[split][0]
        patient_counts = (default.reshape(-1, 256, request['dict_size']) > 0).any(1).sum(0)
        output[f'{split}_active_patients'] = patient_counts.cpu().numpy()
        output[f'{split}_frequency'] = (default > 0).float().mean(0).cpu().numpy()
        output[f'{split}_mean'] = default.mean(0).cpu().numpy()
        output[f'{split}_maximum'] = default.amax(0).cpu().numpy()
        del tokens
    duplicate_counts = {}
    for index in [1, 2]:
        seed = dictionaries[index].request['seed']
        discovery_correlation = correlation(encoded['discovery'][0], encoded['discovery'][index])
        targets = discovery_correlation.argmax(1)
        rows = torch.arange(request['dict_size'], device='cuda')
        confirmation_correlation = correlation(encoded['confirmation'][0], encoded['confirmation'][index])
        output[f'counterpart_s{seed}'] = targets.cpu().numpy()
        output[f'discovery_correlation_s{seed}'] = discovery_correlation[rows, targets].cpu().numpy()
        output[f'confirmation_correlation_s{seed}'] = confirmation_correlation[rows, targets].cpu().numpy()
        duplicate_counts[str(seed)] = request['dict_size'] - len(torch.unique(targets))
    table = pd.DataFrame(output)
    enough_patients = (table.discovery_active_patients >= 20) & (table.confirmation_active_patients >= 20)
    correlation_columns = [column for column in table if '_correlation_' in column]
    table['minimum_seed_correlation'] = table[correlation_columns].min(axis=1)
    table['repeatability_screen_pass'] = enough_patients & (table.minimum_seed_correlation >= 0.6)
    table.to_csv(destination / 'feature_screen.csv', index=False)
    sample_members = []
    for split in split_names:
        eligible = members[members.feature_split == split]
        selected_patients = generator.permutation(sorted(eligible.PatientID.unique()))[:64]
        for patient in selected_patients:
            rows = eligible[eligible.PatientID == patient].sort_values('identifier')
            sample_members.append(rows.iloc[int(generator.integers(len(rows)))])
    pd.DataFrame(sample_members).to_csv(destination / 'image_sample.csv', index=False)
    result = {'status': 'completed', 'created_at': timestamp(), 'model': args.model, 'runs': runs,
              'patients': {name: len(indices) for name, indices in zip(split_names, [discovery_indices, confirmation_indices])},
              'default_dictionary': str(dictionaries[0].path), 'dictionary_sha256': sha256(dictionaries[0].path),
              'features': len(table), 'repeatability_screen_pass': int(table.repeatability_screen_pass.sum()),
              'active_in_20_patients_each_half': int(enough_patients.sum()),
              'counterpart_duplicate_targets': duplicate_counts, 'test_used': False,
              'minimum_correlation_quantiles': table.minimum_seed_correlation.quantile([0, .1, .5, .9, 1]).to_dict(),
              'source_sha256': sha256(Path(__file__)), 'matrix_identity': read_json(matrix / 'matrix.json'),
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
              'design_sha256': sha256(args.protocol), 'smoke': args.smoke}
    write_json(receipt, result)
    print({key: result[key] for key in ['model', 'features', 'repeatability_screen_pass', 'active_in_20_patients_each_half', 'minimum_correlation_quantiles']}, flush=True)


if __name__ == '__main__':
    main()
