import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_sites import SITES, SpatialEncoder, volume_with_geometry


def shifted_input(tensor, axis, amount):
    result = torch.zeros_like(tensor)
    source = [slice(None)] * 4
    target = [slice(None)] * 4
    source[axis + 1] = slice(None, -amount) if amount > 0 else slice(-amount, None)
    target[axis + 1] = slice(amount, None) if amount > 0 else slice(None, amount)
    result[tuple(target)] = tensor[tuple(source)]
    return result


def paired_sums(first, second):
    first = first.flatten(1).double()
    second = second.flatten(1).double()
    return torch.stack([torch.full_like(first[:, 0], first.shape[1]), first.sum(1), second.sum(1),
                        first.square().sum(1), second.square().sum(1), (first * second).sum(1)]).cpu().numpy()


def sums_correlation(values):
    count, first, second, first_square, second_square, product = values
    variance = np.maximum(first_square - first ** 2 / count, 0) * np.maximum(second_square - second ** 2 / count, 0)
    result = np.full(first.shape, np.nan)
    np.divide(product - first * second / count, np.sqrt(variance), out=result, where=variance > 1e-16)
    return result


def summarize(destination, members, dictionary):
    feature_count = dictionary.request['dict_size']
    results = pd.read_csv(destination / 'feature_screen.csv')
    all_scores = []
    for split in ['discovery', 'confirmation']:
        selected = members[members.feature_split == split]
        accumulated = None
        square_sum = np.zeros(feature_count)
        sums = np.zeros(feature_count)
        total_positions = 0
        aligned = np.zeros((6, feature_count))
        fixed = np.zeros((6, feature_count))
        edge_sum = np.zeros(feature_count)
        for row in tqdm(selected.itertuples(), total=len(selected), desc=f'{split} spatial summaries', mininterval=3):
            with np.load(destination / 'cases' / f'{row.identifier}.npz') as saved:
                activities = saved['activations'].astype(np.float64)
                assert np.isfinite(activities).all()
                if accumulated is None:
                    accumulated = np.zeros_like(activities)
                accumulated += activities
                flattened = activities.reshape(feature_count, -1)
                sums += flattened.sum(1)
                square_sum += np.square(flattened).sum(1)
                total_positions += flattened.shape[1]
                edge_mask = np.ones(activities.shape[1:], dtype=bool)
                edge_mask[1:-1, 1:-1, 1:-1] = False
                edge_sum += activities[:, edge_mask].sum(1)
                aligned += saved['aligned_sums']
                fixed += saved['fixed_sums']
                maxima = flattened.max(1)
                means = flattened.mean(1)
                peak = flattened.argmax(1)
                all_scores.append(pd.DataFrame({'identifier': row.identifier, 'PatientID': row.PatientID,
                                                'feature_split': split, 'feature_id': np.arange(feature_count),
                                                'maximum': maxima, 'mean': means, 'peak_index': peak}))
        spatial_mean = accumulated / len(selected)
        total_variance = square_sum / total_positions - (sums / total_positions) ** 2
        position_variance = spatial_mean.reshape(feature_count, -1).var(1)
        explained = np.full(feature_count, np.nan)
        np.divide(position_variance, total_variance, out=explained, where=total_variance > 1e-16)
        results[f'{split}_position_variance_fraction'] = explained
        results[f'{split}_translation_correlation'] = sums_correlation(aligned)
        results[f'{split}_fixed_position_correlation'] = sums_correlation(fixed)
        results[f'{split}_translation_advantage'] = results[f'{split}_translation_correlation'] - results[f'{split}_fixed_position_correlation']
        results[f'{split}_boundary_activation_fraction'] = np.divide(edge_sum, sums, out=np.full(feature_count, np.nan), where=sums > 0)
        np.savez(destination / f'{split}_spatial_statistics.npz', spatial_mean=spatial_mean.astype(np.float32),
                 aligned_sums=aligned, fixed_sums=fixed, activation_sum=sums, activation_square_sum=square_sum)
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(destination / 'image_scores.csv', index=False)
    for constant in ['zero', 'midpoint']:
        with np.load(destination / f'{constant}_input.npz') as saved:
            response = saved['activations'].reshape(feature_count, -1).max(1)
        reference = scores[scores.feature_split == 'confirmation'].groupby('feature_id').maximum.median().reindex(np.arange(feature_count)).to_numpy()
        results[f'{constant}_input_maximum'] = response
        results[f'{constant}_input_to_median_patient_max_ratio'] = np.divide(response, reference, out=np.full(feature_count, np.nan), where=reference > 0)
    results['translation_screen_pass'] = (results.confirmation_translation_correlation >= .6) & (results.confirmation_translation_advantage >= .1)
    results['repeatability_and_translation_pass'] = results.repeatability_screen_pass & results.translation_screen_pass
    results.to_csv(destination / 'feature_checks.csv', index=False)
    summary = {'status': 'completed', 'completed_at': timestamp(), 'patients': len(members),
               'features': feature_count, 'repeatability_screen_pass': int(results.repeatability_screen_pass.sum()),
               'translation_screen_pass': int(results.translation_screen_pass.sum()),
               'repeatability_and_translation_pass': int(results.repeatability_and_translation_pass.sum()),
               'confirmation_translation_correlation_quantiles': results.confirmation_translation_correlation.quantile([0, .1, .5, .9, 1]).to_dict(),
               'source_sha256': sha256(Path(__file__)), 'dictionary_sha256': sha256(dictionary.path), 'test_used': False}
    write_json(destination / 'image_checks.json', summary)
    print(summary, flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    parser.add_argument('--model', default='fmcib')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / args.model
    screen = read_json(destination / 'screen.json')
    assert screen['smoke'] == args.smoke
    dictionary = SpatialDictionary(screen['default_dictionary'])
    assert sha256(dictionary.path) == screen['dictionary_sha256']
    members = pd.read_csv(destination / 'image_sample.csv', dtype={'PatientID': str})
    assert members.PatientID.is_unique and set(members.split) == {'development'}
    site = dictionary.request['site']
    encoder = SpatialEncoder(args.model)
    cases = destination / 'cases'
    cases.mkdir(exist_ok=True)
    stride = SITES[args.model][site]['stride']
    source_cases = Path(storage['activations']) / '20261008_spatial_sae' / args.model / 'cases'
    started = time.perf_counter()
    for index, row in enumerate(tqdm(members.itertuples(), total=len(members), desc='Real CT and paired translations', mininterval=3)):
        target = cases / f'{row.identifier}.npz'
        if target.exists() and target.with_suffix('.json').exists():
            metadata = read_json(target.with_suffix('.json'))
            assert metadata['source_sha256'] == sha256(Path(__file__))
            continue
        tensor, geometry = volume_with_geometry(row.image, args.model)
        features = encoder.extract(tensor, sites=[site])[site][0]
        with np.load(source_cases / f'{row.identifier}.npz') as cached:
            np.testing.assert_allclose(features.cpu().numpy(), cached[site], rtol=1e-4, atol=1e-5)
        activations = dictionary.spatial_activations(features)
        aligned_sums = np.zeros((6, dictionary.request['dict_size']))
        fixed_sums = np.zeros_like(aligned_sums)
        for axis in range(3):
            for direction in [-1, 1]:
                shifted = shifted_input(tensor, axis, direction * stride)
                shifted_features = encoder.extract(shifted, sites=[site])[site][0]
                translated = dictionary.spatial_activations(shifted_features)
                original_slices = [slice(None), slice(2, -2), slice(2, -2), slice(2, -2)]
                translated_slices = original_slices.copy()
                size = activations.shape[axis + 1]
                original_slices[axis + 1] = slice(2, size - 3) if direction > 0 else slice(3, size - 2)
                translated_slices[axis + 1] = slice(3, size - 2) if direction > 0 else slice(2, size - 3)
                original = activations[tuple(original_slices)]
                aligned_sums += paired_sums(original, translated[tuple(translated_slices)])
                fixed_sums += paired_sums(original, translated[tuple(original_slices)])
        with target.with_suffix('.tmp').open('wb') as handle:
            np.savez(handle, input=tensor.cpu().numpy(), activations=activations.cpu().numpy(),
                     aligned_sums=aligned_sums, fixed_sums=fixed_sums)
        target.with_suffix('.tmp').replace(target)
        geometry.update({'identifier': row.identifier, 'PatientID': row.PatientID, 'feature_split': row.feature_split,
                         'source_sha256': sha256(Path(__file__)), 'image': row.image, 'cache_forward_match': True})
        write_json(target.with_suffix('.json'), geometry)
        write_json(destination / 'image_progress.json', {'completed': index + 1, 'total': len(members), 'elapsed_seconds': time.perf_counter() - started, 'updated_at': timestamp()})
        del features, activations, shifted_features, translated
    for label, value in [('zero', 0.0), ('midpoint', 0.5)]:
        target = destination / f'{label}_input.npz'
        if target.exists():
            continue
        settings = read_json(next(cases.glob('*.json')))
        constant = torch.full((1, *settings['input_shape']), value)
        features = encoder.extract(constant, sites=[site])[site][0]
        np.savez(target, activations=dictionary.spatial_activations(features).cpu().numpy())
    summarize(destination, members, dictionary)


if __name__ == '__main__':
    main()
