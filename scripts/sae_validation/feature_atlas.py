import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from sclvmi.context import context, read_json, sha256, timestamp, write_json


def select_review_features(checks):
    generator = np.random.default_rng(20261009)
    ordered = checks.sort_values(['discovery_frequency', 'feature_id']).feature_id.to_numpy()
    selected = []
    for quartile, indices in enumerate(np.array_split(ordered, 4)):
        selected.extend({'feature_id': int(index), 'selection': f'frequency_quartile_{quartile + 1}'} for index in generator.choice(indices, 6, replace=False))
    discovery = checks.copy()
    discovery['discovery_seed_minimum'] = discovery[['discovery_correlation_s2026', 'discovery_correlation_s2027']].min(axis=1)
    candidates = discovery[(discovery.discovery_active_patients >= 20) & (discovery.discovery_seed_minimum >= .6) & (discovery.discovery_translation_correlation >= .6) & (discovery.discovery_translation_advantage >= .1)]
    candidates = candidates.sort_values(['discovery_seed_minimum', 'feature_id'], ascending=[False, True])
    for feature in candidates.feature_id.head(8):
        if feature not in [row['feature_id'] for row in selected]:
            selected.append({'feature_id': int(feature), 'selection': 'repeatable_translation_candidate'})
    return selected


def section(volume, axis, index):
    return np.take(volume, index, axis=axis).T


def patient_examples(scores, split, feature):
    selected = scores[(scores.feature_split == split) & (scores.feature_id == feature)].sort_values(['maximum', 'identifier'], ascending=[False, True])
    assert selected.PatientID.is_unique
    size = len(selected)
    indices = sorted(set([0, 1, 2, 3, size // 2 - 1, size // 2, size - 2, size - 1]))
    return selected.iloc[indices].copy()


def render_feature(destination, scores, checks, feature, split, scale, scope='full', atlas_name=None):
    examples = patient_examples(scores, split, feature)
    fig, axes = plt.subplots(2, len(examples), figsize=(3 * len(examples), 6.6), squeeze=False)
    orthogonal_fig, orthogonal_axes = plt.subplots(3, len(examples), figsize=(3 * len(examples), 9.3), squeeze=False)
    example_rows = []
    for column, row in enumerate(examples.itertuples()):
        with np.load(destination / 'cases' / f'{row.identifier}.npz') as saved:
            volume = saved['input'][0] * 3072 - 1024
            activation = saved['activations'][feature]
        if scope == 'interior':
            activation = activation.copy()
            activation[:2] = 0
            activation[-2:] = 0
            activation[:, :2] = 0
            activation[:, -2:] = 0
            activation[:, :, :2] = 0
            activation[:, :, -2:] = 0
        geometry = read_json(destination / 'cases' / f'{row.identifier}.json')
        affine = np.asarray(geometry['input_affine_ras'])
        axis = int(np.argmax(np.abs(affine[2, :3])))
        peak = np.asarray(np.unravel_index(int(row.peak_index), activation.shape))
        if row.maximum == 0:
            peak = np.asarray(activation.shape) // 2
        input_peak = np.minimum(peak * 4, np.asarray(volume.shape) - 1)
        plane = section(volume, axis, int(input_peak[axis]))
        overlay = section(activation, axis, int(peak[axis]))
        plane_axes = [value for value in range(3) if value != axis]
        coords = np.meshgrid(np.arange(plane.shape[0]) / 4, np.arange(plane.shape[1]) / 4, indexing='ij')
        interpolated = map_coordinates(overlay, coords, order=1, mode='nearest')
        for r in [0, 1]:
            axes[r, column].imshow(plane, cmap='gray', vmin=-1000, vmax=400, origin='lower')
            axes[r, column].set_xticks([])
            axes[r, column].set_yticks([])
        alpha = .62 * np.clip(interpolated / scale, 0, 1)
        axes[1, column].imshow(interpolated, cmap='magma', vmin=0, vmax=scale, alpha=alpha, origin='lower')
        for r in [0, 1]:
            axes[r, column].plot(input_peak[plane_axes[0]], input_peak[plane_axes[1]], '+', color='#00c5ff', markersize=8)
        rank = scores[(scores.feature_split == split) & (scores.feature_id == feature)].sort_values(['maximum', 'identifier'], ascending=[False, True]).index.get_loc(row.Index)
        axes[0, column].set_title(f'Patient example {column + 1}\nrank {rank + 1}/{len(scores[(scores.feature_split == split) & (scores.feature_id == feature)])}; max {row.maximum:.3f}', fontsize=10)
        axes[1, column].set_xlabel(f'Native peak {tuple(int(value) for value in peak)}', fontsize=9)
        for world_axis, label in enumerate(['Sagittal', 'Coronal', 'Axial']):
            view_axis = int(np.argmax(np.abs(affine[world_axis, :3])))
            view = section(volume, view_axis, int(input_peak[view_axis]))
            heatmap = section(activation, view_axis, int(peak[view_axis]))
            view_coords = np.meshgrid(np.arange(view.shape[0]) / 4, np.arange(view.shape[1]) / 4, indexing='ij')
            heatmap = map_coordinates(heatmap, view_coords, order=1, mode='nearest')
            ax = orthogonal_axes[world_axis, column]
            ax.imshow(view, cmap='gray', vmin=-1000, vmax=400, origin='lower')
            if heatmap.max() > scale * .5 and heatmap.min() < scale * .5:
                ax.contour(heatmap, levels=[scale * .5], colors=['#ff4c25'], linewidths=1.1, origin='lower')
            view_axes = [value for value in range(3) if value != view_axis]
            ax.plot(input_peak[view_axes[0]], input_peak[view_axes[1]], '+', color='#00c5ff', markersize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{label}; rank {rank + 1}; max {row.maximum:.3f}', fontsize=10)
        example_rows.append({'feature_id': feature, 'split': split, 'example': column + 1, 'identifier': row.identifier,
                             'PatientID': row.PatientID, 'rank': rank + 1, 'peak_grid': peak.tolist(), 'peak_input': input_peak.tolist(),
                             'physical_ras': nib.affines.apply_affine(affine, input_peak).tolist(), 'maximum': row.maximum})
    data = checks.set_index('feature_id').loc[feature]
    seed_minimum = min(data[f'{split}_correlation_s2026'], data[f'{split}_correlation_s2027'])
    fig.suptitle(f'FMCIB feature {feature} | {split} | {scope} grid | same activation scale 0–{scale:.3f}\nSeed correlation min {seed_minimum:.3f} | translation {data[f"{split}_translation_correlation"]:.3f} | position variance {data[f"{split}_position_variance_fraction"]:.3f}', fontsize=14)
    fig.text(.01, .012, 'Top: model-input CT (HU window -1000 to 400). Bottom: interpolated native activation. Cyan cross: native peak location. No clinical labels.', fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .90])
    target = destination / (atlas_name or ('atlas' if scope == 'full' else 'atlas_interior')) / split / f'feature_{feature:04d}.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    orthogonal_fig.suptitle(f'FMCIB feature {feature} | {split} | {scope} grid | orthogonal sections through the strongest native location\nRed contour: activation {scale * .5:.3f}, fixed from discovery. Inactive cases use the central location.', fontsize=14)
    orthogonal_fig.tight_layout(rect=[0, 0, 1, .92])
    orthogonal_fig.savefig(target.with_name(f'feature_{feature:04d}_orthogonal.png'), dpi=110)
    plt.close(orthogonal_fig)
    return example_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    parser.add_argument('--split', choices=['discovery', 'confirmation'], default='discovery')
    parser.add_argument('--features', nargs='*', type=int)
    args = parser.parse_args()
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / 'fmcib'
    checks = pd.read_csv(destination / 'feature_checks.csv')
    scores = pd.read_csv(destination / 'image_scores.csv', dtype={'PatientID': str})
    selection_path = destination / 'review_selection.json'
    if selection_path.exists():
        selection = read_json(selection_path)['features']
    else:
        selection = select_review_features(checks)
        write_json(selection_path, {'features': selection, 'created_at': timestamp(), 'seed': 20261009, 'method': 'Discovery only: six features per frequency quartile plus eight highest repeatable/translation candidates, with duplicate IDs removed'})
    features = args.features or [row['feature_id'] for row in selection]
    scale_path = destination / 'review_scales.json'
    if scale_path.exists():
        scales = read_json(scale_path)
    else:
        scales = {}
    for feature in features:
        if str(feature) not in scales:
            reference = scores[(scores.feature_split == 'discovery') & (scores.feature_id == feature)].maximum
            scales[str(feature)] = max(float(reference.quantile(.95)), 1e-5)
    write_json(scale_path, scales)
    if args.split == 'confirmation':
        assert (destination / 'discovery_interpretations.json').exists(), 'Freeze discovery interpretations before confirmation image review'
    records = []
    for feature in tqdm(features, desc=f'{args.split} feature images', mininterval=3):
        records.extend(render_feature(destination, scores, checks, feature, args.split, scales[str(feature)]))
    write_json(destination / 'atlas' / args.split / 'image_manifest.json', {'examples': records, 'created_at': timestamp(), 'source_sha256': sha256(Path(__file__)), 'private_patient_data': True})
    print({'images': len(features), 'split': args.split, 'directory': str(destination / 'atlas' / args.split)}, flush=True)


if __name__ == '__main__':
    main()
