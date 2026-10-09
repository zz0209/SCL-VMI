import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_atlas import render_feature
from sclvmi.context import context, read_json, sha256, timestamp, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    parser.add_argument('--split', choices=['discovery', 'confirmation'], default='discovery')
    args = parser.parse_args()
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / 'fmcib'
    checks = pd.read_csv(destination / 'feature_interior_checks.csv')
    score_path = destination / 'interior_image_scores.csv'
    if score_path.exists():
        scores = pd.read_csv(score_path, dtype={'PatientID': str})
    else:
        members = pd.read_csv(destination / 'image_sample.csv', dtype={'PatientID': str})
        tables = []
        for row in tqdm(members.itertuples(), total=len(members), desc='Interior reference-image scores', mininterval=3):
            with np.load(destination / 'cases' / f'{row.identifier}.npz') as saved:
                activities = saved['activations'][:, 2:-2, 2:-2, 2:-2].reshape(2048, -1)
            peaks = np.unravel_index(activities.argmax(1), (9, 9, 9))
            global_peaks = np.ravel_multi_index(tuple(value + 2 for value in peaks), (13, 13, 13))
            tables.append(pd.DataFrame({'feature_id': np.arange(2048), 'identifier': row.identifier, 'PatientID': row.PatientID,
                                       'feature_split': row.feature_split, 'maximum': activities.max(1), 'mean': activities.mean(1), 'peak_index': global_peaks}))
        scores = pd.concat(tables, ignore_index=True)
        scores.to_csv(score_path, index=False)
    selection_path = destination / 'interior_review_selection.json'
    if selection_path.exists():
        selection = read_json(selection_path)
    else:
        columns = [f'discovery_{mode}_correlation_s{seed}' for mode in ['raw', 'position_residual'] for seed in [2026, 2027]]
        checks['discovery_interior_minimum'] = checks[columns].min(axis=1)
        candidates = checks[(checks.discovery_interior_active_patients >= 20) & (checks.discovery_interior_minimum >= .6) &
                            (checks.discovery_translation_correlation >= .6) & (checks.discovery_translation_advantage >= .1)]
        features = candidates.sort_values(['discovery_interior_minimum', 'feature_id'], ascending=[False, True]).feature_id.head(16).tolist()
        selection = {'created_at': timestamp(), 'features': features, 'method': 'Sixteen highest discovery-only minimum raw/residual correlations among discovery-qualified features; confirmation outcomes not used'}
        write_json(selection_path, selection)
    if args.split == 'confirmation':
        assert (destination / 'discovery_interpretations_interior.json').exists()
    scales_path = destination / 'interior_review_scales.json'
    if scales_path.exists():
        scales = read_json(scales_path)
    else:
        scales = {str(feature): max(float(scores[(scores.feature_split == 'discovery') & (scores.feature_id == feature)].maximum.quantile(.95)), 1e-5) for feature in selection['features']}
        write_json(scales_path, scales)
    for split in ['discovery', 'confirmation']:
        for seed in [2026, 2027]:
            checks[f'{split}_correlation_s{seed}'] = checks[f'{split}_position_residual_correlation_s{seed}']
    records = []
    for feature in tqdm(selection['features'], desc=f'{args.split} interior image review', mininterval=3):
        records.extend(render_feature(destination, scores, checks, feature, args.split, scales[str(feature)], scope='interior'))
    write_json(destination / 'atlas_interior' / args.split / 'image_manifest.json',
               {'examples': records, 'created_at': timestamp(), 'source_sha256': sha256(Path(__file__)), 'renderer_sha256': sha256(Path(__file__).with_name('feature_atlas.py')), 'private_patient_data': True})


if __name__ == '__main__':
    main()
