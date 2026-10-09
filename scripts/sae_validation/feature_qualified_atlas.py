import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from feature_atlas import render_feature
from sclvmi.context import context, sha256, timestamp, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='20261008_feature_validation')
    parser.add_argument('--split', choices=['discovery', 'confirmation'], default='discovery')
    args = parser.parse_args()
    storage, _, _ = context()
    destination = Path(storage['runs']) / args.name / 'fmcib'
    checks = pd.read_csv(destination / 'feature_patient_checks.csv')
    scores = pd.read_csv(destination / 'interior_image_scores.csv', dtype={'PatientID': str})
    features = checks.loc[checks.qualified_numeric_feature, 'feature_id'].tolist()
    if args.split == 'confirmation':
        assert (destination / 'qualified_discovery_interpretations.json').exists()
    for split in ['discovery', 'confirmation']:
        for seed in [2026, 2027]:
            checks[f'{split}_correlation_s{seed}'] = checks[f'{split}_patient_raw_s{seed}_mean']
    records = []
    for feature in tqdm(features, desc=f'{args.split} qualified feature images', mininterval=3):
        scale = max(float(scores[(scores.feature_split == 'discovery') & (scores.feature_id == feature)].maximum.quantile(.95)), 1e-5)
        records.extend(render_feature(destination, scores, checks, feature, args.split, scale, scope='interior', atlas_name='atlas_qualified'))
    write_json(destination / 'atlas_qualified' / args.split / 'image_manifest.json',
               {'examples': records, 'created_at': timestamp(), 'source_sha256': sha256(Path(__file__)),
                'renderer_sha256': sha256(Path(__file__).with_name('feature_atlas.py')), 'private_patient_data': True,
                'selection': 'All thirteen features passing numerical screens in both halves; conditional descriptive image review'})


if __name__ == '__main__':
    main()
