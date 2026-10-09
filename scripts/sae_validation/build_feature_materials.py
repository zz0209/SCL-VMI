from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


def main():
    storage, _, _ = context()
    root = Path(storage['runs']) / '20261008_feature_validation/fmcib'
    output = ROOT / 'results/20261008_sae_feature_validation'
    output.mkdir(parents=True, exist_ok=True)
    checks = pd.read_csv(root / 'feature_patient_checks.csv')
    hypotheses = read_json(root / 'qualified_discovery_interpretations.json')
    reviews = read_json(root / 'qualified_confirmation_review.json')
    original = read_json(root / 'confirmation_review_partial.json')['features'] + read_json(root / 'confirmation_review_remaining.json')['features']
    assert len(original) == len({row['feature_id'] for row in original}) == 32
    numeric = set(checks.loc[checks.qualified_numeric_feature, 'feature_id'])
    assert numeric == {row['feature_id'] for row in reviews['features']}
    catalog = checks.merge(pd.DataFrame(hypotheses['features']), on='feature_id', validate='one_to_one').merge(pd.DataFrame(reviews['features']), on='feature_id', validate='one_to_one')
    catalog.to_csv(output / 'reviewed_features.csv', index=False)
    checks.to_csv(output / 'all_feature_checks.csv', index=False)
    starters = catalog[catalog.use == 'content_starter'].feature_id.tolist()
    source = read_json(root / 'screen.json')
    independent = read_json(root / 'patient_independent_checks.json')
    assert independent['status'] == 'passed'
    assert sha256(Path(source['default_dictionary'])) == source['dictionary_sha256']
    for split in ['discovery', 'confirmation']:
        for feature in numeric:
            for suffix in ['', '_orthogonal']:
                assert (root / 'atlas_qualified' / split / f'feature_{feature:04d}{suffix}.png').is_file()
    summary = {
        'status': 'completed', 'created_at': timestamp(), 'default_run': source['runs'][0],
        'dictionary_sha256': source['dictionary_sha256'], 'dictionary_features': len(checks),
        'development_patients': source['patients'], 'image_patients_per_half': 64,
        'numeric_screen_features': len(numeric), 'content_starter_features': starters,
        'original_image_review_features': 32, 'qualified_image_review_features': len(numeric),
        'clinical_annotation': False, 'test_used': False,
        'scope': 'Conditional development-set qualification for feature research: nine broad image-content starting points in one fixed dictionary. No dictionary-wide semantic purity, unique-concept count, clinical identity or feature-specific causal effect is established.',
        'selection': 'Cross-seed and patient-balanced numerical screening in both halves; descriptive images from patient-disjoint halves. Development data previously selected dictionary configurations. Criteria are project screens, not universal SAE quality standards.',
        'independent_checks': independent,
        'source_sha256': sha256(Path(__file__))
    }
    write_json(output / 'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.6]})
    counts = [len(checks), int(checks.interior_screen_pass.sum()), len(numeric), len(starters)]
    axes[0].barh(['All dictionary features', 'Pooled interior screen', 'Patient-balanced screen', 'Content starting points'], counts, color=['#aab6c5', '#718aa7', '#386f9c', '#087f8c'])
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_xlim(1, 5000)
    axes[0].set_xlabel('Feature count (log scale)')
    for index, count in enumerate(counts):
        axes[0].text(count * 1.08, index, str(count), va='center')
    selected = catalog[catalog.use == 'content_starter'].sort_values('feature_id')
    axes[1].barh([str(value) for value in selected.feature_id], selected.patient_mean_minimum, color='#087f8c', label='Minimum mean across halves/modes/seeds')
    axes[1].scatter(selected.patient_confirmation_lower95_minimum, np.arange(len(selected)), color='#d55e00', marker='|', s=110, label='Minimum confirmation 95% lower bound')
    axes[1].axvline(.6, color='#697582', linestyle='--', linewidth=1)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Within-patient spatial correlation')
    axes[1].set_ylabel('Default dictionary feature ID')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower left', fontsize=7)
    fig.suptitle('FMCIB layer1: feature research material checks', fontsize=15)
    fig.text(.015, .015, '128 development patients; three SAE seeds. Image descriptions are single-reviewer observations. Counts do not represent distinct clinical concepts.', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94])
    fig.savefig(output / 'qualification.png', dpi=160)
    fig.savefig(output / 'qualification.svg')
    plt.close(fig)
    manifest = {**summary, 'dictionary': source['default_dictionary'], 'validation_root': str(root),
                'catalog': str(output / 'reviewed_features.csv'), 'all_feature_checks': str(output / 'all_feature_checks.csv'),
                'input': {'field_mm': 50, 'tensor_shape': [1, 50, 50, 50], 'native_grid': [13, 13, 13], 'interior_index_slice': [2, 11]},
                'inference': 'sclvmi.sae_predict.SpatialSAEPipeline',
                'qualified_atlas': str(root / 'atlas_qualified'), 'original_atlas': str(root / 'atlas'),
                'frozen_discovery_interpretations_sha256': sha256(root / 'qualified_discovery_interpretations.json'),
                'confirmation_review_sha256': sha256(root / 'qualified_confirmation_review.json'),
                'source_files': {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('feature_*.py'))}}
    write_json(root / 'research_materials.json', manifest)
    write_json(root / 'original_confirmation_review.json', {'features': original, 'reviewer': 'Codex descriptive image inspection', 'completed_at': timestamp()})
    print({'status': 'completed', 'content_starter_features': starters, 'manifest': str(root / 'research_materials.json')}, flush=True)


if __name__ == '__main__':
    main()
