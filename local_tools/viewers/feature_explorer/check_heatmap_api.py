import argparse
import json
from pathlib import Path
import time

import numpy as np
import requests

from sclvmi.context import context, timestamp, write_json


def get(path, **params):
    response = requests.get('http://127.0.0.1:8773' + path, params=params, timeout=180)
    response.raise_for_status()
    return response


def check(require_whole):
    rows = get('/api/heatmaps/catalog').json()['dictionaries']
    assert len(rows) == 20
    selected = [row for row in rows if not row['reference']]
    assert len(selected) == 18 and len({row['family'] for row in selected}) == 6
    checks = []
    for index, row in enumerate(rows):
        for scope in row['scopes']:
            catalog = get('/api/heatmaps/features', run=row['run_id'], scope=scope).json()
            assert [item['id'] for item in catalog['features']] == list(range(row['features']))
            assert len(catalog['cases']) == (128 if scope == 'crop' else 8)
            if not all(item['ready'] for item in catalog['cases']):
                assert scope == 'whole' and not require_whole
                checks.append({'run_id': row['run_id'], 'scope': scope, 'state': 'preparing'})
                continue
            mode = 'global' if row['site'] == 'global' else 'spatial'
            params = dict(run=row['run_id'], scope=scope, case=0, feature=0, mode=mode)
            started = time.monotonic()
            meta = get('/api/heatmaps/map', **params).json()
            activity = np.frombuffer(get('/api/heatmaps/activity', **params).content, '<f4')
            volume = np.frombuffer(get(f'/api/heatmaps/ct/{scope}/{row["model"]}/0').content, '<f4')
            assert len(activity) == np.prod(meta['map_shape']) and len(volume) == np.prod(meta['shape'])
            assert np.isfinite(activity).all() and (activity >= 0).all() and np.isfinite(volume).all()
            assert meta['scale'] > 0 and len(meta['spacing']) == 3
            if scope == 'crop':
                baseline = np.frombuffer(get(f'/api/dictionaries/{row["run_id"]}/0/0/map').content, '<f4')
                np.testing.assert_array_equal(activity, baseline)
                if row['site'] != 'global':
                    nearest = np.frombuffer(get('/api/heatmaps/activity', **{**params, 'mode': 'native'}).content, '<f4')
                    assert len(nearest) == len(volume) and np.isfinite(nearest).all()
            elif row['reference']:
                baseline = np.frombuffer(get('/api/whole/activity-volume', model=row['model'], case=0, feature=0, mode=mode).content, '<f4')
                np.testing.assert_array_equal(activity, baseline)
            else:
                if row['site'] != 'global':
                    for pooled_mode in ['single', 'mean', 'maximum']:
                        pooled_meta = get('/api/heatmaps/map', **{**params, 'mode': pooled_mode}).json()
                        pooled = np.frombuffer(get('/api/heatmaps/activity', **{**params, 'mode': pooled_mode}).content, '<f4')
                        assert len(pooled) == np.prod(pooled_meta['map_shape']) and np.isfinite(pooled).all()
            checks.append({'run_id': row['run_id'], 'scope': scope, 'state': 'passed', 'seconds': time.monotonic() - started, 'shape': meta['shape'], 'map_shape': meta['map_shape']})
        print(f'{index + 1}/{len(rows)} dictionaries checked: {row["model"]} {row["site"]} seed {row["seed"]}', flush=True)
    row = selected[0]
    params = dict(run=row['run_id'], scope='crop', case=0, feature=0, mode='spatial')
    for changes in [{'feature': -1}, {'feature': row['features']}, {'case': -1}, {'case': 128}, {'scope': 'unknown'}, {'mode': 'unknown'}]:
        response = requests.get('http://127.0.0.1:8773/api/heatmaps/map', params={**params, **changes}, timeout=180)
        assert response.status_code == 422, (changes, response.status_code)
    assert get('/').url.endswith('/whole')
    assert '/experiments?' in get('/', feature=308).url
    assert '/whole?scope=crop&' in get('/dictionaries', run=row['run_id'], feature=308).url
    output = Path(context()[0]['runs']) / '20261010_selected_sae_whole_ct' / 'verification'
    stamp = timestamp().replace(':', '').replace('+', '_')
    result = {'created_at': timestamp(), 'require_whole': require_whole, 'checks': checks, 'invalid_input_checks': 6, 'route_checks': 3, 'test_used': False}
    write_json(output / f'api_{stamp}.json', result)
    print(json.dumps({'state': 'passed', 'checks': len(checks), 'whole_complete': require_whole}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-whole', action='store_true')
    args = parser.parse_args()
    check(args.require_whole)
