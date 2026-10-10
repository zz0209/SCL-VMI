import argparse
import json
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from threading import RLock

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from scipy.ndimage import map_coordinates
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from sclvmi.context import context, read_json
from sclvmi.sae import SpatialDictionary
from whole_api import router as whole_router
from content_language import finding_translations, translated_fields
from campaign_api import router as campaign_router
from heatmap_api import router as heatmap_router


ROOT = Path(__file__).parent
storage, _, _ = context()
SOURCE = Path(storage['runs']) / '20261008_feature_validation/fmcib'
RUN = Path(storage['runs']) / '20261009_feature_exploration'
CROP_CACHE = RUN / 'crop_browser_cache'
CROP_LOCK = RLock()
members = pd.read_csv(RUN / 'members.csv', dtype={'PatientID': str})
descriptors = pd.read_csv(RUN / 'feature_descriptors.csv').set_index('feature_id')
rankings = pd.read_csv(RUN / 'rankings.csv')
intervention_records = pd.read_csv(RUN / 'intervention_cases.csv')
probe_summaries = pd.read_csv(RUN / 'perturbation_summary.csv')
screen = read_json(SOURCE / 'screen.json')
torch.set_num_threads(2)
dictionaries = {seed: SpatialDictionary(Path(storage['runs']) / '20261008_spatial_sae' / screen['runs'][i] / 'dictionary.pt', device='cpu') for i, seed in enumerate([2025, 2026, 2027])}
with np.load(SOURCE / 'discovery_spatial_statistics.npz') as saved:
    templates = saved['spatial_mean']

app = FastAPI(docs_url=None, redoc_url=None)
app.include_router(whole_router)
app.include_router(campaign_router)
app.include_router(heatmap_router)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
app.mount('/shared', StaticFiles(directory=ROOT.parent / 'shared'), name='shared')


@app.middleware('http')
async def private_response(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin and origin != 'http://' + request.headers['host']:
        return JSONResponse({'detail': 'Local origin required'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


def case_row(case):
    if not 0 <= case < len(members):
        raise HTTPException(404, 'Case outside the exploration sample')
    return members.iloc[case]


def feature_row(feature):
    if feature not in descriptors.index:
        raise HTTPException(404, 'Unknown feature')
    return descriptors.loc[feature]


@lru_cache(maxsize=4)
def geometry(case):
    row = case_row(case)
    metadata = read_json(SOURCE / 'cases' / f'{row.identifier}.json')
    affine = np.asarray(metadata['input_affine_ras'])
    orientation = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(orientation, nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(transform, (50, 50, 50))
    return affine, transform, canonical_affine


def canonical(array, case):
    return nib.orientations.apply_orientation(array, geometry(case)[1]).copy()


@lru_cache(maxsize=4)
def raw_case(case):
    row = case_row(case)
    with np.load(SOURCE / 'cases' / f'{row.identifier}.npz') as saved:
        return saved['input'][0], saved['activations']


def crop_file(case):
    case_row(case)
    path = CROP_CACHE / f'case_{case:03d}.h5'
    if not path.exists():
        raise HTTPException(409, 'Crop browser cache is unavailable. Run prepare_crop_cache.py.')
    return path


@lru_cache(maxsize=32)
def crop_input(case):
    with h5py.File(crop_file(case), 'r') as saved:
        return saved['input'][0]


@lru_cache(maxsize=256)
def crop_native(feature, case):
    feature_row(feature)
    with h5py.File(crop_file(case), 'r') as saved:
        return saved['activations'][feature]


@lru_cache(maxsize=16)
def native_features(case):
    with h5py.File(crop_file(case), 'r') as saved:
        return torch.from_numpy(saved['layer1'][...]).flatten(1).T


@torch.inference_mode()
def counterpart_map(feature, case, seed):
    target = int(feature_row(feature)[f'patient_raw_counterpart_s{seed}'])
    dictionary = dictionaries[seed]
    tokens = native_features(case)
    normalized = (tokens-dictionary.mean)/dictionary.scale-dictionary.weights['b_dec']
    values = torch.relu(normalized @ dictionary.weights['W_enc'][:, target])
    values *= values > dictionary.threshold
    return target, values.reshape(13, 13, 13).numpy()


@lru_cache(maxsize=24)
def feature_maps(feature, case):
    feature_row(feature)
    _, activations = raw_case(case)
    raw = activations[feature]
    counterparts = {seed: counterpart_map(feature, case, seed) for seed in [2026, 2027]}
    values = {'raw': raw, 'template': templates[feature], 'residual': raw-templates[feature],
              'seed2026': counterparts[2026][1], 'seed2027': counterparts[2027][1]}
    coordinates = np.indices((50, 50, 50), dtype=float)/4
    maps = {}
    for name, value in values.items():
        dense = canonical(map_coordinates(value, coordinates, order=1, mode='nearest'), case)
        maps[name] = np.round(dense, 5).ravel().tolist()
    native = {name: value.ravel().tolist() for name, value in values.items()}
    native_to_canonical = np.linalg.inv(geometry(case)[2]) @ geometry(case)[0] @ np.diag([4, 4, 4, 1])
    return {'maps': maps, 'native': native, 'native_to_canonical': native_to_canonical.tolist(),
            'canonical_to_native': np.linalg.inv(native_to_canonical).tolist(),
            'counterparts': {str(seed): value[0] for seed, value in counterparts.items()}}


@app.get('/')
def index(request: Request):
    return RedirectResponse('/experiments?' + request.url.query if request.url.query else '/whole')


@app.get('/experiments')
def experiments():
    return FileResponse(ROOT / 'static/index.html')


@app.get('/api/catalog')
def catalog():
    features = json.loads(descriptors.reset_index().to_json(orient='records'))
    descriptions = read_json(RUN / 'interpretations.json')
    findings = read_json(RUN / 'findings.json')
    translations = finding_translations(findings, read_json(RUN / 'findings.en.json'), ['items', 'sources'])
    for feature, english in read_json(RUN / 'interpretations.en.json').items():
        translations.extend(translated_fields(descriptions[feature], english))
    return {'features': features, 'interpretations': descriptions, 'patients': len(members), 'analysis': read_json(RUN / 'analysis.json'),
            'findings': findings, 'translations': translations, 'dictionary_sha256': screen['dictionary_sha256'],
            'probes': json.loads(probe_summaries[probe_summaries.feature_split == 'confirmation'].to_json(orient='records')),
            'scales': rankings[rankings.feature_split == 'discovery'].groupby('feature_id').maximum.quantile(.95).to_dict()}


@app.get('/api/cases/{feature}')
def cases(feature: int, split: str = 'confirmation', order: str = 'maximum'):
    feature_row(feature)
    if split not in ['confirmation', 'discovery', 'all'] or order not in ['maximum', 'mean', 'full_maximum', 'full_mean', 'effect']:
        raise HTTPException(422, 'Unknown ranking option')
    selected = rankings[rankings.feature_id == feature]
    if split != 'all':
        selected = selected[selected.feature_split == split]
    effects = intervention_records[intervention_records.feature_id == feature].copy()
    effects['effect'] = (effects.ablate_100-effects.original).abs()
    selected = selected.merge(effects[['identifier', 'effect']], on='identifier', how='left')
    if order == 'effect' and selected.effect.isna().all():
        raise HTTPException(422, 'This feature has no intervention experiment')
    selected = selected.sort_values([order, 'case_index'], ascending=[False, True]).copy()
    selected['rank'] = np.arange(len(selected)) + 1
    selected['label'] = selected.case_index.map(lambda i: f'C{i+1:03d}')
    return json.loads(selected.drop(columns=['identifier']).to_json(orient='records'))


@app.get('/api/volume/{case}')
def volume(case: int):
    row = case_row(case)
    data, _ = raw_case(case)
    hu = canonical(data*3072-1024, case)
    affine = geometry(case)[2]
    mask = np.zeros((50, 50, 50), dtype=np.uint8)
    mask[8:41, 8:41, 8:41] = 1
    valid = np.argwhere(canonical(mask, case))
    return {'case_index': case, 'label': f'C{case+1:03d}', 'split': row.feature_split, 'shape': list(hu.shape),
            'volume': np.rint(hu).astype(int).ravel().tolist(), 'affine': affine.tolist(),
            'spacing': nib.affines.voxel_sizes(affine).tolist(),
            'interior_bounds': [valid.min(0).tolist(), valid.max(0).tolist()], 'axes': ['R', 'A', 'S']}


@app.get('/api/crop/metadata/{feature}/{case}')
@lru_cache(maxsize=256)
def crop_metadata(feature: int, case: int):
    row = case_row(case)
    feature_row(feature)
    affine, _, canonical_affine = geometry(case)
    mask = np.zeros((50, 50, 50), dtype=np.uint8)
    mask[8:41, 8:41, 8:41] = 1
    valid = np.argwhere(canonical(mask, case))
    native_transform = np.linalg.inv(canonical_affine) @ affine @ np.diag([4,4,4,1])
    return {'case_index': case, 'label': f'C{case+1:03d}', 'split': row.feature_split,
            'shape': [50,50,50], 'spacing': nib.affines.voxel_sizes(canonical_affine).tolist(),
            'affine': canonical_affine.tolist(), 'axes': ['R','A','S'],
            'interior_bounds': [valid.min(0).tolist(), valid.max(0).tolist()],
            'native_to_canonical': native_transform.tolist(), 'canonical_to_native': np.linalg.inv(native_transform).tolist(),
            'counterparts': {str(seed): int(feature_row(feature)[f'patient_raw_counterpart_s{seed}']) for seed in (2026,2027)}}


@app.get('/api/crop/volume/{case}')
def crop_volume(case: int):
    hu = np.rint(canonical(crop_input(case)*3072-1024, case)).astype('<f4')
    return Response(hu.tobytes(), media_type='application/octet-stream')


@lru_cache(maxsize=128)
def crop_map_arrays(feature, case, mode):
    feature_row(feature)
    probe_volume = None
    if mode == 'raw':
        native = crop_native(feature, case)
    elif mode == 'template':
        native = templates[feature]
    elif mode == 'residual':
        native = crop_native(feature, case)-templates[feature]
    elif mode in ('seed2026', 'seed2027'):
        native = counterpart_map(feature, case, int(mode[4:]))[1]
    elif mode in ('blur_1mm', 'hu_plus100', 'hu_minus100'):
        with h5py.File(crop_file(case), 'r') as saved:
            matches = np.flatnonzero(saved['feature_ids'][...] == feature)
            if len(matches) != 1:
                raise HTTPException(404, 'Input maps are available for the 13 reviewed features')
            native = saved[f'{mode}_activations'][matches[0]]
            probe_volume = np.rint(canonical(saved[f'{mode}_input'][...]*3072-1024, case))
    else:
        raise HTTPException(422, 'Unknown response source')
    coordinates = np.indices((50,50,50), dtype=float)/4
    dense = np.round(canonical(map_coordinates(native, coordinates, order=1, mode='nearest'), case), 5)
    arrays = [dense.ravel(), native.ravel()]
    if probe_volume is not None:
        arrays.append(probe_volume.ravel())
    return np.concatenate(arrays).astype('<f4').tobytes()


@app.get('/api/crop/map/{feature}/{case}/{mode}')
def crop_map(feature: int, case: int, mode: str):
    with CROP_LOCK:
        data = crop_map_arrays(feature, case, mode)
    return Response(data, media_type='application/octet-stream')


@app.get('/api/maps/{feature}/{case}')
def maps(feature: int, case: int):
    return feature_maps(feature, case)


@app.get('/api/thumbnail/{feature}/{case}')
@lru_cache(maxsize=512)
def thumbnail(feature: int, case: int):
    feature_row(feature)
    data = crop_input(case)
    native = crop_native(feature, case)[2:-2, 2:-2, 2:-2]
    peak = np.asarray(np.unravel_index(native.argmax(), native.shape)) + 2
    original_position = peak*4 if native.max() > 0 else np.array([25, 25, 25])
    transform = np.linalg.inv(geometry(case)[2]) @ geometry(case)[0]
    position = np.rint(nib.affines.apply_affine(transform, original_position)).astype(int)
    hu = canonical(data*3072-1024, case)
    plane = hu[:, :, position[2]].T[::-1]
    pixels = np.uint8(np.clip((plane+1000)/1400, 0, 1)*255)
    image = Image.fromarray(pixels).resize((100, 100), Image.Resampling.NEAREST)
    output = BytesIO()
    image.save(output, format='PNG')
    return Response(output.getvalue(), media_type='image/png')


@app.get('/api/probe/{feature}/{case}/{mode}')
def probe_map(feature: int, case: int, mode: str):
    feature_row(feature)
    row = case_row(case)
    if mode not in ['blur_1mm', 'hu_plus100', 'hu_minus100']:
        raise HTTPException(422, 'Unknown input probe')
    path = RUN / 'probe_maps' / f'{row.identifier}.npz'
    with np.load(path) as saved:
        matches = np.flatnonzero(saved['feature_ids'] == feature)
        if len(matches) != 1:
            raise HTTPException(404, 'Probe maps are available for the 13 reviewed features')
        native = saved[f'{mode}_activations'][matches[0]]
        volume = canonical(saved[f'{mode}_input']*3072-1024, case)
    coordinates = np.indices((50,50,50), dtype=float)/4
    dense = canonical(map_coordinates(native, coordinates, order=1, mode='nearest'), case)
    return {'map':np.round(dense,5).ravel().tolist(), 'native':native.ravel().tolist(),
            'volume':np.rint(volume).astype(int).ravel().tolist(), 'mode':mode}


@app.get('/api/interventions/{feature}')
def interventions(feature: int, case: int = Query(0, ge=0)):
    feature_row(feature)
    path = RUN / 'intervention_cases.csv'
    if not path.exists():
        return {'status': 'running', 'progress': read_json(RUN / 'intervention_progress.json')}
    records = pd.read_csv(path)
    rows = records[records.feature_id == feature]
    if rows.empty:
        return {'status': 'not_studied'}
    identifier = case_row(case).identifier
    selected = rows[rows.identifier == identifier]
    summaries = pd.read_csv(RUN / 'intervention_summary.csv')
    return {'status': 'completed', 'case': json.loads(selected.drop(columns=['identifier']).to_json(orient='records')),
            'summary': json.loads(summaries[summaries.feature_id == feature].to_json(orient='records')),
            'all_cases': json.loads(rows.drop(columns=['identifier']).to_json(orient='records'))}


@app.get('/api/health')
def health():
    return {'status': 'ok', 'patients': len(members), 'features': len(descriptors), 'run': RUN.name}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
