from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock, RLock

import h5py
import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from filelock import FileLock
from PIL import Image

from sclvmi.context import context, read_json
from content_language import finding_translations


router = APIRouter()
ROOT = Path(context()[0]['runs']) / '20261009_whole_ct_exploration'
CACHE = ROOT / 'viewer_cache'
STAGES = {'fmcib': 'continuous_fmcib', 'vista': 'vista_windows'}
ANALYSIS = ROOT / 'complete_analysis'
READ_LOCK = RLock()
VOLUME_LOCK = Lock()


def specification(model):
    return {'fmcib': {'features': 2048, 'step': 4}, 'vista': {'features': 768, 'step': 6}}[model]


def case_file(model, case):
    if model not in ['fmcib', 'vista'] or not 0 <= case < 8:
        raise HTTPException(404, 'Unknown model or case')
    path = ROOT / STAGES[model] / model / f'case_{case:02d}.h5'
    if not path.with_suffix('.json').exists():
        raise HTTPException(409, '此病例的完整提取尚未完成。请查看计算进度。')
    return path


def fast_case_file(model, case):
    return case_file(model, case)


def raw_volume(case):
    with VOLUME_LOCK:
        return cached_raw_volume(case)


@lru_cache(maxsize=8)
def cached_raw_volume(case):
    rows = read_json(ROOT / 'source_audit.json')['cases']
    if not 0 <= case < len(rows):
        raise HTTPException(404, 'Unknown case')
    image = nib.as_closest_canonical(nib.load(rows[case]['image']))
    assert nib.aff2axcodes(image.affine) == ('R', 'A', 'S')
    return image.get_fdata(dtype=np.float32), image.affine


def activity_map(model, case, feature, mode):
    with READ_LOCK:
        return cached_activity_map(model, case, feature, mode)


@lru_cache(maxsize=12)
def cached_activity_map(model, case, feature, mode):
    if not 0 <= feature < specification(model)['features']:
        raise HTTPException(404, 'Feature 编号超出当前模型的字典范围。')
    source = case_file(model, case)
    if mode not in ['spatial', 'mean', 'maximum', 'single']:
        raise HTTPException(422, 'Unknown map mode')
    if mode == 'spatial':
        return read_activity(source, feature, mode)
    directory = CACHE / STAGES[model] / model
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'case_{case:02d}_feature_{feature}_{mode}.npz'
    with FileLock(str(target) + '.lock', timeout=300):
        if target.exists():
            with np.load(target) as cached:
                return cached['values'], cached['origin'], float(cached['step'])
        if mode != 'spatial':
            spatial, origin, step = activity_map(model, case, feature, 'spatial')
            shape = np.asarray(spatial.shape)
            tiles = (shape+7)//8
            lengths = [np.minimum(8, size-np.arange(count)*8) for size, count in zip(shape, tiles)]
            if mode == 'single':
                centers = [np.arange(count)*8+length//2 for count, length in zip(tiles, lengths)]
                values = spatial[np.ix_(*centers)]
            else:
                padded = np.pad(spatial.astype(np.float32), [(0, int(tile*8-size)) for tile, size in zip(tiles, shape)])
                blocks = padded.reshape(tiles[0], 8, tiles[1], 8, tiles[2], 8)
                if mode == 'maximum':
                    values = blocks.max((1, 3, 5))
                else:
                    counts = lengths[0][:, None, None]*lengths[1][None, :, None]*lengths[2][None, None, :]
                    values = blocks.sum((1, 3, 5))/counts
            result = values, origin, step*8
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, values=result[0], origin=result[1], step=result[2])
        temporary.replace(target)
        return result


def read_activity(source, feature, mode):
    with READ_LOCK:
        target = CACHE / source.parent.parent.name / source.parent.name / f'{source.stem}_browser.h5'
        if not target.exists():
            raise HTTPException(409, '当前扫描的浏览缓存尚未生成，请完成缓存准备后重新选择。')
        with h5py.File(target, 'r') as saved:
            assert saved.attrs['status'] == 'completed'
            assert mode == 'spatial'
            return saved['values'][feature], np.asarray(saved.attrs['origin']), float(saved.attrs['step_mm'])


def activity_from_handle(saved, feature, mode):
    assert saved.attrs['status'] == 'completed'
    tiles, shape = np.asarray(saved.attrs['tiles']), np.asarray(saved.attrs['shape'])
    core = int(saved.attrs.get('core_size', 8))
    pool_tiles = np.asarray(saved.attrs.get('pool_tiles', tiles))
    if mode == 'spatial':
        blocks = saved['activations'][:, feature]
        values = blocks.reshape(*tiles, core, core, core).transpose(0, 3, 1, 4, 2, 5).reshape(*(tiles*core))[tuple(slice(0, int(n)) for n in shape)]
    elif mode in ['mean', 'maximum']:
        values = saved[mode][:, feature].reshape(pool_tiles)
    elif mode == 'single':
        values = saved['single'][:, feature].reshape(pool_tiles) if 'single' in saved else saved['activations'][:, feature, 4, 4, 4].reshape(tiles)
    else:
        raise HTTPException(422, 'Unknown map mode')
    return values, np.asarray(saved.attrs['origin']), float(saved.attrs['step_mm']) * (1 if mode == 'spatial' else 8)


@lru_cache(maxsize=8)
def display_catalog(path):
    return read_json(path)


def feature_scale(model, feature, mode, values):
    selected_path = ANALYSIS / 'view_scales.json'
    if selected_path.exists():
        selected = display_catalog(selected_path).get(model, {}).get(str(feature))
        if selected:
            return max(float(selected['mean' if mode=='mean' else 'spatial']), .001), '固定 discovery 颜色上限'
    catalog_path = ANALYSIS / f'{model}_catalog.json'
    if catalog_path.exists():
        catalog = display_catalog(catalog_path)
        key = 'mean_scale' if mode == 'mean' else 'spatial_scale'
        return max(float(catalog['features'][feature][key]), .001), '固定 discovery 颜色上限'
    raise HTTPException(409, '缺少固定 discovery 颜色尺度，请完成分析数据准备。')


@router.get('/whole')
def whole_page():
    return FileResponse(Path(__file__).parent / 'static/whole.html')


@router.get('/api/whole/status')
def status():
    audit = read_json(ROOT / 'source_audit.json')
    cases = []
    for index, row in enumerate(audit['cases']):
        cases.append({'id': index, 'label': f'CT {index+1:02d}',
                      'split': 'discovery' if index % 2 == 0 else 'confirmation',
                      'cohort': row['cohort'], 'source_spacing_mm': row['source_spacing_mm'],
                      'ready': {name: (ROOT / STAGES[name] / name / f'case_{index:02d}.json').exists() for name in ['fmcib', 'vista']}})
    paths = [ROOT / stage / 'progress.json' for stage in STAGES.values()]
    present = [path for path in paths if path.exists()]
    path = max(present, key=lambda item: item.stat().st_mtime) if present else None
    return {'cases': cases, 'progress': read_json(path) if path else None,
            'models': {name: specification(name) for name in ['fmcib', 'vista']}}


@router.get('/api/whole/catalog/{model}')
def catalog(model: str):
    if model not in ['fmcib', 'vista']:
        raise HTTPException(404, 'Unknown model')
    path = ANALYSIS / f'{model}_catalog.json'
    if path.exists():
        result = read_json(path)
        selection = ANALYSIS / 'view_selection.json'
        if selection.exists():
            result['preferred_feature'] = read_json(selection)['models'][model]['rules'][0]['feature']
        return result
    raise HTTPException(409, '当前模型的 Feature 目录尚未生成。')


@router.get('/api/whole/map')
@lru_cache(maxsize=128)
def map_metadata(model: str = 'fmcib', case: int = 0, feature: int = 1976, mode: str = 'spatial'):
    case_file(model, case)
    volume, affine = raw_volume(case)
    values, origin, step = activity_map(model, case, feature, mode)
    maximum = np.unravel_index(np.argmax(values), values.shape)
    world = origin + np.asarray(maximum) * step
    if mode != 'spatial':
        native_step = specification(model)['step']
        native_shape = np.floor((np.asarray(volume.shape)-1)*np.diag(affine)[:3]/native_step).astype(int)+1
        lengths = np.minimum(8, native_shape-np.asarray(maximum)*8)
        world += (lengths//2)*native_step
    peak = np.rint((world - affine[:3, 3]) / np.diag(affine)[:3]).astype(int)
    scale, scale_source = feature_scale(model, feature, mode, values)
    return {'shape': list(volume.shape), 'spacing': np.diag(affine)[:3].tolist(), 'origin': affine[:3, 3].tolist(),
            'map_shape': list(values.shape), 'map_origin': origin.tolist(), 'map_spacing_mm': step, 'scale': scale, 'scale_source': scale_source,
            'peak': np.clip(peak, 0, np.asarray(volume.shape)-1).tolist(), 'maximum': float(values.max()),
            'mean': float(values.mean(dtype=np.float64)), 'nonzero_fraction': float(np.count_nonzero(values)/values.size)}


@router.get('/api/whole/ct-volume/{case}')
def ct_volume(case: int):
    volume, _ = raw_volume(case)
    return Response(np.rint(volume).astype('<f4').tobytes(order='C'), media_type='application/octet-stream')


@router.get('/api/whole/activity-volume')
def activity_volume(model: str = 'fmcib', case: int = 0, feature: int = 1976, mode: str = 'spatial'):
    case_file(model, case)
    values, _, _ = activity_map(model, case, feature, mode)
    return Response(values.astype('<f4').tobytes(order='C'), media_type='application/octet-stream')


@router.get('/api/whole/slice')
def slice_values(model: str = 'fmcib', case: int = 0, feature: int = 1976, mode: str = 'spatial',
                 axis: int = Query(2, ge=0, le=2), position: int = Query(0, ge=0)):
    case_file(model, case)
    volume, affine = raw_volume(case)
    if position >= volume.shape[axis]:
        raise HTTPException(422, 'Slice outside CT')
    values, origin, step = activity_map(model, case, feature, mode)
    axes = [item for item in range(3) if item != axis]
    coordinates = np.meshgrid(*[np.arange(volume.shape[item]) for item in axes], indexing='ij')
    points = np.empty((3, coordinates[0].size), dtype=np.float64)
    points[axis] = position
    for item, coordinate in zip(axes, coordinates):
        points[item] = coordinate.ravel()
    world = np.diag(affine)[:3, None] * points + affine[:3, 3, None]
    sample = (world - origin[:, None]) / step
    sample = np.rint(sample) if mode == 'spatial' else np.floor(sample)
    sample = np.clip(sample.astype(int), 0, np.asarray(values.shape)[:, None]-1)
    heat = values[tuple(sample)].reshape(coordinates[0].shape)
    ct = np.take(volume, position, axis=axis)
    return {'shape': list(ct.shape), 'axes': axes, 'ct': np.rint(ct).astype(int).ravel().tolist(),
            'activity': heat.astype(np.float32).ravel().tolist()}


@router.get('/api/whole/thumbnail/{case}')
@lru_cache(maxsize=8)
def thumbnail(case: int):
    volume, affine = raw_volume(case)
    image = np.take(volume, volume.shape[1] // 2, axis=1)
    grey = np.clip((image+160)/400, 0, 1)
    size = (160, round(160 * image.shape[1] * affine[2, 2] / (image.shape[0] * affine[0, 0])))
    preview = Image.fromarray(np.uint8(grey.T[::-1]*255)).resize(size)
    buffer = BytesIO()
    preview.save(buffer, format='PNG')
    return Response(buffer.getvalue(), media_type='image/png')


@router.get('/api/whole/findings')
def findings():
    path = ANALYSIS / 'findings.json'
    if not path.exists():
        return {'status': 'computing', 'findings': [], 'translations': []}
    value = read_json(path)
    value['translations'] = finding_translations(value, read_json(ANALYSIS / 'findings.en.json'), ['findings'])
    return value


@router.get('/api/whole/figure/{name}')
def figure(name: str):
    if Path(name).name != name or Path(name).suffix != '.png':
        raise HTTPException(404, 'Unknown figure')
    path = ANALYSIS / 'figures' / name
    if not path.is_file():
        raise HTTPException(404, 'Unknown figure')
    return FileResponse(path)
