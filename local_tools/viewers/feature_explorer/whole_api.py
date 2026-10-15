from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image

from sclvmi.context import context, read_json


router = APIRouter()
ROOT = Path(context()[0]['runs']) / '20261009_whole_ct_exploration'
VOLUME_LOCK = Lock()


def cases():
    rows = read_json(ROOT / 'source_audit.json')['cases']
    return [{'id': index, 'label': f'CT {index+1:02d}', 'split': 'discovery' if index % 2 == 0 else 'confirmation'} for index in range(len(rows))]


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


@router.get('/whole')
def whole_page():
    return FileResponse(Path(__file__).parent / 'static/whole.html')


def ct_volume(case):
    volume, _ = raw_volume(case)
    return Response(np.rint(volume).astype('<f4').tobytes(order='C'), media_type='application/octet-stream')


@lru_cache(maxsize=8)
def thumbnail(case):
    volume, affine = raw_volume(case)
    image = np.take(volume, volume.shape[1] // 2, axis=1)
    grey = np.clip((image+160)/400, 0, 1)
    size = (160, round(160 * image.shape[1] * affine[2, 2] / (image.shape[0] * affine[0, 0])))
    preview = Image.fromarray(np.uint8(grey.T[::-1]*255)).resize(size)
    buffer = BytesIO()
    preview.save(buffer, format='PNG')
    return Response(buffer.getvalue(), media_type='image/png')
