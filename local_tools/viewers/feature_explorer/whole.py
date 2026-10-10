import os
os.environ['OMP_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
os.environ['MKL_NUM_THREADS'] = '3'
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform

from sclvmi.context import context, read_json
from sclvmi.models import load_encoder
from sclvmi.sae import SpatialDictionary


def run_root():
    return Path(context()[0]['runs']) / '20261009_whole_ct_exploration'


def specification(name):
    return {'fmcib': {'size': 50, 'spacing': 1., 'step': 4., 'features': 2048, 'site': 'layer1'},
            'vista': {'size': 48, 'spacing': 1.5, 'step': 6., 'features': 768, 'site': 'stage2'}}[name]


def load_case(row):
    image = nib.as_closest_canonical(nib.load(row['image']))
    assert nib.aff2axcodes(image.affine) == ('R', 'A', 'S')
    assert np.allclose(image.affine[:3, :3], np.diag(np.diag(image.affine[:3, :3])))
    return image.get_fdata(dtype=np.float32), image.affine


def input_affine(name, start):
    matrix = np.eye(4)
    if name == 'fmcib':
        matrix[:3, :3] = [[0, 0, -1], [0, -1, 0], [1, 0, 0]]
        matrix[:3, 3] = start + np.asarray([36, 36, -8])
    else:
        matrix[:3, :3] *= 1.5
        matrix[:3, 3] = start - 12
    return matrix


def sample_input(volume, affine, name, start):
    transform = np.linalg.inv(affine) @ input_affine(name, start)
    size = specification(name)['size']
    low, high = (-1024., 2048.) if name == 'fmcib' else (-963.8247715525971, 1053.678477684517)
    crop = affine_transform(volume, transform[:3, :3], offset=transform[:3, 3],
                            output_shape=(size,) * 3, order=1, mode='constant', cval=low, prefilter=False)
    crop = (crop - low) / (high - low)
    if name == 'vista':
        np.clip(crop, 0, 1, out=crop)
    return crop[None]


def grid_geometry(volume, affine, name):
    step = specification(name)['step']
    extent = (np.asarray(volume.shape) - 1) * np.diag(affine)[:3]
    shape = np.floor(extent / step).astype(int) + 1
    tiles = (shape + 7) // 8
    return shape, tiles, affine[:3, 3].copy()


def starts_from_indices(indices, tiles, origin, name):
    return np.asarray(np.unravel_index(indices, tiles)).T * (8 * specification(name)['step']) + origin


class WholeEncoder:
    def __init__(self, name):
        self.name = name
        self.model, self.assets = load_encoder(name)
        self.model.cuda()
        index = read_json(Path(context()[0]['runs']) / '20261008_spatial_sae/dictionary_index.json')
        self.identity = next(row for row in index['dictionaries'] if row['model'] == name and row['default'])
        self.dictionary = SpatialDictionary(self.identity['dictionary'])

    @torch.inference_mode()
    def prefix(self, batch):
        if self.name == 'fmcib':
            model = self.model
            return model.layer1(model.maxpool(model.act(model.bn1(model.conv1(batch)))))
        encoder = self.model.image_encoder.encoder
        values = encoder.conv_init(batch)
        for level in encoder.layers[:2]:
            values = level['downsample'](level['blocks'](values))
        return encoder.layers[2]['blocks'](values)

    @torch.inference_mode()
    def extract(self, inputs):
        values = self.prefix(torch.as_tensor(np.stack(inputs), device='cuda'))
        core = values[:, :, 2:10, 2:10, 2:10]
        if self.name == 'fmcib':
            core = core.permute(0, 1, 4, 3, 2).flip((2, 3))
        tokens = core.permute(0, 2, 3, 4, 1).reshape(-1, core.shape[1])
        activity = self.dictionary.encode(tokens)
        recovered = self.dictionary.decode(activity)
        stats = torch.stack([(tokens - recovered).square().sum(1),
                             (tokens - self.dictionary.mean).square().sum(1),
                             (activity > 0).sum(1).float()], 1)
        activity = activity.reshape(len(inputs), 8, 8, 8, -1).permute(0, 4, 1, 2, 3)
        return activity.cpu().numpy(), stats.reshape(len(inputs), 8, 8, 8, 3).cpu().numpy()


def assemble(blocks, tiles, shape):
    return blocks.reshape(*tiles, 8, 8, 8).transpose(0, 3, 1, 4, 2, 5).reshape(*(tiles * 8))[tuple(slice(0, n) for n in shape)]
