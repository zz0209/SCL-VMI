import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.models import load_encoder, spatial_features
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_campaign import campaign_root
from selected_whole import configuration, map_geometry, output_root, selected_runs
from whole import input_affine, load_case, sample_input


@torch.inference_mode()
def check(model_name):
    audit = read_json(Path(context()[0]['runs']) / configuration()['source_run'] / 'source_audit.json')
    volume, affine = load_case(audit['cases'][0])
    model, _ = load_encoder(model_name)
    model.cpu().eval()
    runs = selected_runs(model_name)
    captures = {}
    handles = []
    requests = {run: read_json(campaign_root() / 'runs' / run / 'request.json') for run in runs}
    sites = {request['site']: request['site_specification'] for request in requests.values()}
    for site, value in sites.items():
        if site == 'global':
            continue
        def capture(module, inputs, result, key=site):
            captures[key] = result.detach()
        handles.append(model.get_submodule(value['module']).register_forward_hook(capture))
    tiles = map_geometry(model_name, 'global', volume.shape, affine)['tiles']
    tile = tiles // 2
    start = affine[:3, 3] + tile * configuration()['models'][model_name]['region_step_mm']
    inputs = torch.from_numpy(sample_input(volume, affine, model_name, start)[None])
    last = spatial_features(model, model_name, inputs)
    captures['global'] = last.mean((2, 3, 4), keepdim=True)
    rows = []
    for run in runs:
        request = requests[run]
        site = request['site']
        raw = captures[site][0]
        sae = SpatialDictionary(campaign_root() / 'runs' / run / 'dictionary.pt', device='cpu')
        encoded = sae.encode(raw.flatten(1).T).numpy().reshape(*raw.shape[1:], -1)
        normalized = (raw.flatten(1).T - sae.mean) / sae.scale
        unthresholded = torch.relu((normalized - sae.weights['b_dec']) @ sae.weights['W_enc']).numpy().reshape(*raw.shape[1:], -1)
        geometry = map_geometry(model_name, site, volume.shape, affine)
        if site == 'global':
            canonical = encoded
            canonical_unthresholded = unthresholded
        else:
            first, stop = configuration()['models'][model_name]['core'][site]
            native = encoded[first:stop, first:stop, first:stop]
            source_affine = input_affine(model_name, start) @ np.diag([request['site_specification']['stride']] * 3 + [1])
            shift = np.eye(4)
            shift[:3, 3] = first
            source_affine = source_affine @ shift
            image = nib.as_closest_canonical(nib.Nifti1Image(native, source_affine))
            canonical = np.asarray(image.dataobj)
            canonical_unthresholded = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(unthresholded[first:stop, first:stop, first:stop], source_affine)).dataobj)
            np.testing.assert_allclose(image.affine[:3, 3], geometry['origin'] + tile * geometry['core'] * geometry['spacing'], atol=1e-6)
            np.testing.assert_allclose(nib.affines.voxel_sizes(image.affine), geometry['spacing'], atol=1e-6)
        first = tile * geometry['core']
        slices = tuple(slice(int(value), int(value + geometry['core'])) for value in first)
        with h5py.File(output_root() / 'smoke' / run / 'case_00.h5', 'r') as saved:
            assert saved.attrs['status'] == 'smoke_completed'
            stored = saved['values'][(slice(None), *slices)].astype(np.float32)
        expected = canonical.transpose(3, 0, 1, 2)
        delta = np.abs(stored - expected)
        relative_l2 = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-8))
        support_changed = (stored > 0) != (expected > 0)
        prethreshold = canonical_unthresholded.transpose(3, 0, 1, 2)
        threshold_margin = float(np.max(np.abs(prethreshold[support_changed] - sae.threshold), initial=0))
        stored_margin = float(np.max(np.abs(stored[support_changed & (stored > 0)] - sae.threshold), initial=0))
        assert relative_l2 < .003, (run, relative_l2)
        assert threshold_margin < .002 and stored_margin < .002, (run, threshold_margin, stored_margin)
        assert float(np.max(delta[~support_changed], initial=0)) < .02 + float(expected.max()) * .003
        rows.append({'run_id': run, 'relative_l2': relative_l2, 'maximum_error': float(delta.max()), 'support_disagreements': int(support_changed.sum()), 'prethreshold_margin': threshold_margin, 'stored_threshold_margin': stored_margin, 'shape': list(stored.shape)})
        print(rows[-1], flush=True)
    write_json(output_root() / f'smoke_witness_{model_name}.json', {'state': 'passed', 'created_at': timestamp(), 'code_sha256': sha256(Path(__file__)), 'method': 'Independent CPU full-window SAE encoding, NIfTI canonical orientation and affine coordinates, compared with GPU tiled HDF5 output at the central smoke window. Support changes must remain within 0.002 of the saved inference threshold; overall relative L2 must stay below 0.003.', 'rows': rows, 'test_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fmcib', 'vista'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    check(args.model)
