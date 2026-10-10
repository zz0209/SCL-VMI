import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae_campaign import campaign_root, ensure_identity, pause_requested
from selected_whole import SelectedEncoder, configuration, initialize, map_geometry, output_root
from whole import load_case, sample_input


_outputs = {}


def open_outputs(paths):
    torch.set_num_threads(1)
    for run, path in paths.items():
        _outputs[run] = h5py.File(path, 'r+')
    atexit.register(close_outputs)


def close_outputs():
    for saved in _outputs.values():
        saved.close()
    _outputs.clear()


def write_batch(blocks, coordinates, offset):
    stop = offset + len(coordinates)
    for run, compressed in blocks.items():
        saved = _outputs[run]
        if saved.attrs['completed_tiles'] >= stop:
            continue
        assert saved.attrs['completed_tiles'] == offset
        core = int(saved.attrs['core_size'])
        shape = np.asarray(saved.attrs['shape'])
        maximum = saved['maximum'][...]
        for index, tile in enumerate(coordinates):
            first = tile * core
            lengths = np.minimum(core, shape - first)
            if np.any(lengths <= 0):
                continue
            source = tuple(slice(0, int(length)) for length in lengths)
            target = tuple(slice(int(begin), int(begin + length)) for begin, length in zip(first, lengths))
            block = compressed[(index, slice(None), *source)]
            saved['values'][(slice(None), *target)] = block
            maximum = np.maximum(maximum, block.max((1, 2, 3)).astype(np.float32))
        saved['maximum'][...] = maximum
        saved.flush()
        saved.attrs['completed_tiles'] = stop
        saved.flush()
    return stop


class FastEncoder(SelectedEncoder):
    @torch.inference_mode()
    def encode(self, run, values):
        sae = self.dictionaries[run]
        batch, _, x, y, z = values.shape
        tokens = values.permute(0, 2, 3, 4, 1).reshape(-1, values.shape[1])
        encoded = sae.encode(tokens).reshape(batch, x, y, z, -1).permute(0, 4, 1, 2, 3)
        if self.name == 'fmcib' and sae.request['site'] != 'global':
            encoded = encoded.permute(0, 1, 4, 3, 2).flip((2, 3))
        encoded = encoded.to(torch.float16).contiguous()
        assert torch.isfinite(encoded).all() and (encoded >= 0).all()
        return encoded.cpu().numpy()


def run(model, batch_size, writers, profile_windows):
    execution_id = 'whole_parallel_' + model + '_' + timestamp().replace(':', '').replace('+', '_')
    config = configuration()
    source = Path(context()[0]['runs']) / config['source_run'] / 'source_audit.json'
    audit = read_json(source)
    destination = Path('profile_output') / execution_id if profile_windows else output_root()
    root = destination / 'maps'
    root.mkdir(parents=True, exist_ok=True)
    base_source = Path(__file__).with_name('selected_whole.py')
    identity = {'configuration': config, 'source_audit_sha256': sha256(source), 'selection_sha256': sha256(campaign_root() / 'selection.json'), 'code_sha256': sha256(base_source), 'input_code_sha256': sha256(Path(__file__).with_name('whole.py')), 'smoke': False, 'test_used': False}
    ensure_identity(root / 'request.json', identity)
    execution = {'run_id': execution_id, 'parent_asset_run': config['run_id'], 'source_sha256': sha256(Path(__file__)), 'base_source_sha256': sha256(base_source), 'batch_size': batch_size, 'writers': writers, 'profile_windows': profile_windows, 'started_at': timestamp(), 'test_used': False}
    write_json(destination / 'executions' / f'{execution_id}.json', execution)
    encoder = FastEncoder(model)
    rows = []
    for case, row in enumerate(audit['cases'][:1] if profile_windows else audit['cases']):
        if not profile_windows and all((root / run / f'case_{case:02d}.json').exists() for run in encoder.runs):
            continue
        volume, affine = load_case(row)
        geometries = {run: map_geometry(model, sae.request['site'], volume.shape, affine) for run, sae in encoder.dictionaries.items()}
        reference = next(iter(geometries.values()))
        indices = np.arange(int(np.prod(reference['tiles'])))
        if profile_windows:
            center = np.ravel_multi_index(tuple(reference['tiles'] // 2), reference['tiles'])
            indices = indices[center:center + profile_windows]
        paths, positions = {}, []
        for run_id in encoder.runs:
            path = root / run_id / f'case_{case:02d}.h5'
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, 'a') as saved:
                initialize(saved, run_id, geometries[run_id], len(indices), case)
                positions.append(int(saved.attrs['completed_tiles']))
            paths[run_id] = str(path)
        completed = min(positions)
        initial = completed
        started = time.perf_counter()
        groups = [encoder.runs[index::writers] for index in range(writers)]
        with ExitStack() as stack:
            pools = [stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=open_outputs, initargs=({run_id: paths[run_id] for run_id in group},))) for group in groups]
            pending = []

            def finish_pending():
                nonlocal completed
                if not pending:
                    return
                positions = [future.result() for future in pending]
                assert len(set(positions)) == 1
                completed = positions[0]
                pending.clear()
                elapsed = time.perf_counter() - started
                rate = (completed - initial) / elapsed
                progress = {'state': 'running', 'model': model, 'case': case, 'cases': len(audit['cases']), 'completed': completed, 'total': len(indices), 'tiles_per_second': rate, 'case_eta_seconds': (len(indices) - completed) / rate, 'updated_at': timestamp(), 'execution_run_id': execution_id}
                write_json(destination / f'progress_{model}.json', progress)
                if completed % (batch_size * 8) == 0 or completed == len(indices):
                    print(progress, flush=True)

            for offset in range(completed, len(indices), batch_size):
                if not profile_windows and pause_requested(campaign_root(), 'local'):
                    finish_pending()
                    write_json(destination / f'progress_{model}.json', {'state': 'paused', 'model': model, 'case': case, 'completed': completed, 'total': len(indices), 'updated_at': timestamp()})
                    raise SystemExit(75)
                selected = indices[offset:offset + batch_size]
                coordinates = np.asarray(np.unravel_index(selected, reference['tiles'])).T
                starts = coordinates * reference['tile_step'] + affine[:3, 3]
                values = encoder.extract([sample_input(volume, affine, model, position) for position in starts])
                blocks = {run_id: encoder.encode(run_id, values[encoder.dictionaries[run_id].request['site']]) for run_id in encoder.runs}
                finish_pending()
                pending = [pool.submit(write_batch, {run_id: blocks[run_id] for run_id in group}, coordinates, offset) for pool, group in zip(pools, groups)]
            finish_pending()
            for pool in pools:
                pool.submit(close_outputs).result()
        elapsed = time.perf_counter() - started
        for run_id, path in paths.items():
            with h5py.File(path, 'r+') as saved:
                assert saved.attrs['completed_tiles'] == len(indices)
                saved.attrs['status'] = 'profile_completed' if profile_windows else 'completed'
                saved.flush()
                receipt = {'state': saved.attrs['status'], 'run_id': run_id, 'case': case, 'tiles': len(indices), 'shape': geometries[run_id]['shape'].tolist(), 'dictionary_sha256': saved.attrs['dictionary_sha256'], 'seconds': elapsed, 'peak_vram_mib': torch.cuda.max_memory_allocated() / 2**20, 'execution_run_id': execution_id, 'resumed_tiles': initial, 'created_at': timestamp(), 'test_used': False}
            write_json(Path(path).with_suffix('.json'), receipt)
        result = {'case': case, 'windows': len(indices) - initial, 'seconds': elapsed, 'windows_per_second': (len(indices) - initial) / elapsed, 'peak_vram_mib': torch.cuda.max_memory_allocated() / 2**20}
        rows.append(result)
        print(result, flush=True)
    write_json(destination / 'executions' / f'{execution_id}_completed.json', {'state': 'completed', 'execution': execution, 'rows': rows, 'created_at': timestamp()})
    if not profile_windows:
        write_json(destination / f'completed_{model}.json', {'state': 'completed', 'runs': encoder.runs, 'cases': len(audit['cases']), 'execution_run_id': execution_id, 'created_at': timestamp()})
        write_json(destination / f'progress_{model}.json', {'state': 'completed', 'model': model, 'cases': len(audit['cases']), 'updated_at': timestamp()})
    print({'output': str(destination)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['fmcib', 'vista'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--writers', type=int, default=3)
    parser.add_argument('--profile-windows', type=int, default=0)
    args = parser.parse_args()
    assert args.batch_size > 0 and 1 <= args.writers <= 9
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args.model, args.batch_size, args.writers, args.profile_windows)
