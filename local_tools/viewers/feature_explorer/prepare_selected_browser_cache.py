import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
import psutil
from filelock import FileLock

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


def prepare(source, destination, smoke_features):
    target = destination / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix('.json')
    source_receipt = read_json(source.with_suffix('.json'))
    identity = {'source_bytes': source.stat().st_size, 'source_receipt_sha256': sha256(source.with_suffix('.json')), 'generator_sha256': sha256(__file__), 'smoke_features': smoke_features}
    with FileLock(str(target) + '.lock', timeout=1):
        if receipt.exists():
            saved_receipt = read_json(receipt)
            assert saved_receipt['identity'] == identity
            assert target.stat().st_size == saved_receipt['bytes']
            return saved_receipt
        started = time.perf_counter()
        partial = target.with_suffix('.partial.h5')
        print({'state': 'reading_source', 'run': source.parent.name, 'case': source_receipt['case'], 'bytes': identity['source_bytes']}, flush=True)
        with h5py.File(source, 'r', driver='core', backing_store=False) as saved, h5py.File(partial, 'a') as output:
            assert saved.attrs['status'] == 'completed'
            shape = saved['values'].shape
            if 'values' not in output:
                for key, value in saved.attrs.items():
                    output.attrs[key] = value
                output.attrs['status'] = 'preparing'
                output.attrs['completed_features'] = 0
                for key, value in identity.items():
                    output.attrs[key] = value
                output.create_dataset('values', shape, dtype='float16', chunks=(1, *shape[1:]), compression='lzf', shuffle=True)
                output.create_dataset('maximum', data=saved['maximum'][...])
            for key, value in identity.items():
                assert output.attrs[key] == value
            limit = min(smoke_features, shape[0]) if smoke_features else shape[0]
            for first in range(int(output.attrs['completed_features']), limit, 16):
                last = min(first + 16, limit)
                values = saved['values'][first:last]
                assert np.isfinite(values).all() and (values >= 0).all()
                output['values'][first:last] = values
                np.testing.assert_array_equal(output['values'][first:last], values)
                output.flush()
                output.attrs['completed_features'] = last
                output.flush()
                elapsed = time.perf_counter() - started
                print({'state': 'preparing', 'run': source.parent.name, 'case': source_receipt['case'], 'features': last, 'total': limit, 'seconds': elapsed, 'bytes': partial.stat().st_size}, flush=True)
            output.attrs['status'] = 'smoke_completed' if smoke_features else 'completed'
            output.flush()
        partial.replace(target)
        result = {'state': 'smoke_completed' if smoke_features else 'completed', 'identity': identity, 'run_id': source.parent.name, 'case': source_receipt['case'], 'features': limit, 'shape': list(shape), 'bytes': target.stat().st_size, 'sha256': sha256(target), 'seconds': time.perf_counter() - started, 'verification': 'Every copied float16 voxel compared with source', 'created_at': timestamp(), 'test_used': False}
        write_json(receipt, result)
        print(result, flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--case', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke-features', type=int, default=0)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4 and args.smoke_features >= 0
    config = read_json(ROOT / 'configs/sae_whole_viewer.json')
    root = Path(context()[0]['runs']) / config['run_id']
    destination = args.output_root or root / 'browser_cache'
    assert not args.smoke_features or args.output_root
    sources = sorted((root / 'maps').glob(f'{args.run or "*"}/case_*.h5'))
    if args.case is not None:
        sources = [path for path in sources if path.stem == f'case_{args.case:02d}']
    assert sources and all(path.with_suffix('.json').exists() for path in sources)
    if args.run is None and args.case is None:
        assert len(sources) == 144
    required = args.workers * (max(path.stat().st_size for path in sources) + 2 * 1024**3)
    assert psutil.virtual_memory().available > required
    identity = {'generator_sha256': sha256(__file__), 'source_request_sha256': sha256(root / 'maps/request.json'), 'precision': 'Exact float16', 'compression': 'lzf', 'chunking': 'One complete feature per chunk', 'smoke_features': args.smoke_features, 'files': [path.relative_to(root).as_posix() for path in sources]}
    request = destination / 'request.json'
    if request.exists():
        assert read_json(request) == identity
    else:
        write_json(request, identity)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare, path, destination, args.smoke_features) for path in sources]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(destination / 'progress.json', {'state': 'preparing_browser_cache', 'completed_files': len(rows), 'total_files': len(sources), 'updated_at': timestamp()})
    result = {'state': 'smoke_completed' if args.smoke_features else 'completed', 'files': len(rows), 'bytes': sum(row['bytes'] for row in rows), 'created_at': timestamp(), 'generator_sha256': identity['generator_sha256'], 'test_used': False}
    write_json(destination / 'completed.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
