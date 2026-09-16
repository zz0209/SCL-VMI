"""Download the complete pinned Hub snapshot and verify every published file."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

WORKSPACE = Path(__file__).resolve().parents[2]

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Remote path escapes snapshot directory')
    return path

def file_digest(path: Path, entry: dict) -> tuple[str, str]:
    if entry.get('lfs'):
        expected = entry['lfs']['oid']
        digest = hashlib.sha256()
    else:
        expected = entry['oid']
        digest = hashlib.sha1()
        digest.update(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest(), expected

def check_inventory(public, authenticated):
    if authenticated['revision'] != public['revision'] or authenticated['repo_id'] != public['repo_id']:
        raise ValueError('Authenticated inventory identity mismatch')
    expected = {x['path']: x['size'] for x in public['files']}
    actual = {x['path']: x['size'] for x in authenticated['files']}
    if expected != actual or len(authenticated['files']) != len(expected):
        raise ValueError('Authenticated tree differs from the pinned public inventory')
    for entry in authenticated['files']:
        digest = entry['lfs']['oid'] if entry.get('lfs') else entry['oid']
        length = 64 if entry.get('lfs') else 40
        if not re.fullmatch('[0-9a-f]{' + str(length) + '}', digest):
            raise ValueError('Hub file hashes are unavailable or redacted; cannot verify content')

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    storage = json.loads((WORKSPACE/'configs/storage.local.json').read_text(encoding='utf-8'))
    inventory = json.loads((WORKSPACE/'data/manifests/flare_automsc_remote_inventory.json').read_text(encoding='utf-8'))
    config = json.loads((WORKSPACE/'configs/download_flare.json').read_text(encoding='utf-8'))
    if inventory['revision'] != config['revision']:
        raise ValueError('Inventory revision differs from the download configuration')
    if inventory['file_count'] != config['expected_file_count'] or inventory['total_bytes'] != config['expected_total_bytes']:
        raise ValueError('Inventory size/count differs from the download configuration')
    snapshot = Path(storage['dataset_root'])/'snapshots'/inventory['revision']
    if not Path(storage['storage_root']).is_dir():
        raise SystemExit('Configured storage root is unavailable')
    marker = Path(storage['project_storage'])/'storage_identity.json'
    if json.loads(marker.read_text(encoding='utf-8'))['storage_id'] != storage['storage_id']:
        raise ValueError('External storage identity mismatch')
    if not snapshot.resolve().is_relative_to(Path(storage['dataset_root']).resolve()):
        raise ValueError('Invalid snapshot destination')
    for env, key in [('HF_HOME','hf_home'),('HF_HUB_CACHE','hf_hub_cache'),('HF_XET_CACHE','hf_xet_cache'),('TEMP','temp_root'),('TMP','temp_root')]:
        os.environ[env] = storage[key]
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    from huggingface_hub import HfApi, RepoFile, get_hf_file_metadata, hf_hub_url, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError
    from filelock import FileLock, Timeout

    status_path = Path(storage['project_storage'])/'logs/flare_download_status.json'
    lock = FileLock(str(status_path.with_suffix('.lock')), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        print('A download/preflight for this storage is already running; no duplicate started.')
        return 12
    state = {'repo_id':inventory['repo_id'], 'revision':inventory['revision'], 'destination':str(snapshot),
             'expected_file_count':inventory['file_count'], 'expected_bytes':inventory['total_bytes'],
             'pid':os.getpid(), 'started_utc':datetime.now(timezone.utc).isoformat()}
    def status(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(status_path,state)
        print(json.dumps({'phase':phase, **extra},ensure_ascii=False), flush=True)
    try:
        if not args.verify_only:
            status('checking_access')
            first = next(x for x in inventory['files'] if x['path'].endswith('.nii.gz'))
            try:
                get_hf_file_metadata(hf_hub_url(inventory['repo_id'],first['path'],repo_type='dataset',revision=inventory['revision']))
            except (GatedRepoError, LocalTokenNotFoundError) as exc:
                status('blocked_access', reason='Hugging Face login and approved gated-dataset access are required.', error_type=type(exc).__name__)
                return 10
            except HfHubHTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                status('blocked_access' if code in (401,403) else 'preflight_failed', http_status=code, error_type=type(exc).__name__)
                return 10 if code in (401,403) else 11
        if args.preflight_only:
            status('access_ready')
            return 0
        # The anonymous gated-repository tree replaces LFS hashes with '*'.
        # Fetch authoritative hashes after login, before transferring large files.
        manifest_path = status_path.parent/f'flare_authenticated_inventory_{inventory["revision"]}.json'
        if manifest_path.is_file():
            authenticated = json.loads(manifest_path.read_text(encoding='utf-8'))
        elif args.verify_only:
            status('verification_blocked', reason='Authenticated hash inventory has not been saved yet.')
            return 17
        else:
            status('fetching_authenticated_hashes')
            files = []
            for entry in HfApi().list_repo_tree(inventory['repo_id'], repo_type='dataset', revision=inventory['revision'], recursive=True):
                if isinstance(entry, RepoFile):
                    row = {'path':entry.path, 'size':entry.size, 'oid':entry.blob_id}
                    if entry.lfs:
                        row['lfs'] = {'oid':entry.lfs.sha256, 'size':entry.lfs.size}
                    files.append(row)
            authenticated = {'repo_id':inventory['repo_id'], 'revision':inventory['revision'], 'files':files}
        check_inventory(inventory, authenticated)
        atomic_json(manifest_path, authenticated)
        inventory['files'] = authenticated['files']
        already = sum(e['size'] for e in inventory['files'] if safe_file(snapshot,e['path']).is_file() and safe_file(snapshot,e['path']).stat().st_size == e['size'])
        required = max(0,inventory['total_bytes']-already) + 20*2**30
        if not args.verify_only and shutil.disk_usage(storage['storage_root']).free < required:
            status('blocked_space', required_free_bytes=required)
            return 13
        if not args.verify_only:
            status('downloading', previously_present_bytes=already)
            for attempt in range(1,4):
                try:
                    snapshot_download(repo_id=inventory['repo_id'],repo_type='dataset',revision=inventory['revision'],
                                      local_dir=snapshot,max_workers=config['max_workers'])
                    break
                except GatedRepoError:
                    status('blocked_access',reason='Gated access was rejected during transfer.')
                    return 10
                except Exception as exc:
                    status('download_retry' if attempt<3 else 'download_failed',attempt=attempt,error_type=type(exc).__name__)
                    if attempt == 3: return 14
                    time.sleep(10*attempt)
        status('verifying',verified_files=0,verified_bytes=0)
        verified_bytes = 0
        errors = []
        receipt = status_path.parent/f'flare_verification_{inventory["revision"]}.jsonl'
        with receipt.open('w',encoding='utf-8') as stream:
            for i,entry in enumerate(inventory['files'],1):
                path = safe_file(snapshot,entry['path'])
                valid = path.is_file() and path.stat().st_size == entry['size']
                actual = None
                if valid:
                    actual, expected = file_digest(path,entry)
                    valid = actual == expected
                row={'path':entry['path'],'size':entry['size'],'verified':valid,'digest':actual}
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                if valid: verified_bytes += entry['size']
                else: errors.append(entry['path'])
                if i % 250 == 0: status('verifying',verified_files=i-len(errors),verified_bytes=verified_bytes,failed_files=len(errors))
        status('complete' if not errors else 'verification_failed',verified_files=len(inventory['files'])-len(errors),
               verified_bytes=verified_bytes,failed_files=len(errors),verification_receipt=str(receipt))
        return 0 if not errors else 15
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        return 16
    finally:
        lock.release()

if __name__ == '__main__':
    sys.exit(main())
