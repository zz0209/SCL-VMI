"""Create external storage and local path configuration without moving data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

WORKSPACE = Path(__file__).resolve().parents[2]


def build_config(root: Path, storage_id: str) -> dict:
    project = root / 'projects' / storage_id
    paths = {
        'storage_root': root,
        'project_storage': project,
        'dataset_root': root / 'datasets' / 'FLARE-MedFM' / 'FLARE-AutoMSC',
        'hf_home': root / 'cache' / 'huggingface',
        'hf_hub_cache': root / 'cache' / 'huggingface' / 'hub',
        'hf_xet_cache': root / 'cache' / 'huggingface' / 'xet',
        'temp_root': project / 'tmp',
        'environments': project / 'environments',
        'preprocessed': project / 'preprocessed',
        'activations': project / 'activations',
        'checkpoints': project / 'checkpoints',
        'runs': project / 'runs',
        'logs': project / 'logs',
    }
    return {'storage_id': storage_id, 'codename': 'SCL-VMI',
            **{key: str(value) for key, value in paths.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-root', required=True, type=Path)
    parser.add_argument('--storage-id', default='SCL-VMI')
    args = parser.parse_args()
    root = args.storage_root.expanduser().resolve()
    if not args.storage_id or args.storage_id in ('.', '..') or any(c in args.storage_id for c in '/\\:'):
        parser.error('storage-id must be a single directory name')
    if root == WORKSPACE or root.is_relative_to(WORKSPACE):
        parser.error('Choose storage outside the source repository')
    config = build_config(root, args.storage_id)
    config_path = WORKSPACE / 'configs' / 'storage.local.json'
    if config_path.exists():
        current = json.loads(config_path.read_text(encoding='utf-8'))
        if current['storage_id'] != args.storage_id or Path(current['storage_root']).resolve() != root:
            parser.error('Existing local storage configuration differs; refusing to overwrite it')
    ancestor = root
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        parser.error('Storage volume is unavailable')
    if shutil.disk_usage(ancestor).free < 100 * 2**30:
        parser.error('Storage requires at least 100 GiB free for the pinned snapshot and headroom')
    for key, value in config.items():
        if key not in ('storage_id', 'codename'):
            path = Path(value)
            if not path.resolve().is_relative_to(root):
                raise ValueError('Storage path escapes the selected root')
            path.mkdir(parents=True, exist_ok=True)
    marker = Path(config['project_storage']) / 'storage_identity.json'
    identity = {'storage_id': args.storage_id, 'codename': 'SCL-VMI', 'workspace': str(WORKSPACE)}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))['storage_id'] != args.storage_id:
            raise ValueError('Storage identity mismatch')
    else:
        marker.write_text(json.dumps(identity, indent=2) + chr(10), encoding='utf-8')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + chr(10), encoding='utf-8')
    print(f'Local storage configuration saved to {config_path}')


if __name__ == '__main__':
    main()
