"""Read public Hub metadata and test one file without downloading the dataset."""
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

REPO = 'FLARE-MedFM/FLARE-AutoMSC'
OUT = Path(__file__).resolve().parents[2] / 'data' / 'manifests'
OUT.mkdir(parents=True, exist_ok=True)

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'SCL-VMI/0.1'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response), response.headers.get('Link', '')

info, _ = get_json(f'https://huggingface.co/api/datasets/{REPO}')
config_path = OUT.parents[1] / 'configs' / 'download_flare.json'
sha = json.loads(config_path.read_text(encoding='utf-8'))['revision'] if config_path.exists() else info['sha']
url = f'https://huggingface.co/api/datasets/{REPO}/tree/{sha}?recursive=true&expand=false&limit=1000'
entries = []
while url:
    page, links = get_json(url)
    entries.extend(page)
    url = None
    for link in links.split(','):
        if 'rel="next"' in link:
            url = link[link.index('<')+1:link.index('>')]
files = [x for x in entries if x['type'] == 'file']
probe = next((x for x in files if x['path'].lower().endswith(('.zip','.tar','.gz','.7z'))), files[0])
request = urllib.request.Request(f'https://huggingface.co/datasets/{REPO}/resolve/{sha}/{probe["path"]}', method='HEAD')
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
except urllib.error.HTTPError as e:
    status = e.code
record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'repo_id':REPO,
          'revision':sha, 'gated':info.get('gated'), 'files':files,
          'file_count':len(files), 'total_bytes':sum(x.get('size',0) for x in files),
          'anonymous_file_probe':{'path':probe['path'],'http_status':status}}
(OUT/'flare_automsc_remote_inventory.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k != 'files'},ensure_ascii=False,indent=2))
from collections import defaultdict
groups = defaultdict(lambda: {'files': 0, 'bytes': 0})
for entry in files:
    parts = entry['path'].split('/')
    group = '/'.join(parts[:2]) if parts[0] == 'validation' else parts[0] if len(parts) > 1 else '_root'
    groups[group]['files'] += 1
    groups[group]['bytes'] += entry.get('size', 0)
print(json.dumps(dict(groups), ensure_ascii=False, indent=2))
