import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STORAGE = json.loads((ROOT / "configs/storage.local.json").read_text())
SOURCES = json.loads((ROOT / "configs/model_sources.json").read_text())
os.environ["HF_HOME"] = STORAGE["hf_home"]
os.environ["HF_HUB_CACHE"] = STORAGE["hf_hub_cache"]
os.environ["HF_XET_CACHE"] = STORAGE["hf_xet_cache"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import HfApi, snapshot_download


REPOSITORIES = {
    "coralbay": "kaiko-ai/coralbay",
    "tapct": "fomofo/tap-ct-b-3d",
    "ctfm": "project-lighter/ct_fm_feature_extractor",
    "genesis": "MrGiovanni/ModelsGenesis",
    "vista": "nvidia/NV-Segment-CT",
    "automsc": "FLARE-MedFM/FLARE-AutoMSC-Val-Baseline",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "download"])
    parser.add_argument("--model", choices=[*REPOSITORIES, "fmcib"])
    args = parser.parse_args()
    destination = Path(STORAGE["checkpoints"]) / "published"
    destination.mkdir(parents=True, exist_ok=True)
    names = [args.model] if args.model else [*REPOSITORIES, "fmcib"]
    for name in names:
        if name == "fmcib":
            source = SOURCES[name]
            url = source["url"]
            if args.action == "inspect":
                print(json.dumps({name: source}, indent=2), flush=True)
                continue
            directory = destination / name / "zenodo-10528450"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "model_weights.torch"
            if not path.exists():
                temporary = path.with_suffix(".torch.part")
                subprocess.run(["curl.exe", "--fail", "--location", "--retry", "5", "--continue-at", "-", "--output", str(temporary), url], check=True)
                temporary.rename(path)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            with path.open("rb") as handle:
                md5 = hashlib.file_digest(handle, "md5").hexdigest()
            assert path.stat().st_size == source["bytes"] and md5 == source["md5"], "FMCIB asset integrity mismatch"
            manifest = {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": digest, "verified_zenodo_md5": md5}
        else:
            repo = REPOSITORIES[name]
            api = HfApi()
            info = api.model_info(repo, revision=SOURCES[name]["revision"], files_metadata=True)
            files = [{"path": file.rfilename, "bytes": file.size, "sha256": file.lfs.sha256 if file.lfs else None} for file in info.siblings]
            manifest = {"repo_id": repo, "revision": info.sha, "files": files}
            if args.action == "inspect":
                print(json.dumps({name: manifest}, indent=2), flush=True)
                continue
            patterns = {
                "coralbay": ["*.json", "*.py", "*.safetensors", "README.md", "LICENSE*", "requirements.txt"],
                "tapct": ["*.json", "*.py", "*.safetensors", "README.md", "LICENSE*"],
                "ctfm": ["*.json", "*.safetensors", "*.bin", "README.md", "LICENSE*"],
                "genesis": ["Genesis_Chest_CT.pt", "README.md", "LICENSE*"],
                "vista": ["vista3d_pretrained_model/model.pt", "vista3d_pretrained_model/config.json", "README.md", "LICENSE*"],
                "automsc": ["automsc_baseline.tar.gz", "README.md", "LICENSE*"],
            }[name]
            directory = destination / name / info.sha
            download_directory = Path("\\\\?\\" + str(directory.resolve())) if os.name == "nt" else directory
            snapshot_download(repo, revision=info.sha, local_dir=download_directory, allow_patterns=patterns, max_workers=2)
            manifest["directory"] = str(directory)
            manifest["downloaded"] = []
            for path in sorted(directory.rglob("*")):
                if path.is_file() and ".cache" not in path.parts:
                    with path.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    expected = next(item for item in files if item["path"] == path.relative_to(directory).as_posix())
                    assert path.stat().st_size == expected["bytes"], str(path)
                    if expected["sha256"]:
                        assert digest == expected["sha256"], str(path)
                    manifest["downloaded"].append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": digest})
        manifest["recorded_at"] = datetime.now(timezone.utc).isoformat()
        (destination / f"{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: verified manifest {destination / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
