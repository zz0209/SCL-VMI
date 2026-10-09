import csv
import hashlib
import io
import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, find_objects
from skimage.measure import marching_cubes


COLORS = ["#ffc857", "#57d3b4", "#e692c3", "#79a9ff", "#e99065"]


class DatasetStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.storage = json.loads((self.workspace / "configs/storage.local.json").read_text())
        self.download = json.loads((self.workspace / "configs/download_flare.json").read_text())
        self.root = Path(self.storage["dataset_root"]) / "snapshots" / self.download["revision"]
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset snapshot unavailable: {self.root}")
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        self.cache = Path(self.storage["project_storage"]) / "browser_cache" / self.download["revision"] / source_hash
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.loaded_key = None
        self.loaded = None
        self.datasets = {}
        self.index()

    def index(self):
        for descriptor in sorted(self.root.rglob("dataset.json")):
            folder = descriptor.parent
            meta = json.loads(descriptor.read_text(encoding="utf-8"))
            dataset_id = folder.name
            with (folder / "cls_data.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            labels = {row["identifier"]: {k: v for k, v in row.items() if k != "identifier"} for row in rows}
            if len(labels) != len(rows):
                raise ValueError(f"Duplicate classification identifiers: {dataset_id}")
            fold_by_case = {}
            split_path = folder / "splits_final.json"
            if split_path.exists():
                for fold, split in enumerate(json.loads(split_path.read_text())):
                    for case in split["val"]:
                        if case in fold_by_case:
                            raise ValueError(f"Repeated validation member: {dataset_id}/{case}")
                        fold_by_case[case] = fold
            clinical = {}
            if dataset_id == "Dataset005_LUNA25":
                with (folder / "clinical_data.csv").open(encoding="utf-8-sig", newline="") as handle:
                    clinical = {row["AnnotationID"]: row for row in csv.DictReader(handle)}
            cases = {}
            for partition in ("Tr", "Ts"):
                for path in sorted((folder / f"images{partition}").glob("*_0000.nii.gz")):
                    case = path.name.removesuffix("_0000.nii.gz")
                    if case in cases:
                        raise ValueError(f"Duplicate image identifier: {dataset_id}/{case}")
                    fold = fold_by_case.get(case)
                    role = "source_training" if partition == "Tr" else "source_unlabeled"
                    if "validation" in folder.relative_to(self.root).parts:
                        role = "provider_validation" if partition == "Tr" else "provider_unlabeled"
                    elif dataset_id == "Dataset005_LUNA25" and fold is not None:
                        role = "proposed_test" if fold == 0 else "proposed_dev" if fold == 1 else "proposed_train"
                    cases[case] = {"id": case, "partition": partition, "labels": labels.get(case, {}), "fold": fold, "role": role, "has_mask": (folder / f"labels{partition}" / f"{case}.nii.gz").exists()}
            counts = {}
            for row in cases.values():
                for task, value in row["labels"].items():
                    counts.setdefault(task, Counter())[value or "missing"] += 1
            self.datasets[dataset_id] = {"folder": folder, "meta": meta, "cases": cases, "clinical": clinical, "counts": counts}
            print(f"Indexed {dataset_id}: {len(cases)} cases", flush=True)

    def dataset(self, dataset_id):
        if dataset_id not in self.datasets:
            raise KeyError("Unknown dataset")
        return self.datasets[dataset_id]

    def case(self, dataset_id, case_id):
        dataset = self.dataset(dataset_id)
        if case_id not in dataset["cases"]:
            raise KeyError("Unknown case")
        return dataset, dataset["cases"][case_id]

    def catalog(self):
        output = []
        for key, dataset in self.datasets.items():
            output.append({"id": key, "name": key.split("_", 1)[1], "count": len(dataset["cases"]), "channels": dataset["meta"]["channel_names"], "tasks": dataset["meta"].get("classification_labels", {}), "label_counts": dataset["counts"], "mask_count": sum(row["has_mask"] for row in dataset["cases"].values()), "description": dataset["meta"].get("description", ""), "patient_count": len({row["PatientID"] for case, row in dataset["clinical"].items() if case in dataset["cases"]}) if dataset["clinical"] else None})
        return {"revision": self.download["revision"], "datasets": output}

    def list_cases(self, dataset_id, search="", role="all", label="all", offset=0, limit=60):
        dataset = self.dataset(dataset_id)
        rows = list(dataset["cases"].values())
        if search:
            rows = [row for row in rows if search.casefold() in row["id"].casefold()]
        if role != "all":
            rows = [row for row in rows if row["role"] == role]
        if label != "all":
            task, value = label.split(":", 1)
            rows = [row for row in rows if row["labels"].get(task) == value]
        return {"total": len(rows), "offset": offset, "items": rows[offset:offset+limit]}

    def volume(self, dataset_id, case_id, channel):
        dataset, case = self.case(dataset_id, case_id)
        if str(channel) not in dataset["meta"]["channel_names"]:
            raise KeyError("Unknown image channel")
        key = (dataset_id, case_id, channel)
        if key == self.loaded_key:
            return self.loaded
        self.loaded = None
        self.loaded_key = None
        path = dataset["folder"] / f"images{case['partition']}" / f"{case_id}_{channel:04d}.nii.gz"
        native = nib.load(path)
        if len(native.shape) != 3:
            raise ValueError("Expected a three-dimensional NIfTI image")
        image = nib.as_closest_canonical(native)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("Image contains non-finite voxel values")
        mask = None
        mask_path = dataset["folder"] / f"labels{case['partition']}" / f"{case_id}.nii.gz"
        if case["has_mask"]:
            source_mask = nib.load(mask_path)
            if source_mask.shape != native.shape or not np.allclose(source_mask.affine, native.affine, atol=1e-4):
                raise ValueError("Image and mask geometry differ; overlay requires explicit registration")
            mask_data = nib.as_closest_canonical(source_mask).get_fdata(dtype=np.float32)
            if not np.isfinite(mask_data).all() or not np.array_equal(mask_data, np.rint(mask_data)):
                raise ValueError("Mask contains invalid label values")
            if not set(np.unique(mask_data)).issubset(set(dataset["meta"]["labels"].values())):
                raise ValueError("Mask label is absent from the dataset descriptor")
            mask = mask_data.astype(np.uint16)
        loaded = {"image": image, "native": native, "data": data, "mask": mask, "path": path}
        self.loaded_key, self.loaded = key, loaded
        return loaded

    def describe(self, dataset_id, case_id, channel):
        with self.lock:
            dataset, case = self.case(dataset_id, case_id)
            loaded = self.volume(dataset_id, case_id, channel)
            image, data, mask = loaded["image"], loaded["data"], loaded["mask"]
            spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
            voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
            quantiles = np.percentile(data, [.5, 50, 99.5]).tolist()
            modality = dataset["meta"]["channel_names"][str(channel)]
            default_window = [-1000, 400] if modality.upper() == "CT" else [quantiles[0], quantiles[2]]
            if default_window[0] == default_window[1]:
                default_window[1] = default_window[0]+1
            center = (np.asarray(data.shape)//2).tolist()
            regions = []
            if mask is not None:
                objects = find_objects(mask)
                for name, label in dataset["meta"]["labels"].items():
                    if label == 0:
                        continue
                    count = int(np.count_nonzero(mask == label))
                    bounds = objects[label-1] if label <= len(objects) else None
                    regions.append({"name": name, "value": label, "color": COLORS[(label-1) % len(COLORS)], "voxels": count, "volume_mm3": count*voxel_volume, "bounds": [[axis.start, axis.stop-1] for axis in bounds] if bounds else None, "touches_edge": bool(bounds and any(axis.start == 0 or axis.stop == data.shape[i] for i, axis in enumerate(bounds)))})
                populated = [r for r in regions if r["voxels"]]
                if populated:
                    center = [(a+b)//2 for a,b in max(populated, key=lambda r:r["voxels"])["bounds"]]
            histogram, edges = np.histogram(data, bins=64)
            result = {**case, "dataset": dataset_id, "channel": channel, "modality": modality, "shape": list(data.shape), "native_shape": list(loaded["native"].shape), "native_orientation": list(nib.aff2axcodes(loaded["native"].affine)), "orientation": list(nib.aff2axcodes(image.affine)), "spacing": spacing.tolist(), "coverage_mm": (np.asarray(data.shape)*spacing).tolist(), "affine": image.affine.tolist(), "obliquity_degrees": np.rad2deg(nib.affines.obliquity(image.affine)).tolist(), "range": [float(data.min()), float(data.max())], "quantiles": quantiles, "window": default_window, "center": center, "regions": regions, "histogram": {"counts": histogram.tolist(), "edges": edges.tolist()}, "clinical": dataset["clinical"].get(case_id, {}), "image_bytes": loaded["path"].stat().st_size}
            location = self.cache / dataset_id / case_id
            location.mkdir(parents=True, exist_ok=True)
            (location / f"channel_{channel}.json").write_text(json.dumps(result), encoding="utf-8")
            with (self.cache / "viewed_cases.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": datetime.now().astimezone().isoformat(), "dataset": dataset_id, "case": case_id, "channel": channel, "role": case["role"]}) + "\n")
            return result

    def slice_png(self, dataset_id, case_id, channel, axis, index, low, high, overlay, region):
        if axis not in (0,1,2) or not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("Invalid slice axis or display window")
        with self.lock:
            loaded = self.volume(dataset_id, case_id, channel)
            if not 0 <= index < loaded["data"].shape[axis]:
                raise ValueError("Slice index is outside the volume")
            plane = np.take(loaded["data"], index, axis=axis).T
            gray = np.rint(np.clip((plane-low)/(high-low), 0, 1)*255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            if overlay and loaded["mask"] is not None:
                segment = np.take(loaded["mask"], index, axis=axis).T
                for label in np.unique(segment):
                    if label == 0 or (region != -1 and label != region):
                        continue
                    selected = segment == label
                    edge = selected & ~binary_erosion(selected)
                    color = COLORS[(int(label)-1) % len(COLORS)]
                    rgb[edge] = [int(color[i:i+2], 16) for i in (1,3,5)]
            buffer = io.BytesIO()
            Image.fromarray(np.flipud(rgb)).save(buffer, format="PNG")
            return buffer.getvalue()

    def mesh(self, dataset_id, case_id, channel, region):
        with self.lock:
            loaded = self.volume(dataset_id, case_id, channel)
            mask = loaded["mask"]
            if mask is None:
                return {"meshes": [], "reason": "该病例未提供分割标注。"}
            location = self.cache / dataset_id / case_id / f"mesh_{channel}_{region}.json"
            if location.exists():
                return json.loads(location.read_text())
            dataset = self.dataset(dataset_id)
            meshes = []
            for name, label in dataset["meta"]["labels"].items():
                if label == 0 or (region != -1 and label != region):
                    continue
                selected = mask == label
                if not selected.any():
                    continue
                bounds = find_objects(selected.astype(np.uint8))[0]
                start = np.array([axis.start for axis in bounds])
                cropped = np.pad(selected[bounds], 1)
                step = max(1, int(np.ceil((cropped.size / 2000000) ** (1/3))))
                vertices, faces, _, _ = marching_cubes(cropped, level=.5, step_size=step, allow_degenerate=False)
                vertices += start-1
                vertices = nib.affines.apply_affine(loaded["image"].affine, vertices) - loaded["image"].affine[:3, 3]
                meshes.append({"name": name, "value": label, "color": COLORS[(label-1) % len(COLORS)], "vertices": np.round(vertices, 4).tolist(), "faces": faces.tolist(), "step": step})
            result = {"meshes": meshes, "reason": "当前标注全部为背景。" if not meshes else None, "method": "Marching cubes 0.5 surface; source affine; no smoothing. Padding closes surfaces at crop boundaries."}
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_text(json.dumps(result), encoding="utf-8")
            return result
