import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from .automsc import paths
from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .models import load_encoder, source_module, spatial_features
from .preprocessing import preprocess


def i3d_input(image_path):
    from .luna25 import configure
    configure(Path(image_path).parent, 1, 1)
    utilities = source_module("luna25_inference_conversion", ROOT / "third_party/luna25/preprocessing/utils.py")
    loader = source_module("luna25_inference_loader", ROOT / "third_party/luna25/dataloader.py")
    image, metadata = utilities.itk_image_to_numpy_image(sitk.ReadImage(str(image_path)))
    assert image.shape == (64, 128, 128)
    patch = loader.extract_patch(CTData=image, coord=tuple(np.array(image.shape) // 2), srcVoxelOrigin=metadata["origin"], srcWorldMatrix=metadata["transform"], srcVoxelSpacing=metadata["spacing"], output_shape=(64, 64, 64), voxel_spacing=(50 / 64,) * 3, coord_space_world=False, mode="3D")
    return torch.from_numpy(loader.clip_and_scale(patch.astype(np.float32))).unsqueeze(0)


def i3d_predict(checkpoint, image_path):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    image = i3d_input(image_path).cuda()
    module = source_module("luna25_inference_model", ROOT / "third_party/luna25/models/model_3d.py")
    model = module.I3D(num_classes=1, input_channels=3, pre_trained=False, freeze_bn=True).cuda()
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    with torch.inference_mode():
        probability = float(model(image).sigmoid().reshape(-1)[0])
    return {"model": "luna25_i3d", "malignancy_probability": probability, "input_shape": list(image.shape), "checkpoint": str(checkpoint), "purpose": "research inference"}


def fm_predict(name, head_directory, image_path):
    _, configuration, _ = context()
    head_directory = Path(head_directory)
    result = read_json(head_directory / "result.json")
    feature_configuration = read_json(Path(result["feature_directory"]) / "config.json")
    assert feature_configuration["model"] == name
    assert feature_configuration["preprocessing"] == configuration["models"][name]
    for file, digest in feature_configuration["code"].items():
        assert sha256(ROOT / "src/sclvmi" / file) == digest, f"Feature code changed: {file}"
    model, _ = load_encoder(name)
    model = model.cuda()
    image = preprocess(image_path, name).unsqueeze(0).cuda()
    with torch.inference_mode():
        vector = spatial_features(model, name, image).mean((2, 3, 4)).cpu().numpy()
    head = joblib.load(head_directory / "head.joblib")
    probability = float(head.predict_proba(vector)[0, 1])
    return {"model": name, "malignancy_probability": probability, "input_shape": list(image.shape), "embedding_shape": list(vector.shape), "head": str(head_directory), "purpose": "research inference"}


def automsc_predict(source_run, output_id, limit, resume=False):
    storage, _, _ = context()
    root = paths(source_run)
    status = read_json(root / "training_status.json")
    model_directory = Path(status["checkpoint"]).parent.parent
    output = Path(storage["runs"]) / output_id
    output.mkdir(parents=True, exist_ok=resume)
    request = {"source_run": source_run, "checkpoint_sha256": sha256(status["checkpoint"]), "limit": limit}
    request_path = output / "request.json"
    if request_path.exists():
        assert read_json(request_path) == request, "Inference resume requires the same checkpoint and cases"
    write_json(request_path, request)
    module = source_module("automsc_inference", ROOT / "third_party/automsc/segcls_ensemble_infer.py")
    torch.set_num_threads(4)
    predictor = module.SimplePredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True, device=torch.device("cuda", 0), verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(model_directory), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    predictor.network.cuda()
    rows = []
    cases = select_cases(load_manifest(), "development", limit)
    for row in cases.itertuples():
        receipt = output / f"{row.identifier}.json"
        if receipt.exists():
            rows.append(read_json(receipt))
            continue
        image, properties = module.SimpleITKIO().read_images([row.image])
        segmentation, probability = predictor.inference(image, properties, use_softmax=False)
        image_out = sitk.GetImageFromArray(np.asarray(segmentation, dtype=np.uint8))
        source = sitk.ReadImage(row.image)
        assert image_out.GetSize() == source.GetSize()
        image_out.CopyInformation(source)
        sitk.WriteImage(image_out, str(output / f"{row.identifier}.nii.gz"))
        score = float(probability.reshape(-1)[0])
        assert torch.all(probability == probability.reshape(-1)[0]), "Unexpected non-scalar binary probability"
        assert 0 <= score <= 1
        result = {"identifier": row.identifier, "PatientID": row.PatientID, "label": row.label, "probability": score}
        write_json(receipt, result)
        rows.append(result)
        print(f"AutoMSC prediction {len(rows)}/{len(cases)}", flush=True)
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    write_json(output / "status.json", {"status": "completed", "completed_at": timestamp(), "source_checkpoint": status["checkpoint"], "cases": len(rows), "tile_step_size": 0.5, "mirroring": True, "segmentation_decision": "official use_softmax=False", "test_evaluated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fm = commands.add_parser("fm")
    fm.add_argument("--model", required=True)
    fm.add_argument("--head", required=True)
    fm.add_argument("--image", required=True)
    i3d = commands.add_parser("i3d")
    i3d.add_argument("--checkpoint", required=True)
    i3d.add_argument("--image", required=True)
    auto = commands.add_parser("automsc")
    auto.add_argument("--source-run", required=True)
    auto.add_argument("--output-id", required=True)
    auto.add_argument("--limit", type=int, default=2)
    auto.add_argument("--all-development", action="store_true")
    auto.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fm":
        print(fm_predict(args.model, args.head, args.image), flush=True)
    elif args.command == "i3d":
        print(i3d_predict(args.checkpoint, args.image), flush=True)
    else:
        automsc_predict(args.source_run, args.output_id, None if args.all_development else args.limit, args.resume)
