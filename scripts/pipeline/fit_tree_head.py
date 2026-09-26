import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.frozen_features import MODELS, SUITE
from sclvmi.head import feature_table, metrics


def run(name):
    storage, _, _ = context()
    root = Path(storage["runs"]) / SUITE
    directory = Path(read_json(root / f"{name}_features.json")["directory"])
    output = root / f"{name}_native_gbdt"
    if (output / "result.json").exists():
        print(read_json(output / "result.json"), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading {name} train/development embeddings", flush=True)
    train, x = feature_table(directory, "train")
    dev, v = feature_table(directory, "development")
    assert set(train.PatientID).isdisjoint(dev.PatientID)
    assert np.isfinite(x).all() and np.isfinite(v).all()
    configuration = {"max_iter":200,"max_leaf_nodes":7,"max_depth":3,"learning_rate":0.05,"min_samples_leaf":20,"l2_regularization":1.0,"early_stopping":False,"random_state":2025,"verbose":2}
    ix = train.groupby("label", group_keys=False).head(64).index.to_numpy()
    with threadpool_limits(limits=4):
        smoke = HistGradientBoostingClassifier(**{**configuration,"max_iter":3}).fit(x[ix],train.iloc[ix].label)
        smoke_p = smoke.predict_proba(v[:32])[:,1]
        assert np.isfinite(smoke_p).all() and np.ptp(smoke.predict_proba(x[ix])[:,1])>0
        write_json(output / "smoke.json", {"status":"passed","real_training_cases":len(ix),"iterations":smoke.n_iter_,"completed_at":timestamp()})
        started = time.perf_counter()
        head = HistGradientBoostingClassifier(**configuration).fit(x,train.label)
        p = head.predict_proba(v)[:,1]
    joblib.dump(head, output / "head.joblib")
    np.testing.assert_array_equal(p,joblib.load(output / "head.joblib").predict_proba(v)[:,1])
    dev.assign(probability=p).to_csv(output / "development_predictions.csv",index=False)
    result = {"model":name,"arm":"native_gbdt","run_id":output.name,"status":"completed","metrics":metrics(dev.label,p),"configuration":configuration,"seconds":time.perf_counter()-started,"feature_directory":str(directory),"feature_config_sha256":sha256(directory / "config.json"),"source_sha256":sha256(Path(__file__)),"completed_at":timestamp(),"reload":"passed","test_used":False,"selection":"One fixed shallow-tree configuration; no hyperparameter search"}
    write_json(output / "result.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=MODELS,default="fmcib")
    args=parser.parse_args()
    run(args.model)
