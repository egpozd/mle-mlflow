"""Register the verified reconstructed baseline. Repeat calls reuse Version 1."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from common import ROOT, RUNTIME, artifact_root, configure, redact, required

EXPERIMENT = "real_estate_model_improvement"
MODEL_NAME = "real_estate_price_model"
METADATA = ROOT / "model_improvement/baseline"
HISTORICAL = {"RMSE": 10547270.062205384, "MAE": 3744622.8052026215, "R2": 0.8948994777286887}
EXPECTED = {
    "data/initial_data.csv": (19414324, "84045cd0f0e5a6f7a36eb4e2320fa41f"),
    "data/train.csv": (13499140, "2d4bb3ba2091eb269bc56a93f05f9119"),
    "data/test.csv": (3375734, "1d5874c7f4950c457e1ff2678d44a121"),
    "cv_results/metrics.json": (95, "c76d1f590d0cafcb94f4078b26faca6f"),
    "models/fitted_model.pkl": (1269774, "bd7cc21e56d9b904832d7c3ed2878e82")}
TAGS = {"stage": "baseline", "baseline_origin": "reconstructed", "data_reproduction": "byte_identical",
        "historical_metrics": "exact", "historical_model_md5": "ee7aa528bbdcdc207747e7ddc1194182",
        "reconstructed_model_md5": "bd7cc21e56d9b904832d7c3ed2878e82",
        "source_project_git_sha": "e2decc886f6a9a5d633ff6e710ba2a74cf2db6a2"}


def validate_recovery(recovery):
    import yaml
    for relative, (size, md5) in EXPECTED.items():
        p = recovery / relative
        if not p.is_file() or p.stat().st_size != size or hashlib.md5(p.read_bytes()).hexdigest() != md5:
            raise ValueError("Recovery artifact mismatch: " + relative)
    for source, target in [("params.yaml", "params.yaml"), ("cv_results/metrics.json", "metrics.json"),
                           ("environment.freeze.txt", "environment.freeze.txt")]:
        if (recovery / source).read_bytes() != (METADATA / target).read_bytes():
            raise ValueError("Baseline metadata mismatch: " + target)
    metrics = json.loads((recovery / "cv_results/metrics.json").read_text())
    assert {k.upper(): v for k, v in metrics.items()} == HISTORICAL
    provenance = json.loads((METADATA / "provenance.json").read_text())
    assert provenance["reconstructed_model_md5"] == TAGS["reconstructed_model_md5"]
    return yaml.safe_load((METADATA / "params.yaml").read_text())


def versions(client):
    return list(client.search_model_versions("name='" + MODEL_NAME + "'"))


def artifact_files(client, run_id, path=""):
    result = []
    for item in client.list_artifacts(run_id, path):
        result.extend(artifact_files(client, run_id, item.path) if item.is_dir else [item.path])
    return sorted(result)


def verify(client, number, recovery, params, reused):
    import joblib
    import mlflow
    import mlflow.catboost
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    experiment = client.get_experiment_by_name(EXPERIMENT)
    version = client.get_model_version(MODEL_NAME, str(number))
    run = client.get_run(version.run_id)
    if run.info.experiment_id != experiment.experiment_id or run.info.status != "FINISHED":
        raise ValueError("Unexpected baseline run/experiment/status")
    for k, v in TAGS.items():
        if run.data.tags.get(k) != v or version.tags.get(k) != v:
            raise ValueError("Baseline identity tag mismatch: " + k)
    for k, v in params["model"].items():
        if run.data.params.get(k) != str(v):
            raise ValueError("Baseline parameter mismatch: " + k)
    assert {k: run.data.metrics.get(k) for k in HISTORICAL} == HISTORICAL
    artifacts = artifact_files(client, run.info.run_id)
    for name in ["params.yaml", "metrics.json", "provenance.json", "environment.freeze.txt", "data_hashes.json"]:
        if "baseline/" + name not in artifacts:
            raise ValueError("Missing artifact: " + name)
    assert "model/MLmodel" in artifacts and "model/input_example.json" in artifacts
    uri = "models:/" + MODEL_NAME + "/" + str(number)
    destination = Path(tempfile.mkdtemp(prefix="registry-verification-", dir=RUNTIME / "tmp"))
    loaded = mlflow.catboost.load_model(uri, dst_path=str(destination))
    signature = mlflow.models.Model.load(str(next(destination.rglob("MLmodel")))).signature
    assert signature is not None
    test = pd.read_csv(recovery / "data/test.csv")
    X = test.drop(columns=params["target_col"])
    assert [column.name for column in signature.inputs.inputs] == list(X.columns)
    predictions = loaded.predict(X)
    reference = joblib.load(recovery / "models/fitted_model.pkl").predict(X)
    assert np.array_equal(predictions, reference), "Registry and recovered model predictions differ"
    metrics = {"RMSE": float(np.sqrt(mean_squared_error(test.price, predictions))),
               "MAE": float(mean_absolute_error(test.price, predictions)), "R2": float(r2_score(test.price, predictions))}
    assert all(math.isclose(metrics[k], v, rel_tol=1e-12, abs_tol=1e-10) for k, v in HISTORICAL.items())
    result = {"tracking_uri": required("MLFLOW_TRACKING_URI"), "experiment_name": EXPERIMENT,
              "experiment_id": experiment.experiment_id, "run_id": run.info.run_id,
              "registered_model": MODEL_NAME, "version": str(number), "model_uri": uri,
              "version_source": version.source, "signature": signature.to_dict(),
              "logged_metrics": {k: run.data.metrics[k] for k in HISTORICAL}, "recomputed_metrics": metrics,
              "predictions_equal_recovery": True, "test_rows": len(test), "artifacts": artifacts,
              "params": run.data.params, "tags": run.data.tags, "artifact_uri": run.info.artifact_uri,
              "existing_baseline_reused": reused, "registered_versions": sorted(int(v.version) for v in versions(client))}
    (RUNTIME / "stage1_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def register(args):
    import joblib
    import mlflow
    import mlflow.catboost
    import pandas as pd
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    recovery = Path(args.recovery_dir or required("BASELINE_RECOVERY_DIR")).expanduser().resolve()
    params = validate_recovery(recovery)
    assert os.environ.get("MLFLOW_EXPERIMENT_NAME", EXPERIMENT) == EXPERIMENT
    assert os.environ.get("REGISTERED_MODEL_NAME", MODEL_NAME) == MODEL_NAME
    uri = required("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    current = versions(client)
    if current and not args.force_new_version:
        baseline = next((v for v in current if str(v.version) == "1"), None)
        if baseline is None or baseline.tags.get("baseline_origin") != "reconstructed":
            raise ValueError("Unexpected registry history; refusing to create a new version")
        return verify(client, 1, recovery, params, reused=True)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        client.create_experiment(EXPERIMENT, artifact_location=artifact_root().rstrip("/") + "/" + EXPERIMENT)
        experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment.lifecycle_stage != "active" or experiment.artifact_location.rstrip("/") != artifact_root().rstrip("/") + "/" + EXPERIMENT:
        raise ValueError("Unexpected experiment lifecycle/artifact root")
    train = pd.read_csv(recovery / "data/train.csv")
    X = train.drop(columns=params["target_col"])
    model = joblib.load(recovery / "models/fitted_model.pkl")
    assert list(X.columns) == model.feature_names_
    signature = infer_signature(X, model.predict(X))
    freeze = RUNTIME / "registration_environment.freeze.txt"
    freeze.write_text("\n".join(sorted(d.metadata["Name"] + "==" + d.version
                                      for d in importlib.metadata.distributions())) + "\n", encoding="utf-8")
    requirements = ["mlflow==2.7.1", "catboost==1.2.3", "numpy==1.26.2", "pandas==2.1.3",
                    "scikit-learn==1.4.1.post1", "scipy==1.12.0", "joblib==1.3.2"]
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="reconstructed_baseline_v1", tags=TAGS) as active:
        run_id = active.info.run_id
        mlflow.log_params(params["model"])
        mlflow.log_params({"target_col": params["target_col"], "cat_features": json.dumps(params["cat_features"]),
                           "drop_cols": json.dumps(params["drop_cols"]), "test_size": params["test_size"],
                           "random_state": params["random_state"], "train_rows": len(train), "test_rows": 28184,
                           "feature_count": len(X.columns)})
        mlflow.log_metrics(HISTORICAL)
        for name in ["params.yaml", "metrics.json", "provenance.json", "environment.freeze.txt", "data_hashes.json"]:
            mlflow.log_artifact(str(METADATA / name), artifact_path="baseline")
        mlflow.log_artifact(str(freeze), artifact_path="runtime")
        for name in ["RECOVERY_REPORT.md", "recovery_report.json", "initial_diagnostics.json"]:
            if (recovery / name).is_file():
                mlflow.log_artifact(str(recovery / name), artifact_path="recovery")
        for name in ["prepare.py", "fit.py", "evaluate.py", "cleaning.py"]:
            mlflow.log_artifact(str(recovery / "scripts" / name), artifact_path="source_training_code")
        mlflow.catboost.log_model(cb_model=model, artifact_path="model", signature=signature,
                                  input_example=X.head(5), pip_requirements=requirements)
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    if versions(client) and not args.force_new_version:
        raise ValueError("Registry changed concurrently; refusing a duplicate version")
    version = mlflow.register_model("runs:/" + run_id + "/model", MODEL_NAME)
    assert args.force_new_version or str(version.version) == "1", "Unexpected version: inspect registry"
    for k, v in TAGS.items():
        client.set_model_version_tag(MODEL_NAME, version.version, k, v)
    client.update_model_version(MODEL_NAME, version.version,
                               description="Reconstructed baseline: byte-identical data and metrics; pickle differs.")
    return verify(client, version.version, recovery, params, reused=False)


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", help="Read-only workspace; defaults to BASELINE_RECOVERY_DIR")
    parser.add_argument("--force-new-version", action="store_true", help="Intentionally register another baseline version")
    args = parser.parse_args()
    lock = RUNTIME / "baseline-registration.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Registration running, or a stale lock needs inspection")
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        return register(args)
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(redact(error), file=sys.stderr)
        sys.exit(1)
