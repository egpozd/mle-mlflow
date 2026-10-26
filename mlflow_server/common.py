"""Local configuration. Secret values are never printed."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "mle-mlflow-runtime"


def configure():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise ValueError("Invalid local .env entry")
            os.environ.setdefault(key, value.strip().strip("\"").strip("'"))
    (RUNTIME / "tmp").mkdir(parents=True, exist_ok=True)
    (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(RUNTIME / "tmp")
    os.environ["MPLCONFIGDIR"] = str(RUNTIME / "matplotlib")
    os.environ["NUMBA_CACHE_DIR"] = str(RUNTIME / "numba")
    return ROOT


def required(name):
    value = os.environ.get(name, "")
    if not value or value.startswith("<"):
        raise ValueError("Configure " + name + " in environment or local .env")
    return value


def redact(message):
    message = str(message)
    for key in ("MLFLOW_DB_PASSWORD", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[REDACTED]")
    return message


def backend_uri():
    from sqlalchemy.engine import URL
    if required("MLFLOW_DB_NAME") != "mlflow" or required("MLFLOW_DB_USER") != "mlflow":
        raise ValueError("Refusing backend outside dedicated mlflow database/user")
    return URL.create("postgresql+psycopg2", username="mlflow", password=required("MLFLOW_DB_PASSWORD"),
                      host=required("MLFLOW_DB_HOST"), port=int(required("MLFLOW_DB_PORT")),
                      database="mlflow").render_as_string(hide_password=False)


def artifact_root(mode=None):
    mode = mode or os.environ.get("MLFLOW_STORAGE_MODE", "local")
    if mode == "local":
        path = RUNTIME / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve().as_uri()
    if mode != "s3":
        raise ValueError("MLFLOW_STORAGE_MODE must be local or s3")
    uri = required("MLFLOW_ARTIFACT_ROOT").rstrip("/")
    if not uri.startswith("s3://") or not uri[5:]:
        raise ValueError("S3 mode requires an s3://bucket[/prefix] artifact root")
    for key in ("MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        required(key)
    return uri
