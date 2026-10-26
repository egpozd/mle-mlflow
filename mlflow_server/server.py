"""MLflow 2.7.1: PostgreSQL backend with local or Yandex/S3 artifacts."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from common import artifact_root, backend_uri, configure, redact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "s3"), default=None)
    args = parser.parse_args()
    root = configure()
    mode = args.mode or os.environ.get("MLFLOW_STORAGE_MODE", "local")
    command = [sys.executable, "-B", "-m", "mlflow", "server",
               "--host", "127.0.0.1", "--port", "5000",
               "--backend-store-uri", backend_uri(),
               "--default-artifact-root", artifact_root(mode), "--no-serve-artifacts"]
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    return subprocess.call(command, cwd=root, env=environment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(redact(error), file=sys.stderr)
        sys.exit(1)
