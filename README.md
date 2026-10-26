# Real estate model improvement - Stage 1

MLflow **2.7.1** uses PostgreSQL for Tracking and Model Registry. Experiment:
`real_estate_model_improvement`; registered model: `real_estate_price_model`.
Version **1** is the reconstructed baseline. Stages 2-5 are not implemented here.

## LOCAL DEVELOPMENT — VERIFIED

Local health, PostgreSQL backend, filesystem artifacts, Registry Version 1,
model loading and all three historical metrics were verified on 2026-10-26.

Docker Compose in `../mle-local-infra` retains PostgreSQL and Airflow. MinIO is
removed from Compose: after the out-of-disk incident, local artifacts use the
filesystem to avoid additional images, source builds and storage overhead.
Existing Docker volumes and the `flats`/`buildings` tables are preserved.

- Backend and Registry: dedicated PostgreSQL database/user `mlflow`, host
  `127.0.0.1`, port `5433`. SQLite is not used.
- Tracking URI: `http://127.0.0.1:5000` (loopback only).
- Artifact root: the file URI of `../mle-mlflow-runtime/artifacts`, computed
  from the checkout location with `Path.resolve().as_uri()`; no username or
  machine-specific absolute path is required in source configuration.
- Runtime logs, temporary downloads and verification evidence:
  `../mle-mlflow-runtime`, outside Git.
- The existing Python 3.10 `.venv` is reused. No dependency reinstall is needed.

Keep the existing `.env`; for a fresh setup copy `.env_template` and fill the
database password and recovery path locally. Do not overwrite working secrets.
`MLFLOW_STORAGE_MODE=local` selects the filesystem for registration. The local
launcher explicitly selects local mode even if S3 variables exist in `.env`.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\mlflow_server\run_server_local.ps1
# Compatibility alias: .\mlflow_server\run_server.ps1
# In another terminal:
Invoke-RestMethod http://127.0.0.1:5000/health
.\.venv\Scripts\python.exe -B mlflow_server/register_baseline.py
# Read-only source integrity check against the saved content hashes:
.\.venv\Scripts\python.exe -B mlflow_server/provision_postgres.py --verify-only
```

The existing PostgreSQL migrations are reused. Do not run provisioning or schema
upgrades merely to restart the server. The launchers above run in the foreground.
The background launch performed during local setup saved its launcher PID in
the external runtime's `mlflow_server.pid` and logs under `logs/`. Stop only
the relevant MLflow processes when switching modes. Do not remove Docker volumes.

Direct filesystem artifacts (`--no-serve-artifacts`) require the client to run on
this Windows machine with access to the same paths. This configuration is for local
development. Experiment artifact locations persist in PostgreSQL; changing the
server default does not relocate existing experiments or registered versions.
The old unused `Default` experiment may still reference the former S3 root; use
the named project experiment, whose location is checked by the registration script.

## YANDEX / S3 FINAL RUN — PREPARED, NOT PRODUCTION-TESTED

**Not yet verified with production credentials.** This is a prepared final-run
configuration, not evidence of a successful Yandex/S3 deployment.

`mlflow_server/run_server.sh` explicitly selects S3 mode and uses environment
variables (or an untracked local `.env`); it embeds no credentials.

Configure the production PostgreSQL connection through `MLFLOW_DB_HOST`,
`MLFLOW_DB_PORT`, `MLFLOW_DB_NAME=mlflow`, `MLFLOW_DB_USER=mlflow`, and
`MLFLOW_DB_PASSWORD`. Configure Object Storage with:

```sh
export MLFLOW_STORAGE_MODE=s3
export MLFLOW_ARTIFACT_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX
export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_DEFAULT_REGION=ru-central1
# Supply AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY through your secret environment.
# Set MLFLOW_DB_* and BASELINE_RECOVERY_DIR in that environment as well.
sh mlflow_server/run_server.sh
# Run the client with the same S3 variables and the chosen tracking URI:
.venv/bin/python -B mlflow_server/register_baseline.py
```

Create the bucket and grant the service account object read/write permissions
before the final run. Both client and server need access to PostgreSQL/S3 as
appropriate; artifact access is direct, without an MLflow artifact proxy. The
server binds to loopback; remote deployment needs an authenticated TLS reverse
proxy or SSH tunnel. `PYTHON` can select another compatible interpreter.

Use a **separate production PostgreSQL database instance** with the dedicated
`mlflow` database/user for the final run, then register its Version 1 there.
Reusing the local database does not migrate its filesystem artifacts to S3.
The script refuses a mismatched experiment location when creating a run and
reuses an already registered Version 1 by default. Production S3 has not been
executed or validated with real credentials during local recovery.

## Reconstructed baseline and verification

Registration reads `../mle-baseline-recovery` without changing it or retraining.
Before loading the trusted recovered pickle it validates data/model sizes and MD5.
It records model/split parameters, exact historical metrics, provenance, hashes,
tags, recovery reports, original training code, recovery/current environments,
CatBoost model, `infer_signature` signature, five-row input example, and model
requirements/conda metadata. CSV datasets and the pickle are not added to Git.

| Metric | Exact baseline value |
|---|---:|
| RMSE | 10547270.062205384 |
| MAE | 3744622.8052026215 |
| R2 | 0.8948994777286887 |

| File | Historical MD5 |
|---|---|
| initial_data.csv | 84045cd0f0e5a6f7a36eb4e2320fa41f |
| train.csv | 2d4bb3ba2091eb269bc56a93f05f9119 |
| test.csv | 1d5874c7f4950c457e1ff2678d44a121 |
| metrics.json | c76d1f590d0cafcb94f4078b26faca6f |
| fitted_model.pkl | ee7aa528bbdcdc207747e7ddc1194182 |

The reconstructed pickle MD5 is `bd7cc21e56d9b904832d7c3ed2878e82`.
Data and metrics match historical files byte for byte, while the model pickle
differs. Historical individual predictions were unavailable; the recovered model
reproduces historical aggregate metrics. Source Git SHA:
`e2decc886f6a9a5d633ff6e710ba2a74cf2db6a2`.

The registration script checks experiment/run linkage, metrics, parameters, tags,
artifacts and signature, loads `models:/real_estate_price_model/1`, and recomputes
RMSE/MAE/R2 on all **28184** historical test rows. Predictions must equal those
of the recovered model; metric tolerance is `rel_tol=1e-12`, `abs_tol=1e-10`.
Evidence is saved to `../mle-mlflow-runtime/stage1_verification.json`.

Repeated normal runs verify/reuse Version 1 without creating Version 2 or another
run. An exclusive local registration lock prevents overlapping local invocations;
there is no distributed lock. `--force-new-version` is an explicit opt-in and is
unnecessary for Stage 1.

`model_improvement/baseline/` contains only small reproducibility metadata.
Root `run_mlflow_server*.sh` and `test_mlflow_*.ipynb` are legacy exercises; use
the launchers in `mlflow_server/`. Obsolete MinIO build recipes were removed;
no MinIO source download or Docker build is needed.

`.gitignore` excludes `.env`, `.venv`, runtime artifacts, datasets, pickle/native
models, SQLite databases and notebook checkpoints. No setup script stages,
commits or pushes files.

`requirements.txt` pins the verified Python 3.10 environment: direct baseline
dependencies plus selected MLflow server/serialization compatibility dependencies.
It is not a complete transitive lock file. Existing notebook/research dependencies
(JupyterLab, Optuna, autofeat, mlxtend and plotting/widgets) remain for later stages.
`psycopg2-binary` replaces the unused psycopg 3 requirements for the configured
`postgresql+psycopg2` backend; setuptools is pinned because MLflow 2.7.1 imports
`pkg_resources`. The custom .env loader does not need python-dotenv; wheel is not a runtime dependency.
