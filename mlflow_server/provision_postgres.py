"""Create only the dedicated mlflow role/database; preserve source data.

Administrative commands connect to maintenance DB postgres, never mle_db/airflow.
Reads of source tables run inside an explicitly READ ONLY transaction.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from common import RUNTIME, configure, redact, required

EXPECTED_COUNTS = {"flats": 141362, "buildings": 24620}


def psql(database, sql):
    if database not in ("postgres", "mle_db"):
        raise ValueError("Unexpected administrative database")
    result = subprocess.run(["docker", "exec", "-i", "mle-postgres", "psql", "-X", "-q", "-A", "-t",
                             "-v", "ON_ERROR_STOP=1", "-U", "mle_user", "-d", database],
                            input=sql, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(redact(result.stderr))
    return result.stdout.strip()


def snapshot():
    databases = psql("postgres", "SELECT json_agg(datname ORDER BY datname) FROM pg_database WHERE datname IN ('mle_db','airflow','mlflow');")
    counts = {}
    for table in EXPECTED_COUNTS:
        sql = ("BEGIN READ ONLY; SELECT json_build_object('count',count(*),'content_md5',"
               "md5(string_agg(row_to_json(t)::text,'' ORDER BY id))) FROM public." + table + " t; COMMIT;")
        counts[table] = json.loads(psql("mle_db", sql))
    result = {"databases": json.loads(databases), "tables": counts}
    assert {key: entry["count"] for key, entry in counts.items()} == EXPECTED_COUNTS
    assert {"mle_db", "airflow"}.issubset(result["databases"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    configure()
    before_file = RUNTIME / "postgres_before.json"
    before = snapshot()
    if not before_file.exists():
        before_file.write_text(json.dumps(before, indent=2), encoding="utf-8")
    saved = json.loads(before_file.read_text())
    assert before["tables"] == saved["tables"], "Source tables differ from pre-provision snapshot"
    if not args.verify_only:
        # Fail closed if an existing dedicated database has an unexpected owner.
        owner = psql("postgres", "SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='mlflow';")
        if owner and owner != "mlflow":
            raise RuntimeError("Existing mlflow database has another owner; no changes made")
        password = required("MLFLOW_DB_PASSWORD")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", password):
            raise ValueError("Local provisioning expects a URL-safe generated password")
        # This SQL only creates the new mlflow role. It never alters an existing role.
        sql = """DO $$ BEGIN
          IF current_database() <> 'postgres' THEN RAISE EXCEPTION 'Wrong maintenance database'; END IF;
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='mlflow') THEN
            CREATE ROLE mlflow LOGIN PASSWORD '""" + password + """';
          END IF;
        END $$;
        SELECT 'CREATE DATABASE mlflow OWNER mlflow'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='mlflow')
        \\gexec
        """
        psql("postgres", sql)
    after = snapshot()
    assert "mlflow" in after["databases"], "mlflow database missing"
    assert saved["tables"] == after["tables"], "Source table content changed"
    (RUNTIME / "postgres_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    print(json.dumps({"dedicated_database": "mlflow", "source_tables_unchanged": True, **after}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(redact(error), file=sys.stderr)
        sys.exit(1)
