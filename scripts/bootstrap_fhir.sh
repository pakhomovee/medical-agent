#!/usr/bin/env bash
# Get the MedAgentBench FHIR server running from nothing. Idempotent: safe to re-run,
# skips whatever is already done.
#
#   scripts/bootstrap_fhir.sh [OUT_DIR]        default: ./fhir
#
# No Docker anywhere in here -- Colab, AutoDL and most managed GPU hosts don't allow it.
# The image is a Spring Boot HAPI FHIR war plus an H2 database, so a JVM is enough.
set -euo pipefail

OUT="${1:-$(pwd)/fhir}"
PY="${PYTHON:-$(command -v python3)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> java"
if ! command -v java >/dev/null 2>&1; then
  apt-get install -y -q openjdk-17-jre-headless >/dev/null 2>&1 \
    || { echo "install openjdk-17-jre-headless first" >&2; exit 1; }
fi
java -version 2>&1 | head -1

echo "==> FHIR server files"
if [ -f "$OUT/rootfs/app/main.war" ]; then
  echo "    already extracted at $OUT"
else
  "$PY" "$HERE/scripts/fetch_fhir_server.py" --out "$OUT"
fi

if curl -sf --max-time 3 http://localhost:8080/fhir/metadata >/dev/null 2>&1; then
  echo "==> already serving on :8080"
else
  echo "==> starting (detached; ~60-90s to load the database)"
  setsid nohup "$OUT/run.sh" > "$OUT/server.log" 2>&1 < /dev/null &
  for i in $(seq 1 60); do
    curl -sf --max-time 3 http://localhost:8080/fhir/metadata >/dev/null 2>&1 && break
    sleep 5
  done
fi

if ! curl -sf --max-time 5 http://localhost:8080/fhir/metadata >/dev/null 2>&1; then
  echo "server did not come up; last log lines:" >&2
  tail -20 "$OUT/server.log" >&2
  exit 1
fi

echo "==> up"
grep -m1 "Started Application" "$OUT/server.log" || true
"$PY" "$HERE/scripts/g0_environment.py" --fhir http://localhost:8080/fhir --skip-model
