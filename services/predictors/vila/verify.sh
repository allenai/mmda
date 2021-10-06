#!/usr/bin/env bash

set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PROJECT_NAME=$(cat PROJECT_NAME.txt)
export TAG="${BUILD_NUMBER:-$(date +%s)}"
export TTY_DEVICE=0

echo "=====RUNNING TYPE CHECK====="
make type-check

echo "=====RUNNING FORMAT CHECK====="
make format-check

  echo "=====RUNNING INTEGRATION TESTS====="
cleanup() {
  trap - 0
  set +e
  make kill-server
}
trap cleanup 0

make serve-background
make integration-tests
