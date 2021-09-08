#!/usr/bin/env bash

set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export TTY_DEVICE=0

echo "=====RUNNING TYPE CHECK====="
make type-check

echo "=====RUNNING FORMAT CHECK====="
make format-check

echo "=====RUNNING UNIT TESTS====="
make unit

echo "=====RUNNING INTEGRATION TESTS====="
cleanup() {
  trap - 0
  set +e
  make kill-server
}
trap cleanup 0

make background-serve
make integration-tests
