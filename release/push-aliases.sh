#!/usr/bin/env bash

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# fail if build command fails
set -ex

ROOT_PACK_DIR="${SCRIPT_DIR}/pypi-aliases/papermage" \
    bash ${SCRIPT_DIR}/push-to-pypi.sh

ROOT_PACK_DIR="${SCRIPT_DIR}/pypi-aliases/scipdf" \
    bash ${SCRIPT_DIR}/push-to-pypi.sh
