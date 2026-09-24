#!/bin/sh
# Run the test suite in a Linux container:
#
#   sh tests/docker.sh [pytest args]
#
# Installs requirements.txt unmodified, then runs pytest. The repository is
# mounted read-only and copied, so nothing is written back.

set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=python:3.11-slim-bookworm

docker run --rm -v "$REPO":/repo:ro "$IMAGE" sh -c '
set -e
apt-get -qq update >/dev/null 2>&1
apt-get -qq install -y libcairo2 fonts-dejavu-core >/dev/null 2>&1
cp -r /repo /tmp/gd && cd /tmp/gd
pip install -q --disable-pip-version-check --root-user-action=ignore -r requirements.txt pytest
python -m pytest -p no:cacheprovider "$@"
' sh "$@"
