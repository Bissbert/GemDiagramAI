#!/bin/sh
# Run every measurement in docs/measurement.md inside a Linux container.
#
#   sh tools/linux-run.sh [prepared-data-dir] > media/captures/linux-run.txt
#
# prepared-data-dir defaults to ./training-data (git-ignored; 4,992 prepared
# diagrams). It is mounted read-only; without it the dataset measurement and
# the two contact sheets are skipped.
#
# requirements.txt names tensorflow-macos and tensorflow-metal, which have no
# Linux wheels, so the container installs every other line. It leaves numpy
# unpinned, which resolves to numpy 2 and breaks TensorFlow 2.14; the script
# shows that, then reinstalls with the constraint numpy<2.
# The repository is mounted read-only and copied; rendered PNGs are written
# back to media/.

set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${1:-$REPO/training-data}" 2>/dev/null && pwd || echo "")
IMAGE=python:3.11-slim-bookworm

docker pull -q "$IMAGE" >/dev/null

docker run --rm -v "$REPO":/repo:ro -v "$REPO/media":/out \
    ${DATA:+-v "$DATA":/data:ro} "$IMAGE" sh -c '
set -u
section() { printf "\n=== %s\n" "$*"; }

apt-get -qq update >/dev/null 2>&1
apt-get -qq install -y libcairo2 fonts-dejavu-core >/dev/null 2>&1
cp -r /repo /tmp/gd && cd /tmp/gd

section "environment"
uname -srm
python3 --version

section "pip install (requirements.txt without tensorflow-macos, tensorflow-metal)"
grep -vE "^tensorflow-(macos|metal)$" requirements.txt > /tmp/req.txt
pip install -q --disable-pip-version-check --root-user-action=ignore -r /tmp/req.txt >/tmp/pip.log 2>&1
echo "exit=$?"
pip install -q --disable-pip-version-check --root-user-action=ignore -r requirements.txt >/tmp/pip-full.log 2>&1
echo "unmodified requirements.txt: exit=$?"
grep -m1 -E "^ERROR" /tmp/pip-full.log
python3 -c "import numpy; print(\"numpy\", numpy.__version__)"
python3 -c "import tensorflow" >/tmp/tf.log 2>&1
echo "import tensorflow: exit=$?"
grep -m1 -E "^(ImportError|AttributeError)" /tmp/tf.log

section "pip install again with the constraint numpy<2"
echo "numpy<2" > /tmp/constraints.txt
pip install -q --disable-pip-version-check --root-user-action=ignore -r /tmp/req.txt -c /tmp/constraints.txt >/tmp/pip2.log 2>&1
echo "exit=$?"
python3 -c "import tensorflow as tf, numpy, PIL, cairosvg, matplotlib; print(\"tensorflow\", tf.__version__, \"| numpy\", numpy.__version__, \"| Pillow\", PIL.__version__, \"| CairoSVG\", cairosvg.__version__, \"| matplotlib\", matplotlib.__version__)" 2>/dev/null

section "tools/measure_config.py"
python3 tools/measure_config.py

if [ -d /data ]; then
    section "tools/measure_dataset.py --data-dir <prepared dataset>"
    python3 tools/measure_dataset.py --data-dir /data
fi

section "prepare_data_for_training.py on the three fixture SVGs"
printf "%s\n%s\n" "$PWD/tools/fixtures/svg" "$PWD/tools/fixtures/metadata.json" |
    python3 -W error::RuntimeWarning prepare_data_for_training.py --save_dir /tmp/fx >/tmp/prep.log 2>&1
echo "exit=$? (RuntimeWarning promoted to error)"
ls /tmp/fx | wc -l | xargs echo "npz files:"

section "tools/measure_dataset.py --data-dir <fixture output>"
python3 tools/measure_dataset.py --data-dir /tmp/fx

section "tools/measure_model.py"
python3 tools/measure_model.py

section "tools/check_training_wiring.py --batch-size 1"
python3 tools/check_training_wiring.py --batch-size 1

section "tools/check_inference_path.py"
python3 tools/check_inference_path.py

section "prepare_data_for_generation.py --save_dir <new dir>, from an empty working directory"
mkdir -p /tmp/empty && cd /tmp/empty
printf "0.5\n0.8\n" | PYTHONPATH=/tmp/gd python3 /tmp/gd/prepare_data_for_generation.py --save_dir /tmp/empty/output >/tmp/gen.log 2>&1
echo "exit=$?"
grep -m1 -E "Error" /tmp/gen.log
ls -A /tmp/empty | xargs echo "working directory now holds:"
cd /tmp/gd

section "tools/measure_training_step.py --epochs 1 --batch-size 1 --images random"
start=$(date +%s)
python3 tools/measure_training_step.py --epochs 1 --batch-size 1 --images random
echo "exit=$? wall=$(( $(date +%s) - start ))s"

section "media"
python3 tools/render_preprocessing.py
python3 tools/render_parameter_grid.py
[ -d /data ] && python3 tools/render_samples.py --data-dir /data
for f in media/*.png; do cp "$f" /out/; done
ls -l media/*.png | awk "{print \$5, \$9}"
'
