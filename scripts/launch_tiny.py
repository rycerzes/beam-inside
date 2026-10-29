# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Launch the beam_steer tiny-model check on HF Jobs, shipping the files inside the command (nothing is uploaded).

    uv run scripts/launch_tiny.py cpu   # cpu-basic: image entrypoint, plugin import, reference build
    uv run scripts/launch_tiny.py gpu [PHASES]   # rtx-pro-6000: full check (BF16 eager / graphs, NVFP4 Marlin),
                                                 # capture, then a tiny session with PHASES (comma list)

No secrets are passed: the tiny model is built inside the job, so nothing gated is downloaded.
"""

import base64
import io
import pathlib
import sys
import tarfile

from huggingface_hub import HfApi

IMAGE = "vllm/vllm-openai:nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9"
ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = {"steer/beam_steer.py": "steer/beam_steer.py", "steer/pyproject.toml": "steer/pyproject.toml",
         "scripts/vllm_tiny_check.py": "vllm_tiny_check.py",
         "scripts/session.py": "session.py", "meta/config.json": "out/real_config.json"}
FLAVORS = {"cpu": ("cpu-basic", "20m"), "gpu": ("rtx-pro-6000", "30m")}

mode = sys.argv[1]
phases = sys.argv[2] if len(sys.argv) > 2 else "sanity,stats,calib,effort,sweep"
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for src, dst in FILES.items():
        tar.add(ROOT / src, arcname=dst)
payload = base64.b64encode(buf.getvalue()).decode()

script = f"""set -euo pipefail
mkdir -p /w && cd /w && echo {payload} | base64 -d | tar xz
if python3 -m pip --version >/dev/null 2>&1; then PI="python3 -m pip install -q"; else PI="uv pip install --system -q"; fi
python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
$PI ./steer
REF=""
if ! python3 -c "from transformers import BeamForCausalLM" 2>/dev/null; then
  $PI --no-deps --target /ref "transformers==5.21.0"; REF=/ref
fi
PYTHONPATH=$REF python3 vllm_tiny_check.py ref out
"""
if mode == "cpu":
    script += """python3 -c "import beam_steer; beam_steer.register(); print('beam_steer patched')" || echo "register failed on CPU"
echo PROBE DONE
"""
else:
    script += """nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
export VLLM_USE_V2_MODEL_RUNNER=0
for run in "bf16 --eager" "bf16" "nvfp4 --moe marlin"; do
  python3 vllm_tiny_check.py vllm out $run 2>&1 | awk '!/INFO/ || /NvFp4|Marlin|marlin|backend/' | tail -30 || true
done
PYTHONPATH=$REF python3 vllm_tiny_check.py teacher out
PYTHONPATH=$REF python3 vllm_tiny_check.py capture out || true
python3 vllm_tiny_check.py compare out || true
BEAM_STEER_CAPTURE_ROWS=256 python3 session.py --model out/ckpt_nvfp4 --tiny --out run --moe-backend marlin --budget-min 15 \\
    --max-model-len 1024 --effort-questions 12 --effort-gen-questions 6 --phases PHASES 2>&1 | grep -v "INFO" | tail -30
""".replace("PHASES", phases)

flavor, timeout = FLAVORS[mode]
job = HfApi().run_job(image=IMAGE, command=["bash", "-c", script], flavor=flavor, timeout=timeout,
                      name=f"beam-steer-tiny-{mode}")
print(job.id, job.url)
