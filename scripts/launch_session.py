# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Launch the steering session on HF Jobs.

    uv run scripts/launch_session.py rehearsal --repo USER/beam-steer-runs   # 1x rtx-pro-6000, tiny model, ~10 min
    uv run scripts/launch_session.py real --repo USER/beam-steer-runs [--hours 3]  # 8x rtx-pro-6000, real Beam

--repo is a private dataset repo (created private if missing). The real run downloads the sweep units from it and
uploads every phase's results to it as they finish, so a timeout loses at most the wave in flight. The job timeout
is the hard cost cap: hours x the flavor's hourly price. The session stops starting new waves 25 minutes before it.

Code ships inside the command; the HF token goes to the job as a secret (model download and result upload).
"""

import argparse
import base64
import io
import pathlib
import tarfile

from huggingface_hub import HfApi, get_token

IMAGE = "vllm/vllm-openai:nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9"
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "reflection/Beam-1.0-501B-A23B-FP8"
PRICE = {"rtx-pro-6000": 2.75, "rtx-pro-6000x8": 22.0}

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["rehearsal", "real"])
parser.add_argument("--repo", required=True)
parser.add_argument("--hours", type=float, default=3.0)
args = parser.parse_args()
api = HfApi()

files = {"steer/beam_steer.py": "steer/beam_steer.py", "steer/pyproject.toml": "steer/pyproject.toml",
         "scripts/session.py": "session.py", "scripts/vllm_tiny_check.py": "vllm_tiny_check.py",
         "meta/config.json": "out/real_config.json"}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for src, dst in files.items():
        tar.add(ROOT / src, arcname=dst)
payload = base64.b64encode(buf.getvalue()).decode()

api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(args.repo, repo_type="dataset").private, f"{args.repo} is not private"

setup = f"""set -euo pipefail
mkdir -p /w && cd /w && echo {payload} | base64 -d | tar xz
if python3 -m pip --version >/dev/null 2>&1; then PI="python3 -m pip install -q"; else PI="uv pip install --system -q"; fi
$PI ./steer
python3 -c "import pyarrow" 2>/dev/null || $PI pyarrow
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""
if args.mode == "rehearsal":
    flavor, hours = "rtx-pro-6000", 0.5
    script = setup + """REF=""
python3 -c "from transformers import BeamForCausalLM" 2>/dev/null || { $PI --no-deps --target /ref "transformers==5.21.0"; REF=/ref; }
PYTHONPATH=$REF python3 vllm_tiny_check.py ref out
python3 session.py --model out/ckpt_bf16 --tiny --out run --upload REPO --budget-min 20
""".replace("REPO", args.repo)
else:
    flavor, hours = "rtx-pro-6000x8", args.hours
    for name in ("units.npz", "units.jsonl"):
        api.upload_file(path_or_fileobj=str(ROOT / "data/sweep" / name), path_in_repo=f"inputs/{name}",
                        repo_id=args.repo, repo_type="dataset")
    script = setup + f"""python3 -c "from huggingface_hub import snapshot_download as s; s('{args.repo}', repo_type='dataset', allow_patterns=['inputs/*'], local_dir='in')"
python3 session.py --model {MODEL} --tp 8 --units in/inputs/units.npz --meta in/inputs/units.jsonl --out run \\
    --upload {args.repo} --budget-min {int(hours * 60 - 25)}
"""

minutes = int(hours * 60)
print(f"{flavor} for at most {minutes} min = ${PRICE[flavor] * hours:.2f} cap")
job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=flavor, timeout=f"{minutes}m",
                  secrets={"HF_TOKEN": get_token()},
                  env={"VLLM_USE_V2_MODEL_RUNNER": "0", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1",
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
                  name=f"beam-steer-{args.mode}")
print(job.id, job.url)
