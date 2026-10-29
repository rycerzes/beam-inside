# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Launch the steering session on HF Jobs.

    uv run scripts/launch_session.py rehearsal --repo USER/beam-steer-runs   # 1x rtx-pro-6000, tiny model, ~10 min
    uv run scripts/launch_session.py real --repo USER/beam-steer-runs [--variant nvfp4|fp8] [--hours 3]

Variants: nvfp4 = the default NVFP4 checkpoint on 4x rtx-pro-6000, routed experts through Marlin (FP4 weights, BF16
activations); fp8 = the FP8 checkpoint on 8x rtx-pro-6000 with the Triton MoE backend, as in the vLLM recipe.

--repo is a private dataset repo (created private if missing). The real run downloads the sweep units from it and
uploads every phase's results to it as they finish, so a timeout loses at most the wave in flight. With --hours,
the job timeout is a hard cost cap (hours x the flavor's hourly price) and the session stops starting new waves 25
minutes before it. Without it there is no cap: HF Jobs would apply its 30-minute default, so the timeout is set to
one day and the session never stops early; watch spend with scripts/watch_job.py.

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
VARIANTS = {"nvfp4": ("reflection/Beam-1.0-501B-A23B-NVFP4", "rtx-pro-6000x4", 4, "marlin"),
            "fp8": ("reflection/Beam-1.0-501B-A23B-FP8", "rtx-pro-6000x8", 8, "triton")}
PRICE = {"rtx-pro-6000": 2.75, "rtx-pro-6000x4": 11.0, "rtx-pro-6000x8": 22.0}

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["rehearsal", "real"])
parser.add_argument("--repo", required=True)
parser.add_argument("--hours", type=float, help="hard cap; omit for no cap (1-day timeout, watch_job.py warnings)")
parser.add_argument("--variant", choices=list(VARIANTS), default="nvfp4")
parser.add_argument("--phases", default="sanity,stats,calib,effort,sweep")
parser.add_argument("--session-args", default="", help="extra session.py flags, e.g. '--max-model-len 16384'")
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
python3 session.py --model out/ckpt_bf16 --tiny --out run --upload REPO --budget-min 20 --max-model-len 1024 \\
    --effort-questions 12 --effort-gen-questions 6 --effort2-problems 6 --phases PHASES EXTRA
""".replace("REPO", args.repo).replace("PHASES", args.phases).replace("EXTRA", args.session_args)
    capture_rows = 256
else:
    model, flavor, tp, moe = VARIANTS[args.variant]
    hours = args.hours or 24.0
    budget = int(hours * 60 - 25) if args.hours else 10**6
    for src in ("data/sweep/units.npz", "data/sweep/units.jsonl", "data/effort2/inputs.pt"):
        if "sweep" in args.phases or "effort2" in src:
            api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=f"inputs/{pathlib.Path(src).name}",
                            repo_id=args.repo, repo_type="dataset")
    mmlu = ""
    if "mmlu" in args.phases.split(","):  # scripts/mmlu_arms.py
        api.upload_file(path_or_fileobj=str(ROOT / "data/mmlu/arms.json"), path_in_repo="inputs/mmlu_arms.json",
                        repo_id=args.repo, repo_type="dataset")
        mmlu = "--mmlu-arms in/inputs/mmlu_arms.json"
    script = setup + f"""python3 -c "from huggingface_hub import snapshot_download as s; s('{args.repo}', repo_type='dataset', allow_patterns=['inputs/*'], local_dir='in')"
python3 session.py --model {model} --tp {tp} --moe-backend {moe} --units in/inputs/units.npz --meta in/inputs/units.jsonl --out run \\
    --upload {args.repo} --budget-min {budget} --phases {args.phases} \\
    --effort2-inputs in/inputs/inputs.pt {mmlu} {args.session_args}
"""
    capture_rows = 4096  # ~2.4 GB per GPU at Beam's size: 91 questions x 9 efforts x 5 positions per dump

minutes = int(hours * 60)
if args.mode == "real" and not args.hours:
    print(f"{flavor} at ${PRICE[flavor]:.2f}/h, NO cost cap (1-day timeout); watch it with scripts/watch_job.py")
else:
    print(f"{flavor} for at most {minutes} min = ${PRICE[flavor] * hours:.2f} cap")
job = api.run_job(image=IMAGE, command=["bash", "-c", script], flavor=flavor, timeout=f"{minutes}m",
                  secrets={"HF_TOKEN": get_token()},
                  env={"VLLM_USE_V2_MODEL_RUNNER": "0", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1",
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "BEAM_STEER_CAPTURE_ROWS": str(capture_rows)},
                  name=f"beam-steer-{args.mode}" + (f"-{args.variant}" if args.mode == "real" else ""))
print(job.id, job.url)
