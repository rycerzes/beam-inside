# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Watch an HF Job: relay the session's progress lines and warn as its GPU spend crosses thresholds.

    uv run scripts/watch_job.py JOB_ID [--rate 11.0] [--expected 12.8] [--every 2.5] [--grep REGEX]

One line per event on stdout (for a Monitor): the job starting, each log line matching --grep (progress, aborts,
errors), a BUDGET line at $5, $10, the expected cost and then every --every dollars past it, and the final stage
with the spend. Spend is the runtime since the job started times --rate (the flavor's hourly price), which is what
HF bills for Jobs. Nothing is cancelled: stopping the job is the user's call.
"""

import argparse
import re
import threading
import time
from datetime import datetime, timezone

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("job")
parser.add_argument("--rate", type=float, default=11.0, help="USD per hour of the job's flavor")
parser.add_argument("--expected", type=float, default=12.8, help="the quoted cost; warnings past it get louder")
parser.add_argument("--every", type=float, default=2.5)
parser.add_argument("--grep", default=r"mmlu|sanity|loaded|ABORT|SESSION DONE|Traceback|Error|error|Killed|OOM")
args = parser.parse_args()
api = HfApi()
done = threading.Event()


def say(*a):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def logs():
    pattern = re.compile(args.grep)
    while not done.is_set():
        try:
            for line in api.fetch_job_logs(job_id=args.job, follow=True):
                if pattern.search(line):
                    say("LOG", line.strip()[:400])
            return
        except Exception as e:  # the stream can drop while the job is scheduling; retry
            time.sleep(15)


threading.Thread(target=logs, daemon=True).start()
thresholds = sorted({5.0, 10.0, args.expected})
started = None
while True:
    try:
        info = api.inspect_job(job_id=args.job)
    except Exception as e:
        say(f"status check failed ({type(e).__name__}); retrying")
        time.sleep(60)
        continue
    stage = info.status.stage
    if started is None and stage == "RUNNING":
        started = info.started_at or datetime.now(timezone.utc)
        say(f"RUNNING since {started:%H:%M:%S} UTC at ${args.rate:.2f}/h; expected about ${args.expected:.2f}")
    spent = (datetime.now(timezone.utc) - started).total_seconds() / 3600 * args.rate if started else 0.0
    while thresholds and spent >= thresholds[0]:
        t = thresholds.pop(0)
        over = t >= args.expected
        say(f"BUDGET {'WARNING ' if over else ''}spend ${spent:.2f} (crossed ${t:.2f}; expected ${args.expected:.2f})"
            + (" - over the quote, still running" if over else ""))
        if over:
            thresholds.append(t + args.every)
    if stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
        done.set()
        time.sleep(5)  # let the last log lines through
        say(f"FINAL {stage} {info.status.message or ''} spend ~${spent:.2f}")
        break
    time.sleep(60)
