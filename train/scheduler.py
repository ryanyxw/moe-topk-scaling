#!/usr/bin/env python3
"""Tiny GPU job scheduler for the scaling sweep (runs on the host, detached).

Queue: train/queue.txt (or $QUEUE), re-read every loop, one job per line:
    <name> <pair|single> <train.py args...>
Lines starting with '#' are ignored; append lines at any time. A job is skipped if its status
file says OK. Slots: pairs (0,2) (3,4) (5,6) and single (7).
A job with `--parent NAME` waits until NAME has status OK; `--after NAME` (stripped before launch)
waits until NAME has started, e.g. to keep a pair job from taking GPUs a queued quad job needs.

Failure handling: a run that dies with a hang/rank failure is retried (up to MAX_RETRY) by
resuming from its own rolling checkpoint; OOM is not retried. On OK, the rolling
non-persistent checkpoint is deleted so later extensions load only the persistent ones.
"""

import os
import shutil
import subprocess
import time


ROOT = os.environ.get("SCALING_ROOT", "/mnt/nvme/scaling")  # runs/ and logs/
HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.environ.get("QUEUE", f"{HERE}/queue.txt")
SLOTS = {"pair": ["0,2", "3,4", "5,6"], "single": ["7"], "quad": ["0,2,3,4"], "six": ["0,2,3,4,5,6"]}
PORTS = {"0,2": 29600, "3,4": 29610, "5,6": 29620, "7": 29630, "0,2,3,4": 29640, "0,2,3,4,5,6": 29660}
CPUSET = os.environ.get("CPUSET", "")
MAX_RETRY = 5  # the 4-day final run must survive several transient failures


def log(msg):
    with open(f"{ROOT}/logs/scheduler.log", "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def status(name):
    try:
        return open(f"{ROOT}/logs/{name}.status").read().split()[0]
    except (FileNotFoundError, IndexError):
        return None


def gpus_idle(gpus):
    """True if every GPU in the slot has <2 GB in use (guards against jobs started outside the scheduler)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    used = {int(i): int(m) for i, m in (l.split(",") for l in out.strip().splitlines())}
    return all(used.get(int(g), 0) < 2000 for g in gpus.split(","))


def read_queue():
    jobs = []
    if not os.path.exists(QUEUE):
        return jobs
    for line in open(QUEUE):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, slot, *args = line.split()
        jobs.append((name, slot, args))
    return jobs


def launch(name, gpus, args, attempt):
    run_args = list(args)
    if "--after" in run_args:
        i = run_args.index("--after")
        del run_args[i:i + 2]
    ckpt = f"{ROOT}/runs/{name}/ckpt"
    if attempt > 0 and os.path.isdir(ckpt):
        # Resume this run from its newest checkpoint (rolling or persistent). Drop any --load/--load-step
        # pointing at a parent run (train.py keeps --parent only for the warmup): the run's own state supersedes it.
        cleaned, skip = [], 0
        for a in run_args:
            if skip:
                skip -= 1
                continue
            if a in ("--load", "--load-step"):
                skip = 1
                continue
            cleaned.append(a)
        run_args = cleaned + ["--load", ckpt]
    logname = name if attempt == 0 else f"{name}.retry{attempt}"
    env = dict(os.environ, GPUS=gpus, PORT=str(PORTS[gpus]), CPUSET=CPUSET)
    cmd = [f"{HERE}/launch.sh", logname, "--name", name, *run_args]
    log(f"START {logname} on GPUs {gpus}: {' '.join(run_args)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Adopted:
    """A job launched by a previous scheduler instance: finished once its status leaves RUNNING."""

    def __init__(self, logname):
        self.logname = logname

    def poll(self):
        return None if status(self.logname) == "RUNNING" else 0


def adopt(jobs):
    """Re-attach RUNNING jobs from the queue (GPUs from their last START line) after a scheduler restart."""
    running, attempts = {}, {}
    starts = {}
    if os.path.exists(f"{ROOT}/logs/scheduler.log"):
        for line in open(f"{ROOT}/logs/scheduler.log"):
            parts = line.split()
            if len(parts) > 6 and parts[2] == "START" and parts[4:6] == ["on", "GPUs"]:
                starts[parts[3]] = parts[6].rstrip(":")
    for name, _slot, args in jobs:
        for attempt in range(MAX_RETRY, -1, -1):
            logname = name if attempt == 0 else f"{name}.retry{attempt}"
            if status(logname) == "RUNNING" and logname in starts:
                running[starts[logname]] = (name, Adopted(logname), args, attempt)
                attempts[name] = attempt
                log(f"ADOPT {logname} on GPUs {starts[logname]}")
                break
    return running, attempts


def main():
    log("scheduler started")
    running, attempts = adopt(read_queue())  # gpus -> (name, proc, args, attempt)
    while True:
        # Reap finished jobs.
        for gpus, (name, proc, args, attempt) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[gpus]
            logname = name if attempt == 0 else f"{name}.retry{attempt}"
            st = status(logname)
            log(f"END {logname}: {st}")
            if st == "OK":
                if logname != name:  # mirror final status onto the base name
                    shutil.copy(f"{ROOT}/logs/{logname}.status", f"{ROOT}/logs/{name}.status")
                # Checkpoints are written by root inside the container; remove them with sudo.
                subprocess.run(["sudo", "rm", "-rf", f"{ROOT}/runs/{name}/ckpt/non_persistent"])
            elif st and st.startswith("OOM"):
                log(f"GIVE UP {name}: OOM")
            elif attempt < MAX_RETRY:
                attempts[name] = attempt + 1
                log(f"RETRY {name} (attempt {attempt + 1})")
            else:
                log(f"GIVE UP {name} after {attempt + 1} attempts")
        # Launch new jobs into free slots, in queue order.
        busy = {n for n, *_ in running.values()}
        for name, slot, args in read_queue():
            if name in busy or status(name) == "OK":
                continue
            if "--parent" in args and status(args[args.index("--parent") + 1]) != "OK":
                continue  # extension: wait for the parent run to finish
            if "--after" in args and status(args[args.index("--after") + 1]) is None:
                continue  # ordering: wait until that job has started
            attempt = attempts.get(name, 0)
            st = status(name if attempt == 0 else f"{name}.retry{attempt}")
            if attempt == 0 and st is not None:  # already ran (failed/OOM/running elsewhere): leave it
                continue
            if attempt > 0 and st is not None:
                continue
            used = {g for key in running for g in key.split(",")}
            free = [g for g in SLOTS.get(slot, []) if not used & set(g.split(",")) and gpus_idle(g)]
            if not free:
                continue
            running[free[0]] = (name, launch(name, free[0], args, attempt), args, attempt)
            busy.add(name)
        if os.path.exists(f"{ROOT}/STOP_SCHEDULER") and not running:
            log("scheduler stopping")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
