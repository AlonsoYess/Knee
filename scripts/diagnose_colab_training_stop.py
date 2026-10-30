"""Read-only stop evidence from the parent Colab kernel; never train or resume.

Run in the SAME kernel immediately after the exception and before restarting.
Return codes and memory counters are evidence, not a diagnosis of an OOM kill.
No private Drive file, participant record, checkpoint or outcome is opened.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

VERSION = "colab_training_stop_diagnostic_v1.0"
_MEMORY_NAMES = ("memory.current", "memory.max", "memory.peak", "memory.events", "memory.events.local")


def _text(path, limit=16384):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = stream.read(limit + 1)
        return value.strip() if len(value) <= limit else None
    except (OSError, UnicodeError):
        return None


def _returncode(traceback):
    seen = set()
    while traceback is not None and id(traceback) not in seen:
        seen.add(id(traceback))
        frame = traceback.tb_frame
        if frame.f_code.co_name == "training_run":
            code = frame.f_locals.get("status")
            if type(code) is not int:
                code = getattr(frame.f_locals.get("process"), "returncode", None)
            if type(code) is int:
                return code
        traceback = traceback.tb_next
    return None


def collect(traceback=None):
    """Collect bounded operating-system counters, without touching training."""
    code = _returncode(traceback)
    signal_name = None
    if code is not None and code < 0:
        try:
            signal_name = signal.Signals(-code).name
        except ValueError:
            pass
    meminfo = _text("/proc/meminfo")
    memory = {}
    for line in (meminfo or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"}:
            try:
                memory[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                pass
    cgroup = {name: _text(Path("/sys/fs/cgroup") / name) for name in _MEMORY_NAMES}
    try:
        free = shutil.disk_usage("/content").free
    except OSError:
        free = None
    gpu = None
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        if process.returncode == 0 and len(process.stdout) <= 4096:
            gpu = process.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"version": VERSION, "traceback_available": traceback is not None,
        "training_subprocess_returncode": code, "termination_signal": signal_name,
        "termination_cause_confirmed": False,
        "note": "Memory counters are cumulative; low current usage after termination does not exclude an earlier peak. A signal alone does not prove the cause.",
        "kernel_python": sys.version.split()[0], "system_memory_kib": memory,
        "cgroup_memory_counters": cgroup, "content_free_bytes": free, "gpu_memory_readout": gpu,
        "training_executed": False, "training_resumed": False, "files_modified": False,
        "participant_data_opened": False, "reserved_test_opened": False}


def last_traceback():
    value = getattr(sys, "last_traceback", None)
    for name in ("last_exc", "last_value"):
        if value is None:
            value = getattr(getattr(sys, name, None), "__traceback__", None)
    return value


if __name__ == "__main__":
    print(json.dumps(collect(last_traceback()), ensure_ascii=False, indent=2))
