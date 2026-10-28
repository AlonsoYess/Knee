"""Observe the frozen Colab child without modifying its training or checkpoints.

Only operational evidence is written under the supplied private records folder.
An unfinished receipt is evidence of a missing finalization, never proof of OOM.
The helper cannot restart a disconnected Colab VM or its notebook kernel.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

VERSION = "colab_training_supervisor_v1.0"
LOG_FOLDER = "gradual_runtime_logs"
MEMORY_RESERVE = 768 * 1024 ** 2
DISK_RESERVE = 2 * 1024 ** 3
MAX_STREAM_BYTES = 8 * 1024 ** 2
MAX_LINE_BYTES = 64 * 1024
LEASE_SECONDS = 120


def sanitize(text):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text))
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
                  r"\1<redacted>", text)
    return re.sub(r"(?i)((?:token|password|secret|api[_-]?key)\s*[:=]\s*)[^\s,;]+",
                  r"\1<redacted>", text)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path, limit=16384):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = stream.read(limit + 1)
        return value.strip() if len(value) <= limit else None
    except (OSError, UnicodeError):
        return None


def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing_" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        _sync(stream)
    os.replace(temporary, path)


def _sync(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as error:
        # DriveFS/FUSE can reject fsync while still supporting flush and replace.
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise


def _json(path):
    value = _read(path, 128 * 1024)
    if value is None:
        return None
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else None
    except ValueError:
        return None


def _identity(pid):
    raw = _read(Path("/proc") / str(pid) / "stat")
    fields = raw.rpartition(")")[2].split() if raw else []
    return {"pid": pid, "boot_id": _read("/proc/sys/kernel/random/boot_id"),
            "start_ticks": fields[19] if len(fields) > 19 else None}


def _alive(identity):
    if not isinstance(identity, dict) or not identity.get("boot_id") or not identity.get("start_ticks"):
        return None
    current = _identity(identity["pid"])
    if current["boot_id"] != identity["boot_id"]:
        return None
    return current["start_ticks"] == identity["start_ticks"]


def _status(pid):
    result = {}
    for line in (_read(Path("/proc") / str(pid) / "status") or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("VmRSS:", "VmHWM:", "PPid:"):
            try:
                result[parts[0].rstrip(":")] = int(parts[1]) * (1024 if parts[0] != "PPid:" else 1)
            except ValueError:
                pass
    return result


def _cgroup(pid):
    for line in (_read(Path("/proc") / str(pid) / "cgroup") or "").splitlines():
        if line.startswith("0::"):
            relative = line[3:].lstrip("/")
            if ".." not in Path(relative).parts:
                return Path("/sys/fs/cgroup") / relative
    return None


def _refresh_cgroup(sample, group):
    sample["cgroup_scope"] = str(group)
    for filename, key in (("memory.current", "cgroup_memory_current"),
                          ("memory.max", "cgroup_memory_max"), ("memory.peak", "cgroup_memory_peak")):
        text = _read(group / filename)
        sample[key] = int(text) if text and text.isdigit() else text
    for line in (_read(group / "memory.events") or "").splitlines():
        pair = line.split()
        if len(pair) == 2 and pair[0] in ("oom", "oom_kill") and pair[1].isdigit():
            sample["cgroup_" + pair[0]] = int(pair[1])


def sample_resources(child_pid=None):
    """Bounded, best-effort OS sampling; no model or participant file is read."""
    sample = {"observed_at_utc": _now(), "mem_available_bytes": None,
              "mem_total_bytes": None, "swap_free_bytes": None,
              "parent_rss_bytes": _status(os.getpid()).get("VmRSS"),
              "child_rss_bytes": None, "child_peak_rss_bytes": None,
              "process_tree_rss_bytes": None, "disk_free_bytes": None,
              "cgroup_scope": None, "cgroup_memory_current": None,
              "cgroup_memory_max": None, "cgroup_memory_peak": None,
              "cgroup_oom_kill": None, "cgroup_oom": None, "gpu": None}
    for line in (_read("/proc/meminfo") or "").splitlines():
        parts = line.split()
        key = {"MemAvailable:": "mem_available_bytes", "MemTotal:": "mem_total_bytes",
               "SwapFree:": "swap_free_bytes"}.get(parts[0] if parts else "")
        if key and len(parts) > 1:
            try:
                sample[key] = int(parts[1]) * 1024
            except ValueError:
                pass
    pid = child_pid or os.getpid()
    status = _status(pid)
    if child_pid:
        sample.update(child_rss_bytes=status.get("VmRSS"), child_peak_rss_bytes=status.get("VmHWM"))
    try:
        processes = {int(p.name): _status(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()}
        family = {pid}
        for _ in range(16):
            expanded = family | {p for p, s in processes.items() if s.get("PPid") in family}
            if expanded == family:
                break
            family = expanded
        sample["process_tree_rss_bytes"] = sum(processes[p].get("VmRSS", 0) for p in family if p in processes)
    except OSError:
        pass
    group = _cgroup(pid)
    if group is not None:
        _refresh_cgroup(sample, group)
    try:
        sample["disk_free_bytes"] = shutil.disk_usage("/content").free
    except OSError:
        pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=3, check=False)
        if result.returncode == 0 and len(result.stdout) < 4096:
            sample["gpu"] = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return sample


def _oom_delta(samples):
    for key in ("cgroup_oom_kill", "cgroup_oom"):
        previous = {}
        for sample in samples:
            scope, value = sample.get("cgroup_scope"), sample.get(key)
            if scope and type(value) is int:
                if scope in previous and value > previous[scope]:
                    return True
                previous.setdefault(scope, value)
    return False


def _pressure(sample):
    available = sample.get("mem_available_bytes")
    if type(available) is int and available < MEMORY_RESERVE:
        return "low_host_memory"
    current, maximum = sample.get("cgroup_memory_current"), sample.get("cgroup_memory_max")
    if type(current) is int and type(maximum) is int and maximum - current < MEMORY_RESERVE:
        return "low_cgroup_memory"
    free = sample.get("disk_free_bytes")
    if type(free) is int and free < DISK_RESERVE:
        return "low_disk_space"
    return None


def failure_kind(returncode, stderr_tail, samples):
    """A negative return code alone does not identify an OOM or a policy limit."""
    if _oom_delta(samples):
        return "observed_cgroup_oom_event"
    if any(_pressure(s) for s in samples):
        return "resource_pressure"
    text = stderr_tail.lower()
    blockers = ("out of memory", "cuda error", "integrity", "sha256", "hash", "required",
                "mismatch", "incompatible", "quota", "usage limit", "permission", "keyboardinterrupt")
    if any(word in text for word in blockers):
        return "non_transient_error"
    if returncode in (-9, -15):
        return "external_termination_unknown"
    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if re.fullmatch(r"(?:urllib\.error\.)?httperror:\s*(?:http error\s*)?(?:502|503|504)(?::.*)?", last_line):
        return "transient_http_server_error"
    return "process_error_unknown"


def load_previous_status(records_root):
    root = Path(records_root) / LOG_FOLDER
    latest = _json(root / "latest.json")
    if not latest:
        return None
    name = latest.get("run_name", "")
    if not re.fullmatch(r"run_[0-9a-f]{32}", name):
        raise ValueError("invalid_private_supervision_pointer")
    result = _json(root / name / "run.json")
    if not result:
        raise ValueError("private_supervision_record_missing")
    if result.get("state") in ("running", "retrying"):
        result = {**result, "state": "interrupted_unknown", "termination_cause_confirmed": False,
                  "note": "No final receipt was saved. The previous controller or runtime may have ended; cause unknown."}
    return result


def show_previous(records_root):
    result = load_previous_status(records_root)
    if result is not None:
        fields = ("version", "state", "attempts", "returncode", "signal", "failure_kind",
                  "last_progress", "last_sample", "resource_summary", "termination_cause_confirmed", "note")
        print("Último registro operativo de la libreta 42 (no inicia entrenamiento):", flush=True)
        print(json.dumps({key: result[key] for key in fields if key in result}, ensure_ascii=False, indent=2), flush=True)
    return result


@dataclass
class RunOutcome:
    stdout: str
    output_dir: Path
    run_dir: Path
    attempts: int


class TrainingStopped(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("La ejecución se detuvo; el diagnóstico quedó guardado en Drive. "
                         f"Motivo registrado: {report.get('failure_kind')}; salida: {report.get('returncode')}. "
                         "Se conserva el avance verificado; no cambies la receta.")


def _terminate(process):
    if process is None:
        return
    if process.poll() is not None:
        # The leader may be gone while its descendants still own a pipe.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def run(arguments, *, cwd, env, records_root, session_root, code_revision,
        sample_interval=10, heartbeat_interval=30, max_retries=1, retry_delay=30,
        sample_provider=None):
    arguments = [str(x) for x in arguments]
    def argument(name):
        if arguments.count(name) != 1:
            raise ValueError("one_bounded_training_argument_required")
        return arguments[arguments.index(name) + 1]
    if argument("--operation") != "train" or argument("--campaign") != "gradual_four_models":
        raise ValueError("only_authorized_gradual_campaign_may_be_supervised")
    argument("--output-dir")
    if type(max_retries) is not int or not 0 <= max_retries <= 1:
        raise ValueError("at_most_one_transient_retry_per_invocation")
    if min(sample_interval, heartbeat_interval) <= 0 or retry_delay < 0:
        raise ValueError("positive_supervision_intervals_required")
    root = Path(records_root).resolve()
    session = Path(session_root).resolve()
    if not root.is_dir() or not session.is_dir():
        raise ValueError("existing_private_records_and_session_required")
    logs = root / LOG_FOLDER
    logs.mkdir(exist_ok=True)
    active_path = logs / "active.json"
    old = _json(active_path)
    if active_path.exists() and old is None:
        raise ValueError("unreadable_active_supervision_record")
    if old and old.get("state") in ("running", "retrying"):
        if _alive(old.get("child")) is True or _alive(old.get("controller")) is True:
            raise RuntimeError("La campaña anterior todavía tiene un proceso activo; no se inicia otra copia.")
        if time.time() - old.get("heartbeat_unix", 0) < LEASE_SECONDS:
            raise RuntimeError("El intento anterior aún tiene un registro reciente; espera dos minutos para comprobar su cierre.")
    previous = load_previous_status(root)
    if previous and previous.get("state") == "interrupted_unknown":
        print("El intento anterior no dejó un cierre: se conservará su evidencia; la causa sigue sin confirmar.", flush=True)
    run_name = "run_" + uuid.uuid4().hex
    run_dir = logs / run_name
    run_dir.mkdir()
    sampler = sample_provider or sample_resources
    controller = _identity(os.getpid())
    report = {"version": VERSION, "state": "running", "started_at_utc": _now(),
              "code_revision": code_revision, "controller": controller,
              "attempts": 0, "termination_cause_confirmed": False,
              "max_transient_retries": max_retries, "training_recipe_changed": False,
              "reserved_test_opened": False}
    process = None
    attempt_report = None
    attempt_dir = None
    lease_owned = False
    try:
        # O_EXCL protects the initial claim. A stale, finalized claim is retired only here.
        if old is not None:
            if _json(active_path) != old:
                raise RuntimeError("El registro activo cambió; no se inicia una segunda ejecución.")
            active_path.unlink()
        with active_path.open("x", encoding="utf-8") as stream:
            json.dump({"run_name": run_name, "state": "running", "controller": controller,
                       "child": None, "heartbeat_unix": time.time()}, stream)
            _sync(stream)
        lease_owned = True
        _save(run_dir / "run.json", report)
        _save(logs / "latest.json", {"run_name": run_name})
        for attempt in range(1, max_retries + 2):
            attempt_dir = run_dir / f"attempt_{attempt:03d}"
            attempt_dir.mkdir()
            output = session / ("gradual_campaign_" + uuid.uuid4().hex)
            command = list(arguments)
            command[command.index("--output-dir") + 1] = str(output)
            attempt_report = {"version": VERSION, "state": "running", "attempt": attempt,
                              "started_at_utc": _now(), "output_dir": str(output),
                              "termination_cause_confirmed": False}
            report.update(attempts=attempt, state="running")
            samples = deque(maxlen=720)
            tail = deque(maxlen=100)
            stdout_lines = []
            pressure_reason = None
            process = None
            initial = sampler(None)
            samples.append(initial)
            report["last_sample"] = initial
            _save(attempt_dir / "status.json", attempt_report)
            _save(run_dir / "run.json", report)
            if _pressure(initial):
                pressure_reason = _pressure(initial)
            with (attempt_dir / "stdout.log").open("w", encoding="utf-8", newline="\n") as out_log, \
                 (attempt_dir / "stderr.log").open("w", encoding="utf-8", newline="\n") as err_log, \
                 (attempt_dir / "telemetry.jsonl").open("w", encoding="utf-8", newline="\n") as telemetry:
                def persist_sample(sample):
                    telemetry.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    _sync(telemetry)
                persist_sample(initial)
                if not pressure_reason:
                    process = subprocess.Popen(command, cwd=cwd, env=env, text=True, bufsize=1,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               start_new_session=os.name == "posix")
                    attempt_report["child"] = _identity(process.pid)
                messages = queue.Queue(maxsize=1024)
                def forward(stream, channel):
                    try:
                        for line in iter(lambda: stream.readline(MAX_LINE_BYTES + 1), ""):
                            if len(line) > MAX_LINE_BYTES:
                                messages.put(("reader_error", "bounded_pipe_line_limit_exceeded"))
                                return
                            messages.put((channel, line))
                    except (OSError, ValueError) as error:
                        messages.put(("reader_error", type(error).__name__))
                    finally:
                        stream.close()
                        messages.put((channel, None))
                if process is not None:
                    for stream, channel in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                        threading.Thread(target=forward, args=(stream, channel), daemon=True).start()
                started = last_sample = last_heartbeat = time.monotonic()
                closed, bytes_written = set(), {"stdout": 0, "stderr": 0}
                exited_at, stdout_bytes = None, 0
                while process is not None:
                    try:
                        channel, line = messages.get(timeout=min(.25, sample_interval))
                    except queue.Empty:
                        channel, line = None, None
                    if channel == "reader_error":
                        raise RuntimeError(line)
                    if channel is not None and line is None:
                        closed.add(channel)
                    elif line is not None:
                        safe = sanitize(line)
                        if channel == "stdout":
                            stdout_bytes += len(line.encode("utf-8"))
                            if stdout_bytes > MAX_STREAM_BYTES:
                                raise RuntimeError("bounded_structured_stdout_limit_exceeded")
                            stdout_lines.append(line)
                        else:
                            tail.append(safe)
                            print(safe.rstrip(), flush=True)
                            if re.search(r"Recuperación:|Trabajo \d|Epoca \d|Reanudando ", safe):
                                report["last_progress"] = safe.strip()[:1000]
                        if bytes_written[channel] < MAX_STREAM_BYTES:
                            stream = out_log if channel == "stdout" else err_log
                            stream.write(safe[:MAX_STREAM_BYTES - bytes_written[channel]])
                            bytes_written[channel] += len(safe.encode("utf-8"))
                    current = time.monotonic()
                    if current - last_sample >= sample_interval:
                        sample = sampler(process.pid)
                        samples.append(sample)
                        report["last_sample"] = sample
                        persist_sample(sample)
                        for stream in (out_log, err_log):
                            _sync(stream)
                        _save(attempt_dir / "status.json", {**attempt_report, "last_sample": sample,
                                                           "last_progress": report.get("last_progress")})
                        _save(run_dir / "run.json", report)
                        _save(active_path, {"run_name": run_name, "state": "running", "controller": controller,
                                            "child": attempt_report.get("child"), "heartbeat_unix": time.time()})
                        last_sample = current
                        reason = _pressure(sample)
                        if reason:
                            pressure_reason = reason
                            print("Protección de recursos: se detectó poco espacio o memoria; se detiene el proceso y se guarda el motivo en Drive.", flush=True)
                            _terminate(process)
                    if current - last_heartbeat >= heartbeat_interval and process.poll() is None:
                        sample = report.get("last_sample", {})
                        available = sample.get("mem_available_bytes")
                        memory = f"{available / 1024 ** 3:.2f} GiB" if type(available) is int else "no disponible"
                        print(f"Proceso activo: {int(current-started)} s; RAM disponible: {memory}. No es un porcentaje de avance.", flush=True)
                        last_heartbeat = current
                    if process.poll() is not None:
                        exited_at = current if exited_at is None else exited_at
                        if len(closed) == 2 or current - exited_at >= 3:
                            if len(closed) != 2:
                                _terminate(process)
                            break
                code = process.wait(timeout=2) if process is not None else None
                final_sample = sampler(process.pid if process is not None else None)
                if sample_provider is None and not final_sample.get("cgroup_scope"):
                    scope = next((s.get("cgroup_scope") for s in reversed(samples)
                                  if s.get("cgroup_scope")), None)
                    if scope and Path(scope).is_relative_to(Path("/sys/fs/cgroup")):
                        _refresh_cgroup(final_sample, Path(scope))
                samples.append(final_sample)
                persist_sample(final_sample)
                for stream in (out_log, err_log):
                    _sync(stream)
            category = pressure_reason or ("success" if code == 0 else failure_kind(code, "".join(tail), list(samples)))
            state = "resource_guard" if pressure_reason else ("succeeded" if code == 0 else "failed")
            signal_name = None
            if type(code) is int and code < 0:
                try:
                    signal_name = signal.Signals(-code).name
                except ValueError:
                    pass
            attempt_report.update(state=state, finished_at_utc=_now(), returncode=code,
                                  signal=signal_name, failure_kind=category, last_sample=final_sample,
                                  stderr_tail="".join(tail)[-16000:])
            report.update({key: value for key, value in attempt_report.items() if key not in ("child", "stderr_tail")})
            report["last_sample"] = final_sample
            resource_summary = {}
            for key, operation in (("mem_available_bytes", min), ("child_rss_bytes", max),
                                    ("process_tree_rss_bytes", max), ("parent_rss_bytes", max)):
                values = [s[key] for s in samples if type(s.get(key)) is int]
                resource_summary[key] = operation(values) if values else None
            report["resource_summary"] = resource_summary
            retrying = category == "transient_http_server_error" and attempt <= max_retries
            if retrying:
                report["state"] = "retrying"
            _save(attempt_dir / "status.json", attempt_report)
            _save(run_dir / "run.json", report)
            _save(active_path, {"run_name": run_name, "state": "retrying" if retrying else state, "controller": controller,
                                "child": attempt_report.get("child"), "heartbeat_unix": time.time()})
            if state == "succeeded":
                return RunOutcome("".join(stdout_lines), output, run_dir, attempt)
            if category != "transient_http_server_error" or attempt > max_retries:
                raise TrainingStopped(report)
            print("Fallo HTTP transitorio identificado. Se hará un único reintento desde el avance guardado.", flush=True)
            time.sleep(retry_delay)
        raise AssertionError("bounded_attempt_loop_exhausted")
    except BaseException as error:
        _terminate(process)
        if isinstance(error, TrainingStopped):
            raise
        cancelled = isinstance(error, (KeyboardInterrupt, SystemExit))
        report.update(state="cancelled" if cancelled else "failed", finished_at_utc=_now(),
                      failure_kind="controller_cancelled" if cancelled else "supervisor_or_persistence_error",
                      returncode=process.returncode if process is not None else None,
                      supervisor_error=sanitize(f"{type(error).__name__}: {error}")[:2000])
        # A Drive failure must never silently turn supervision off. Try local evidence too.
        for path in (session / (run_name + "_stop.json"), run_dir / "run.json"):
            try:
                _save(path, report)
            except OSError:
                pass
        if attempt_dir is not None and attempt_report is not None:
            try:
                _save(attempt_dir / "status.json", {**attempt_report, **report})
            except OSError:
                pass
        if lease_owned:
            try:
                _save(active_path, {"run_name": run_name, "state": report["state"],
                                    "controller": controller, "heartbeat_unix": time.time()})
            except OSError:
                pass
        if cancelled:
            raise
        raise TrainingStopped(report) from error
