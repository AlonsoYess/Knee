"""Isolated Docker/HTTPS rehearsal with fictional data and retained test volumes.

Requires an already built local application image. Never builds, publishes, prunes,
or uses the investigator's runtime, database, radiographs or scientific assets.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

import rehearse_host
from smoke_https import NoRedirect


DEPLOY = Path(__file__).resolve().parent
PORT = 18443
ORIGIN = f"https://127.0.0.1:{PORT}"
DATA = "/var/lib/kneeoai"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


class Rehearsal:
    def __init__(self, output, image, openssl):
        self.output, self.image, self.openssl = output, image, openssl
        self.project = "kneeoai-rehearsal-" + secrets.token_hex(6)
        self.export_container = self.project + "-backup-export"
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("KNEEOAI_", "UVICORN_", "COMPOSE_"))
                    and key not in {"WEB_CONCURRENCY", "FORWARDED_ALLOW_IPS"}}
        self.env.update(PYTHONUTF8="1")
        self.files = [DEPLOY / "compose.yml", DEPLOY / "compose.rehearsal.yml"]
        self.original_volume = self.project + "_app_data"
        self.restored_volume = self.project + "_restored_data"
        self.started = False
        self.export_created = False
        self.restored_started = False
        self.summary = {
            "scope": "isolated_docker_https_application_only",
            "passed": False, "syntheticOnly": True,
            "applicationImage": image, "project": self.project,
            "osCertificateStoreChanged": False,
            "scientificAssetsLoaded": False, "predictionExecuted": False,
            "originalVolume": self.original_volume,
            "restoredVolume": self.restored_volume,
            "retainedVolumes": [], "checks": {},
        }

    def run(self, arguments, *, stage, timeout=180):
        try:
            result = subprocess.run([str(item) for item in arguments], cwd=DEPLOY,
                                    env=self.env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError(f"Rehearsal stage could not finish: {stage}") from None
        if result.returncode:
            with (self.output / "commands.log").open("a", encoding="utf-8") as stream:
                stream.write(f"Stage: {stage}\nExit: {result.returncode}\n{result.stdout}\n{result.stderr}\n")
            raise RuntimeError(f"Rehearsal stage failed: {stage}; consult its private report")
        return result.stdout.strip()

    def compose(self, arguments, *, stage, override=None, timeout=240):
        command = ["docker", "compose", "--project-name", self.project]
        for path in self.files + ([override] if override else []):
            command.extend(["--file", str(path)])
        return self.run(command + list(arguments), stage=stage, timeout=timeout)

    def inspect(self, container):
        return json.loads(self.run(["docker", "inspect", container], stage="inspect-owned-container"))[0]

    def remove_owned_export(self):
        identifiers = self.run([
            "docker", "ps", "--all", "--filter", "name=^/" + self.export_container + "$",
            "--filter", "label=com.docker.compose.project=" + self.project,
            "--filter", "label=com.docker.compose.service=app", "--format", "{{.ID}}",
        ], stage="identify-owned-export").splitlines()
        if not identifiers:
            return
        if len(identifiers) != 1:
            raise RuntimeError("The owned export container could not be identified uniquely")
        info = self.inspect(identifiers[0])
        labels = info["Config"].get("Labels", {})
        if (info.get("Name") != "/" + self.export_container
                or labels.get("com.docker.compose.project") != self.project
                or labels.get("com.docker.compose.service") != "app"):
            raise RuntimeError("The export container is outside this rehearsal")
        self.run(["docker", "rm", info["Id"]], stage="remove-owned-export-container")

    def app_id(self, override=None):
        identifier = self.compose(["ps", "--quiet", "app"], stage="identify-owned-app", override=override)
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise RuntimeError("The rehearsal application container could not be identified")
        info = self.inspect(identifier)
        labels = info["Config"].get("Labels", {})
        if labels.get("com.docker.compose.project") != self.project or labels.get("com.docker.compose.service") != "app":
            raise RuntimeError("The identified application is outside this rehearsal")
        return identifier

    def wait_https(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(self.output / "ca.pem"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                            urllib.request.HTTPSHandler(context=context))
        for _ in range(60):
            try:
                with opener.open(ORIGIN + "/api/v1/health", timeout=2) as response:
                    data = json.load(response)
                    if response.status == 200 and data == {"status": "ok", "version": "0.5.0"}:
                        return
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise RuntimeError("The isolated proxy did not provide verified HTTPS health")

    def smoke(self, mode):
        self.wait_https()
        self.run([sys.executable, DEPLOY / "smoke_https.py", "--mode", mode,
                  "--base-url", ORIGIN, "--ca-file", self.output / "ca.pem",
                  "--state-file", self.output / "state.json",
                  "--cookie-file", self.output / "test-cookies.json",
                  "--report-file", self.output / (mode + ".json")], stage="https-" + mode, timeout=240)
        report = json.loads((self.output / (mode + ".json")).read_text(encoding="utf-8"))
        if report.get("status") != "passed" or not all(item.get("ok") for item in report["checks"]):
            raise RuntimeError("An HTTPS rehearsal check was not successful")
        self.summary["checks"][mode] = {"passed": True, "count": len(report["checks"])}

    def verify_frontend(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(self.output / "ca.pem"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                            urllib.request.HTTPSHandler(context=context))

        def fetch(path):
            if not path.startswith("/") or path.startswith("//") or any(value in path for value in ("?", "#", "..", "\\")):
                raise RuntimeError("An unexpected public asset path was declared")
            with opener.open(ORIGIN + path, timeout=10) as response:
                if response.status != 200 or response.headers.get("Set-Cookie"):
                    raise RuntimeError("A public frontend resource failed anonymous delivery")
                return response.read(), response.headers

        shell, headers = fetch("/acceso")
        if "text/html" not in headers.get("Content-Type", ""):
            raise RuntimeError("The access route did not deliver the public HTML shell")
        scripts = re.findall(r'''<script\b[^>]*\bsrc=["']([^"']+)["']''', shell.decode("utf-8"), re.IGNORECASE)
        if not scripts:
            raise RuntimeError("The public shell contains no compiled script")
        worker, headers = fetch("/sw.js")
        if (not worker or "javascript" not in headers.get("Content-Type", "")
                or not any(value in headers.get("Cache-Control", "") for value in ("no-cache", "no-store"))
                or headers.get("Service-Worker-Allowed") != "/"):
            raise RuntimeError("The service worker delivery policy is invalid")
        manifest = json.loads(fetch("/manifest.webmanifest")[0])
        version = json.loads(fetch("/pwa-version.json")[0])
        if manifest.get("start_url") != "/acceso" or manifest.get("scope") != "/" or version.get("appVersion") != "0.12.0":
            raise RuntimeError("The PWA identity or version is not the expected application")
        assets = version.get("assets", [])
        urls = {item["url"] for item in assets}
        if not assets or not all(path in urls and path.startswith("/assets/") for path in scripts):
            raise RuntimeError("The HTML scripts are absent from the public PWA inventory")
        for item in assets:
            content, _ = fetch(item["url"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuntimeError("A served PWA resource differs from its declared digest")
        self.summary["checks"]["frontendHttps"] = {
            "passed": True, "publicAssetDigestsVerified": len(assets),
            "compiledScriptsVerified": len(scripts), "serviceWorkerRevalidationPolicyVerified": True,
        }

    def verify_runtime(self, identifier, expected_volume, expected_data):
        info = self.inspect(identifier)
        host = info["HostConfig"]
        if info["Config"]["User"] != "10001:10001" or not host["ReadonlyRootfs"]:
            raise RuntimeError("The app does not use its declared unprivileged read-only runtime")
        if "ALL" not in host.get("CapDrop", []) or not any("no-new-privileges" in value for value in host.get("SecurityOpt", [])):
            raise RuntimeError("The app security restrictions were not applied")
        mount = next((item for item in info["Mounts"] if item["Destination"] == DATA), None)
        if not mount or mount.get("Type") != "volume" or mount.get("Name") != expected_volume or not mount.get("RW"):
            raise RuntimeError("The app is not using the dedicated writable rehearsal volume")
        check = (
            "import importlib.util,json,os,pathlib;"
            "from knee_app.config import Settings;"
            "s=Settings.from_env();"
            "assert os.getuid()==10001 and os.getgid()==10001;"
            f"assert str(s.data_dir)=={expected_data!r};"
            "assert pathlib.Path('/var/lib/kneeoai').stat().st_uid==10001;"
            "assert importlib.util.find_spec('torch') is None;"
            "assert importlib.util.find_spec('numpy') is None;"
            "assert pathlib.Path('/opt/kneeoai/frontend/sw.js').is_file();"
            "assert s.frozen_python is None and s.frozen_checkout is None;"
            "print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'dataDirectoryWritable':os.access(s.data_dir,os.W_OK)}))"
        )
        report = json.loads(self.run(["docker", "exec", identifier, "python", "-c", check], stage="runtime-boundary"))
        if not report["dataDirectoryWritable"]:
            raise RuntimeError("The rehearsal app cannot write its dedicated data directory")
        return {"uid": report["uid"], "gid": report["gid"], "readOnlyRoot": True,
                "persistentVolume": expected_volume, "scientificLibrariesAbsent": True}

    def verify_backup(self):
        backup = self.output / "backup"
        manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "kneeoai-private-package-v1" or not manifest.get("objects"):
            raise RuntimeError("The complete fictional backup has no referenced objects")
        for path, metadata in [(backup / "application.sqlite3", manifest["database"])]:
            data = path.read_bytes()
            if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise RuntimeError("Backup database integrity failed")
        for item in manifest["objects"]:
            key = PurePosixPath(item["key"])
            if key.is_absolute() or ".." in key.parts or not key.parts:
                raise RuntimeError("An unexpected object path was found in the fictional backup")
            data = (backup / "private_objects" / key).read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError("Backup object integrity failed")
        fixture = json.loads((self.output / "state.json").read_text(encoding="utf-8"))["fixtureSha256"]
        if not any(item["sha256"] == fixture for item in manifest["objects"]):
            raise RuntimeError("The generated original DICOM was omitted from the complete backup")
        return manifest

    def observe_resources(self, app_identifier, image_bytes):
        proxy_identifier = self.compose(["ps", "--quiet", "proxy"], stage="identify-owned-proxy")
        if not re.fullmatch(r"[0-9a-f]{64}", proxy_identifier):
            raise RuntimeError("The rehearsal proxy container could not be identified")
        proxy = self.inspect(proxy_identifier)
        labels = proxy["Config"].get("Labels", {})
        host = proxy["HostConfig"]
        if (labels.get("com.docker.compose.project") != self.project
                or labels.get("com.docker.compose.service") != "proxy"
                or proxy["Config"]["User"] != "10002:10002"
                or not host["ReadonlyRootfs"] or "ALL" not in host.get("CapDrop", [])
                or not any("no-new-privileges" in value for value in host.get("SecurityOpt", []))):
            raise RuntimeError("The proxy does not use the owned unprivileged read-only runtime")
        published = [(port, item["HostIp"], item["HostPort"])
                     for port, bindings in proxy["NetworkSettings"]["Ports"].items()
                     for item in bindings or []]
        if published != [("8443/tcp", "127.0.0.1", str(PORT))]:
            raise RuntimeError("The rehearsal proxy is not restricted to its dedicated loopback port")
        statistics = self.run(["docker", "stats", "--no-stream", "--format", "{{json .}}",
                               app_identifier, proxy_identifier], stage="observe-small-fictional-workload", timeout=60)
        snapshots = [json.loads(line) for line in statistics.splitlines() if line.strip()]
        if len(snapshots) != 2:
            raise RuntimeError("The resource snapshots do not cover both owned services")
        volume_bytes = self.run([
            "docker", "exec", app_identifier, "python", "-c",
            "import pathlib;root=pathlib.Path('/var/lib/kneeoai');"
            "print(sum(item.stat().st_size for item in root.rglob('*') if item.is_file()))",
        ], stage="measure-fictional-data-bytes")
        self.summary["proxyRuntime"] = {"uid": 10002, "readOnlyRoot": True,
                                        "publishedAddress": "127.0.0.1", "publishedPort": PORT}
        self.summary["observedResources"] = {
            "scope": "one_small_fictional_case_without_scientific_models",
            "capacityOrMaximumLoadValidated": False,
            "applicationImageBytes": image_bytes,
            "fictionalDataFileBytes": int(volume_bytes), "dockerStatsSnapshots": snapshots,
        }

    def collect_failure_logs(self):
        if self.started:
            try:
                logs = self.compose(["logs", "--no-color", "--tail", "100", "app", "proxy"],
                                    stage="collect-owned-failure-logs")
                (self.output / "containers.log").write_text(logs + "\n", encoding="utf-8")
            except RuntimeError:
                self.summary["failureLogsUnavailable"] = True

    def execute(self):
        # Check all resource names before the first mutation. The random project is
        # still explicitly checked so a pre-existing resource is never adopted.
        existing = self.run(["docker", "ps", "--all", "--filter", "label=com.docker.compose.project=" + self.project,
                             "--format", "{{.ID}}"], stage="check-project-is-new")
        volumes = set(self.run(["docker", "volume", "ls", "--format", "{{.Name}}"], stage="check-volumes-are-new").splitlines())
        export_exists = self.run(["docker", "ps", "--all", "--filter", "name=^/" + self.export_container + "$",
                                  "--format", "{{.ID}}"], stage="check-export-name-is-new")
        networks = set(self.run(["docker", "network", "ls", "--format", "{{.Name}}"], stage="check-network-is-new").splitlines())
        if existing or export_exists or self.project + "_default" in networks or self.original_volume in volumes or self.restored_volume in volumes:
            raise RuntimeError("The rehearsal resource names are not new")
        image = json.loads(self.run(["docker", "image", "inspect", self.image], stage="verify-local-image"))[0]
        self.summary["imageId"] = image["Id"]
        rehearse_host.PRIVATE_LOG = self.output / "commands.log"
        _, key, cert = rehearse_host.certificates(self.openssl, self.output)
        secret = self.output / "session.key"
        secret.write_text(secrets.token_hex(32), encoding="ascii")
        secret.chmod(0o600)
        if os.name == "posix":
            # Compose file-backed secrets keep host ownership. These individual
            # read-only mounts must be readable by UIDs 10001 and 10002, while
            # the enclosing owner-only directory blocks other host users.
            private = self.output.stat()
            if private.st_uid != os.geteuid() or private.st_mode & 0o077:
                raise RuntimeError("The POSIX rehearsal directory must be owner-only")
            for mounted in (secret, key, cert):
                mounted.chmod(0o644)
        self.env.update(KNEEOAI_IMAGE=self.image, KNEEOAI_ALLOWED_ORIGINS=ORIGIN,
                        KNEEOAI_SESSION_KEY_FILE=str(secret),
                        KNEEOAI_REHEARSAL_CERT_FILE=str(cert), KNEEOAI_REHEARSAL_KEY_FILE=str(key))
        self.compose(["config", "--quiet"], stage="validate-rehearsal-declaration")
        self.started = True
        self.compose(["up", "--detach", "--wait", "--wait-timeout", "120"], stage="start-original-rehearsal")
        old_id = self.app_id()
        self.summary["originalRuntime"] = self.verify_runtime(old_id, self.original_volume, DATA)
        provision = (
            "import time;from knee_app.config import Settings;from knee_app.db import Database,create_user;"
            "s=Settings.from_env();d=Database(s.database_path);d.initialize();"
            "create_user(d,email='browser-a@kneeoai.invalid',password='BrowserTestsOnly-2026!',"
            "display_name='Fictional deployment rehearsal',now=time.time())"
        )
        self.compose(["exec", "-T", "app", "python", "-c", provision], stage="provision-new-fictional-account")
        self.smoke("seed")
        self.verify_frontend()
        self.compose(["up", "--detach", "--force-recreate", "--wait", "--wait-timeout", "120", "app"], stage="recreate-original-app")
        new_id = self.app_id()
        if old_id == new_id:
            raise RuntimeError("The app container was not actually recreated")
        self.summary["checks"]["containerRecreated"] = {"passed": True, "before": old_id, "after": new_id}
        self.smoke("resume")
        self.observe_resources(new_id, image["Size"])
        self.compose(["stop", "app"], stage="stop-before-backup")
        self.compose(["run", "--rm", "--no-deps", "--entrypoint", "python", "app", "-m", "knee_app.cli",
                      "backup", "--package", "--destination", DATA + "/rehearsal-backup"], stage="complete-volume-backup")
        self.export_created = True
        self.compose(["run", "--no-deps", "--name", self.export_container, "--entrypoint", "python", "app", "-c", "pass"], stage="create-backup-export-container")
        self.run(["docker", "cp", self.export_container + ":" + DATA + "/rehearsal-backup", self.output / "backup"], stage="export-independent-backup")
        self.remove_owned_export()
        self.export_created = False
        manifest = self.verify_backup()
        self.summary["checks"]["completeIndependentBackupVerified"] = {"passed": True, "objects": len(manifest["objects"])}
        restored_override = self.output / "restored.compose.json"
        write_json(restored_override, {
            "services": {"app": {"environment": {"KNEEOAI_DATA_DIR": DATA + "/recovered"}}},
            "volumes": {"app_data": {"name": self.restored_volume}},
        })
        restore_override = self.output / "restore-command.compose.json"
        write_json(restore_override, {
            "services": {"app": {
                "environment": {"KNEEOAI_DATA_DIR": DATA + "/recovered"},
                "volumes": [{"type": "bind", "source": str(self.output / "backup"),
                             "target": "/rehearsal-input", "read_only": True}],
            }},
            "volumes": {"app_data": {"name": self.restored_volume}},
        })
        self.compose(["config", "--quiet"], stage="validate-restoration-declaration", override=restore_override)
        restore_command = (
            "import shutil;from knee_app.cli import main;"
            "shutil.copytree('/rehearsal-input','/tmp/rehearsal-restore-input');"
            "raise SystemExit(main(['restore','--package','--source','/tmp/rehearsal-restore-input',"
            f"'--destination',{(DATA + '/recovered')!r}]))"
        )
        self.compose(["run", "--rm", "--no-deps", "--entrypoint", "python", "app", "-c", restore_command],
                     stage="restore-to-new-volume", override=restore_override)
        self.restored_started = True
        self.compose(["up", "--detach", "--force-recreate", "--wait", "--wait-timeout", "120", "app"],
                     stage="start-restored-volume", override=restored_override)
        restored_id = self.app_id(restored_override)
        self.summary["restoredRuntime"] = self.verify_runtime(restored_id, self.restored_volume, DATA + "/recovered")
        if any(item["Destination"] == "/rehearsal-input" for item in self.inspect(restored_id)["Mounts"]):
            raise RuntimeError("The restored server still mounts the original volume")
        verify_objects = (
            "import hashlib,json,pathlib;"
            "root=pathlib.Path('/var/lib/kneeoai/recovered/private_objects');"
            f"items={manifest['objects']!r};"
            "assert all((lambda data:len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'])"
            "((root/item['key']).read_bytes()) for item in items);"
            "print('Restored fictional object bytes verified')"
        )
        self.run(["docker", "exec", restored_id, "python", "-c", verify_objects], stage="verify-restored-object-bytes")
        self.smoke("restored")
        self.summary["checks"]["separateVolumeRestoration"] = {"passed": True, "oldSessionsRevoked": True, "objectBytesVerified": True}
        self.summary["passed"] = True

    def cleanup(self):
        failures = []
        if self.export_created:
            try:
                self.remove_owned_export()
            except RuntimeError:
                failures.append("backup-export-container")
        if self.started:
            try:
                self.compose(["down", "--timeout", "30"], stage="stop-owned-rehearsal",
                             override=self.output / "restored.compose.json" if self.restored_started else None)
                self.summary["checks"]["ownedServicesStopped"] = {"passed": True}
            except RuntimeError:
                failures.append("compose-project")
        try:
            volumes = set(self.run(["docker", "volume", "ls", "--format", "{{.Name}}"], stage="record-retained-test-volumes").splitlines())
            self.summary["retainedVolumes"] = [name for name in (self.original_volume, self.restored_volume) if name in volumes]
        except RuntimeError:
            failures.append("volume-inventory")
        if failures:
            self.summary["passed"] = False
            self.summary["cleanupFailures"] = failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--openssl", default=shutil.which("openssl"))
    args = parser.parse_args()
    if not args.openssl:
        parser.error("OpenSSL is required to generate an isolated test CA")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/@-]{0,254}", args.image):
        parser.error("Use an explicit already-built local image reference")
    output = args.output.absolute()
    if output.exists() or any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in output.parents):
        parser.error("The private output directory must be new and must not follow links")
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", PORT))
    output.mkdir(parents=True, mode=0o700)
    rehearsal = Rehearsal(output, args.image, args.openssl)
    try:
        rehearsal.execute()
    except Exception as error:
        rehearsal.summary["error"] = str(error)
        rehearsal.collect_failure_logs()
    finally:
        rehearsal.cleanup()
        write_json(output / "docker-rehearsal.json", rehearsal.summary)
        print(json.dumps(rehearsal.summary, allow_nan=False))
    return 0 if rehearsal.summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
