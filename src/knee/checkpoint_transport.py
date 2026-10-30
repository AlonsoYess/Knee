"""Bounded HTTPS checkpoint acquisition; no model code or deserialization.

Full SHA-256 is mandatory before a checkpoint may be loaded by the verifier.
The optional digest here also supports first acquisition of a public upstream
file whose publisher supplies only a checksum prefix. Its measured full hash
must then be pinned in the reviewed catalog before loading.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request


SOURCE_HOSTS = {"download.pytorch.org", "dl.fbaipublicfiles.com", "huggingface.co"}
# Exact file-service domains published by Hugging Face on 2026-10-22:
# https://huggingface.co/.well-known/meta.json (cdnDomains and lfsDomains).
# Colab can be routed to a regional GCP edge instead of the AWS edge observed
# locally. These hosts are redirect destinations only, never new origins.
HUGGINGFACE_FILE_HOSTS = frozenset({
    "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    "transfer.xethub.hf.co", "transfer.xethub-eu.hf.co",
    "aws.cdn.hf.co", "us.aws.cdn.hf.co", "us-east-1.aws.cdn.hf.co",
    "us-west-2.aws.cdn.hf.co", "eu-west-3.aws.cdn.hf.co",
    "ap-southeast-1.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
    "us-east1.us.gcp.cdn.hf.co", "us-central1.us.gcp.cdn.hf.co",
    "us-west4.us.gcp.cdn.hf.co", "europe-west4.us.gcp.cdn.hf.co",
    "asia-southeast1.us.gcp.cdn.hf.co",
})
REDIRECT_HOSTS = SOURCE_HOSTS | {"cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co",
    "cas-bridge.xethub.hf.co"} | HUGGINGFACE_FILE_HOSTS


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _https(url, hosts):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        # Report only a bounded hostname, never signed queries or credentials.
        host = parsed.hostname or ""
        safe_host = host if re.fullmatch(r"[a-z0-9.-]{1,253}", host) else "invalid"
        raise ValueError("official_https_checkpoint_source_required:host=" + safe_host)


class _OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl, REDIRECT_HOSTS)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def acquire(url, destination, *, expected_sha256=None, sha256_prefix=None,
            expected_bytes=None, maximum_bytes=2_000_000_000):
    """Acquire a public checkpoint atomically without replacing existing data.

    Never accepts credentials, evaluates remote code, or opens a dataset.
    Existing files are read and checked, never overwritten or repaired.
    """
    _https(url, SOURCE_HOSTS)
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid_full_checkpoint_digest")
    if sha256_prefix is not None and not re.fullmatch(r"[0-9a-f]{8,64}", sha256_prefix):
        raise ValueError("invalid_checkpoint_digest_prefix")
    if type(maximum_bytes) is not int or not 0 < maximum_bytes <= 2_000_000_000:
        raise ValueError("bounded_checkpoint_size_required")
    if expected_bytes is not None and (type(expected_bytes) is not int
            or not 0 < expected_bytes <= maximum_bytes):
        raise ValueError("invalid_expected_checkpoint_size")
    path = Path(destination)
    if path.is_symlink():
        raise ValueError("checkpoint_symlink_forbidden")
    temporary = path.with_name(path.name + ".partial")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            raise ValueError("incomplete_checkpoint_present_no_overwrite")
        opener = urllib.request.build_opener(_OfficialRedirects())
        request = urllib.request.Request(url, headers={"User-Agent": "KneeOAI-checkpoint-verification/1.0"})
        created = False
        try:
            with opener.open(request, timeout=60) as response:
                _https(response.geturl(), REDIRECT_HOSTS)
                length = response.headers.get("Content-Length")
                if length is not None and (int(length) > maximum_bytes or
                        (expected_bytes is not None and int(length) != expected_bytes)):
                    raise ValueError("checkpoint_content_length_mismatch")
                count = 0
                with temporary.open("xb") as stream:
                    created = True
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > maximum_bytes:
                            raise ValueError("checkpoint_size_limit_exceeded")
                        stream.write(chunk)
                if length is not None and count != int(length):
                    raise ValueError("checkpoint_download_truncated")
                _verify(temporary, expected_sha256, sha256_prefix, expected_bytes, maximum_bytes)
                # Hard-link publication is atomic and refuses an existing target
                # on Windows and Linux. rename() would overwrite on Linux.
                # Use a local filesystem cache (not a Drive/FUSE mount).
                os.link(temporary, path)
                temporary.unlink()
        except Exception:
            # Only our own incomplete download is removed, never a prior file.
            if created and temporary.exists():
                temporary.unlink()
            raise
    return _verify(path, expected_sha256, sha256_prefix, expected_bytes, maximum_bytes)


def _verify(path, digest, prefix, expected_bytes, maximum_bytes):
    if not path.is_file():
        raise ValueError("regular_checkpoint_file_required")
    size = path.stat().st_size
    if not 0 < size <= maximum_bytes or (expected_bytes is not None and size != expected_bytes):
        raise ValueError("checkpoint_size_mismatch")
    actual = file_digest(path)
    if digest is not None and actual != digest:
        raise ValueError("checkpoint_sha256_mismatch")
    if prefix is not None and not actual.startswith(prefix):
        raise ValueError("checkpoint_publisher_checksum_prefix_mismatch")
    return {"sha256": actual, "bytes": size, "full_digest_was_pinned": digest is not None}
