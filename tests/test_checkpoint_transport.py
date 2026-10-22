"""Bounded mocked HTTPS acquisition; no network or model loading."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.request

from knee import checkpoint_transport as subject


URL = "https://download.pytorch.org/models/synthetic.pth"
PAYLOAD = b"synthetic-checkpoint-bytes"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Response(io.BytesIO):
    def __init__(self, payload=PAYLOAD, headers=None, url=URL):
        super().__init__(payload)
        self.headers = {} if headers is None else headers
        self.url = url

    def geturl(self):
        return self.url


class CheckpointTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "checkpoint.pth"

    def download(self, response=None, **kwargs):
        opener = Mock()
        opener.open.return_value = Response() if response is None else response
        with patch.object(subject.urllib.request, "build_opener", return_value=opener):
            return subject.acquire(URL, self.path, **kwargs)

    def test_full_hash_and_size_verified_before_atomic_publication(self):
        result = self.download(Response(headers={"Content-Length": str(len(PAYLOAD))}),
                               expected_sha256=SHA, expected_bytes=len(PAYLOAD))
        self.assertEqual(result, {"sha256": SHA, "bytes": len(PAYLOAD), "full_digest_was_pinned": True})
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_first_acquisition_can_measure_full_hash_from_publisher_prefix(self):
        result = self.download(sha256_prefix=SHA[:8])
        self.assertFalse(result["full_digest_was_pinned"])
        self.assertEqual(result["sha256"], SHA)

    def test_existing_verified_file_never_downloaded_or_overwritten(self):
        self.path.write_bytes(PAYLOAD)
        with patch.object(subject.urllib.request, "build_opener") as opener:
            self.assertEqual(subject.acquire(URL, self.path, expected_sha256=SHA)["sha256"], SHA)
        opener.assert_not_called()
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_existing_wrong_hash_or_size_preserved(self):
        self.path.write_bytes(PAYLOAD)
        for kwargs in ({"expected_sha256": "0" * 64}, {"expected_bytes": len(PAYLOAD) + 1}):
            with self.subTest(kwargs=kwargs), patch.object(subject.urllib.request, "build_opener") as opener:
                with self.assertRaises(ValueError):
                    subject.acquire(URL, self.path, **kwargs)
                opener.assert_not_called()
                self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_existing_partial_is_never_overwritten(self):
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_bytes(b"previous-incomplete-file")
        with patch.object(subject.urllib.request, "build_opener") as opener:
            with self.assertRaisesRegex(ValueError, "incomplete_checkpoint"):
                subject.acquire(URL, self.path)
            opener.assert_not_called()
        self.assertEqual(partial.read_bytes(), b"previous-incomplete-file")
        self.assertFalse(self.path.exists())

    def test_bad_download_hash_prefix_or_empty_payload_removes_only_owned_partial(self):
        for payload, kwargs in ((PAYLOAD, {"expected_sha256": "0" * 64}),
                                (PAYLOAD, {"sha256_prefix": "0" * 8}), (b"", {})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.download(Response(payload), **kwargs)
            self.assertFalse(self.path.exists())
            self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_unannounced_oversized_response_is_bounded_and_not_published(self):
        with self.assertRaisesRegex(ValueError, "size_limit"):
            self.download(Response(b"123456789"), maximum_bytes=8)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_header_size_expected_size_and_truncation_are_enforced(self):
        cases = [(Response(headers={"Content-Length": "100"}), {"maximum_bytes": 50}),
                 (Response(headers={"Content-Length": str(len(PAYLOAD))}), {"expected_bytes": 10}),
                 (Response(headers={"Content-Length": str(len(PAYLOAD) + 1)}), {}),
                 (Response(headers={"Content-Length": "not-a-number"}), {})]
        for response, kwargs in cases:
            with self.subTest(headers=response.headers), self.assertRaises(ValueError):
                self.download(response, **kwargs)
            self.assertFalse(self.path.exists())
            self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_disallowed_redirect_is_rejected_before_destination_creation(self):
        with self.assertRaisesRegex(ValueError, "official_https"):
            self.download(Response(url="https://attacker.invalid/model.pth"))
        self.assertFalse(self.path.exists())

    def test_redirect_handler_allows_only_explicit_https_hosts(self):
        handler = subject._OfficialRedirects()
        request = urllib.request.Request(URL)
        for url in ("http://download.pytorch.org/model.pth", "https://evil.invalid/model.pth",
                    "https://download.pytorch.org.evil.invalid/model.pth", "https://user:pass@download.pytorch.org/model.pth"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, url)
        redirected = handler.redirect_request(request, None, 302, "Found", {},
                                              "https://cas-bridge.xethub.hf.co/content")
        self.assertEqual(redirected.full_url, "https://cas-bridge.xethub.hf.co/content")

    def test_source_url_credentials_ports_fragments_and_unknown_hosts_are_rejected(self):
        urls = ["http://download.pytorch.org/x", "https://download.pytorch.org:444/x",
                "https://user@download.pytorch.org/x", "https://download.pytorch.org/x#fragment",
                "https://other.invalid/x", "file:///local/checkpoint"]
        for url in urls:
            with self.subTest(url=url), patch.object(subject.urllib.request, "build_opener") as opener:
                with self.assertRaises(ValueError):
                    subject.acquire(url, self.path)
                opener.assert_not_called()

    def test_invalid_hashes_sizes_and_bool_limits_fail_before_network(self):
        cases = [{"expected_sha256": "abc"}, {"expected_sha256": "F" * 64},
                 {"sha256_prefix": "abc"}, {"maximum_bytes": True}, {"maximum_bytes": 0},
                 {"maximum_bytes": 2_000_000_001}, {"expected_bytes": True},
                 {"expected_bytes": 9, "maximum_bytes": 8}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch.object(subject.urllib.request, "build_opener") as opener:
                with self.assertRaises(ValueError):
                    subject.acquire(URL, self.path, **kwargs)
                opener.assert_not_called()

    def test_stream_failure_removes_owned_partial_and_preserves_other_files(self):
        marker = self.path.parent / "unrelated.bin"
        marker.write_bytes(b"preserve")
        response = Response()
        response.read = Mock(side_effect=[b"partial", OSError("connection interrupted")])
        with self.assertRaises(OSError):
            self.download(response)
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_concurrently_created_destination_is_preserved(self):
        response = Response()
        calls = iter([PAYLOAD, b""])
        def read(_):
            value = next(calls)
            if not value:
                self.path.write_bytes(b"created-by-another-process")
            return value
        response.read = read
        with self.assertRaises(FileExistsError):
            self.download(response)
        self.assertEqual(self.path.read_bytes(), b"created-by-another-process")
        self.assertFalse(self.path.with_name(self.path.name + ".partial").exists())

    def test_directory_destination_is_not_treated_as_a_checkpoint(self):
        self.path.mkdir()
        with self.assertRaisesRegex(ValueError, "regular_checkpoint_file"):
            subject.acquire(URL, self.path)
        self.assertTrue(self.path.is_dir())


if __name__ == "__main__":
    unittest.main()
