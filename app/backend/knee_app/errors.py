from __future__ import annotations

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, *, fields=None, retryable=False, headers=None):
        super().__init__(code)
        self.status, self.code, self.message = status, code, message
        self.fields = fields or []
        self.retryable = retryable
        self.headers = headers or {}
