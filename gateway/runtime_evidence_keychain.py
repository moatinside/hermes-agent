"""macOS Keychain custody for the Runtime Evidence V2 HMAC key.

Naming contract (fixed): a Generic Password item with service
``agent-level/runtime-evidence/v2`` and account ``hmac:<profile>:v<version>``
(the pilot uses ``hmac:default:v1``) whose password is 64 lowercase hex
characters (32 bytes). The key is read with ``/usr/bin/security
find-generic-password -s ... -a ... -w`` without a shell, captured directly into
memory, and never placed in argv, environment, a file, a log, or an exception
message. Every failure surfaces only as a value-free ``code``.

The provider is lazy: constructing it never calls ``security``. The first call
happens on the observer's background worker (``build_envelope``), so the response
path never blocks on Keychain. A failed fetch is cached fail-closed so the worker
does not retry the Keychain on every turn; each later envelope fails with the same
code and the observer counts it.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from typing import Any, Callable, Optional, Tuple

KEYCHAIN_SERVICE = "agent-level/runtime-evidence/v2"
KEYCHAIN_COMMAND_TIMEOUT_SECONDS = 10
_SECURITY_COMMAND = "/usr/bin/security"
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_KEY_HEX_RE = re.compile(rb"^[a-f0-9]{64}$")


class RuntimeEvidenceKeyError(RuntimeError):
    """Value-free HMAC key custody error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def keychain_account(profile: str, version: int) -> str:
    if not isinstance(profile, str) or _PROFILE_RE.fullmatch(profile) is None:
        raise RuntimeEvidenceKeyError("keychain_profile_invalid")
    if type(version) is not int or version < 1:
        raise RuntimeEvidenceKeyError("keychain_key_version_invalid")
    return f"hmac:{profile}:v{version}"


def load_keychain_hmac_key(
    profile: str,
    version: int,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> bytes:
    """Return the 32-byte key for ``profile``/``version`` or raise a value-free error."""
    account = keychain_account(profile, version)
    if sys.platform != "darwin":
        raise RuntimeEvidenceKeyError("keychain_platform_unsupported")
    try:
        completed = runner(
            [_SECURITY_COMMAND, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"],
            capture_output=True,
            text=False,
            timeout=KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeEvidenceKeyError("keychain_timeout") from None
    except Exception:
        raise RuntimeEvidenceKeyError("keychain_item_unavailable") from None
    if getattr(completed, "returncode", 1) != 0:
        raise RuntimeEvidenceKeyError("keychain_item_unavailable")
    raw = getattr(completed, "stdout", None)
    if not isinstance(raw, bytes):
        raise RuntimeEvidenceKeyError("keychain_key_malformed")
    value = raw.removesuffix(b"\n").removesuffix(b"\r")
    if len(value) < 64:
        raise RuntimeEvidenceKeyError("keychain_key_too_short")
    if _KEY_HEX_RE.fullmatch(value) is None:
        raise RuntimeEvidenceKeyError("keychain_key_malformed")
    return bytes.fromhex(value.decode("ascii"))


class KeychainHmacKeyProvider:
    """Lazy, cached ``KeyProvider`` returning ``(version, key)``; repr never shows the key."""

    def __init__(
        self,
        profile: str,
        version: int,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        keychain_account(profile, version)  # validate naming eagerly; no Keychain access
        self._profile = profile
        self._version = version
        self._runner = runner
        self._lock = threading.Lock()
        self._key: Optional[bytes] = None
        self._error: Optional[str] = None

    def __call__(self) -> Tuple[int, bytes]:
        with self._lock:
            if self._key is None and self._error is None:
                try:
                    self._key = load_keychain_hmac_key(self._profile, self._version, runner=self._runner)
                except RuntimeEvidenceKeyError as exc:
                    self._error = exc.code
            if self._error is not None:
                raise RuntimeEvidenceKeyError(self._error)
            return self._version, self._key  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"KeychainHmacKeyProvider(profile={self._profile!r}, version={self._version}, key=<redacted>)"

    __str__ = __repr__


def build_keychain_key_provider(cfg: Any) -> Optional[KeychainHmacKeyProvider]:
    """Provider for an enabled config with ``hmac_key_source: keychain`` and exactly one profile.

    Returns None (observer stays unconstructed) otherwise; never touches the Keychain.
    """
    try:
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if getattr(cfg, "hmac_key_source", None) != "keychain":
            return None
        profiles = [p for p in (getattr(cfg, "profiles", None) or []) if isinstance(p, str) and p]
        if len(profiles) != 1:
            return None
        return KeychainHmacKeyProvider(profiles[0], getattr(cfg, "hmac_key_version", None))
    except Exception:
        return None
