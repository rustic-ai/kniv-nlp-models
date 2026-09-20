"""Azure AD token auth for the annotator endpoints.

Preferred over a stored API key. The token is minted on demand from the
identity already signed in to the Azure CLI, lives for about an hour, and is
never written to disk or to the run log. Nothing in the repo or in
``annotators.yaml`` then holds a long-lived secret.

Mirrors how the same two Azure resources are reached elsewhere on this
machine (``~/.codex/config.toml`` uses the same ``az account
get-access-token`` call against the same scope).
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/"
_AZ = "az"
# Refresh this long before stated expiry so a token cannot lapse mid-flight
# on a long fan-out.
_SKEW_SECONDS = 300

_token: str | None = None
_expires_at: float = 0.0
_lock: asyncio.Lock | None = None


def _fetch() -> tuple[str, float]:
    proc = subprocess.run(
        [_AZ, "account", "get-access-token",
         "--resource", COGNITIVE_SERVICES_SCOPE, "--output", "json"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        # stderr can echo account detail; surface only the actionable part.
        raise RuntimeError(
            "az account get-access-token failed — run `az login` first "
            f"(exit {proc.returncode})"
        )
    payload = json.loads(proc.stdout)
    token = payload["accessToken"]
    # expires_on is epoch seconds; expiresOn is local-time text. Prefer the
    # unambiguous one and fall back to a conservative fixed lifetime.
    expiry = payload.get("expires_on")
    expires_at = float(expiry) if expiry else time.time() + 3000
    return token, expires_at


async def get_token() -> str:
    """Return a valid bearer token, refreshing it at most once at a time."""
    global _token, _expires_at, _lock
    if _lock is None:
        _lock = asyncio.Lock()
    if _token and time.time() < _expires_at - _SKEW_SECONDS:
        return _token
    async with _lock:
        if _token and time.time() < _expires_at - _SKEW_SECONDS:
            return _token
        _token, _expires_at = await asyncio.to_thread(_fetch)
        return _token
