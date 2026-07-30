"""URL guards for browser navigation (SSRF / local file protection)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_SCHEMES = frozenset({"file", "javascript", "data", "blob", "about", "chrome", "chrome-extension"})


def validate_public_http_url(url: str) -> tuple[bool, str | None]:
    """Return (ok, error). Only http(s) to non-private hosts is allowed."""
    raw = (url or "").strip()
    if not raw:
        return False, "url is required"

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme in _BLOCKED_SCHEMES:
        return False, f"blocked URL scheme: {scheme}"
    if scheme not in {"http", "https"}:
        return False, f"unsupported URL scheme: {scheme or '(none)'}; use http or https"

    host = parsed.hostname
    if not host:
        return False, "url has no hostname"

    host_l = host.lower().rstrip(".")
    if host_l in {"localhost", "metadata.google.internal"} or host_l.endswith(".localhost"):
        return False, f"blocked host: {host}"

    # Literal IP
    try:
        ip = ipaddress.ip_address(host_l)
        if _is_non_public_ip(ip):
            return False, f"blocked private or special-use IP: {host}"
        return True, None
    except ValueError:
        pass

    # Resolve hostname — block if any address is non-public
    try:
        infos = socket.getaddrinfo(host_l, None)
    except socket.gaierror:
        # DNS failure: allow navigate to proceed (page will error); do not fail closed on flaky DNS
        return True, None

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_non_public_ip(ip):
            return False, f"host {host} resolves to blocked address {addr}"

    return True, None


def _is_non_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or (getattr(ip, "is_site_local", False))
    )
