"""Minimal RFC 6455 WebSocket support (stdlib only).

Just enough for OpenEnv's ``/ws`` session protocol: text frames, ping/pong,
close, and fragmented messages. ``WSClient`` is used by the probe;
``server_handshake`` / ``ServerConn`` let the bundled sample environments
speak the same protocol from ``http.server``.
"""
from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WSError(Exception):
    pass


class WSClosed(WSError):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def _recv_exact(read, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = read(n - len(buf))
        if not chunk:
            raise WSClosed("connection closed")
        buf += chunk
    return buf


def _read_frame(read) -> Tuple[bool, int, bytes]:
    b1, b2 = _recv_exact(read, 2)
    fin, opcode = bool(b1 & 0x80), b1 & 0x0F
    masked, length = bool(b2 & 0x80), b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(read, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(read, 8))[0]
    mask = _recv_exact(read, 4) if masked else None
    payload = _recv_exact(read, length) if length else b""
    if mask:
        payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    return fin, opcode, payload


def _frame(opcode: int, payload: bytes, mask: bool) -> bytes:
    head = bytes([0x80 | opcode])
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 65536:
        head += bytes([mbit | 126]) + struct.pack("!H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack("!Q", n)
    if mask:
        key = os.urandom(4)
        return head + key + bytes(c ^ key[i % 4] for i, c in enumerate(payload))
    return head + payload


def _read_message(read, write_frame) -> Optional[str]:
    """Return the next text message, or None when the peer closes."""
    parts, op0 = [], None
    while True:
        fin, op, payload = _read_frame(read)
        if op == OP_PING:
            write_frame(OP_PONG, payload)
            continue
        if op == OP_PONG:
            continue
        if op == OP_CLOSE:
            try:
                write_frame(OP_CLOSE, payload[:2])
            except OSError:
                pass
            return None
        if op in (OP_TEXT, OP_BIN):
            op0, parts = op, [payload]
        elif op == OP_CONT:
            parts.append(payload)
        if fin:
            return b"".join(parts).decode("utf-8", "replace")


class WSClient:
    def __init__(self, url: str, timeout: float = 10.0, headers: Optional[Dict[str, str]] = None):
        u = urlsplit(url)
        secure = u.scheme in ("wss", "https")
        host, port = u.hostname or "localhost", u.port or (443 if secure else 80)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        sock = socket.create_connection((host, port), timeout=timeout)
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        self.sock = sock
        self.rfile = sock.makefile("rb")
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ] + [f"{k}: {v}" for k, v in (headers or {}).items()]
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        status = self.rfile.readline().decode("latin-1").strip()
        resp_headers = {}
        while True:
            line = self.rfile.readline().decode("latin-1")
            if line in ("\r\n", "\n", ""):
                break
            k, _, v = line.partition(":")
            resp_headers[k.strip().lower()] = v.strip()
        parts = status.split()
        if len(parts) < 2 or parts[1] != "101":
            self.close(send=False)
            raise WSError(f"handshake failed: {status or 'no status line'}")
        if resp_headers.get("sec-websocket-accept") != accept_key(key):
            self.close(send=False)
            raise WSError("handshake failed: bad Sec-WebSocket-Accept")
        self.closed = False

    def _write(self, op: int, payload: bytes) -> None:
        self.sock.sendall(_frame(op, payload, mask=True))

    def send_text(self, text: str) -> None:
        self._write(OP_TEXT, text.encode())

    def recv_text(self) -> Optional[str]:
        msg = _read_message(self.rfile.read, self._write)
        if msg is None:
            self.closed = True
        return msg

    def close(self, send: bool = True) -> None:
        try:
            if send:
                self._write(OP_CLOSE, struct.pack("!H", 1000))
        except OSError:
            pass
        for f in (self.rfile, self.sock):
            try:
                f.close()
            except OSError:
                pass
        self.closed = True


# --- server side (used by the sample envs) --------------------------------
def server_handshake(handler) -> bool:
    """Complete an upgrade inside a BaseHTTPRequestHandler.do_GET."""
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key or "websocket" not in (handler.headers.get("Upgrade") or "").lower():
        return False
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept_key(key))
    handler.end_headers()
    handler.wfile.flush()
    return True


class ServerConn:
    def __init__(self, handler):
        self.h = handler

    def _write(self, op: int, payload: bytes) -> None:
        self.h.wfile.write(_frame(op, payload, mask=False))
        self.h.wfile.flush()

    def recv_text(self) -> Optional[str]:
        return _read_message(self.h.rfile.read, self._write)

    def send_text(self, text: str) -> None:
        self._write(OP_TEXT, text.encode())

    def drop(self) -> None:
        """Abruptly drop the connection (used by the broken sample)."""
        self.h.close_connection = True
        try:
            self.h.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
