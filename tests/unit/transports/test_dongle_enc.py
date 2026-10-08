"""Real TLS-PSK loopback coverage for the ENC connection and reply layout."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import struct
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pylxpweb.transports.dongle import (
    MODBUS_READ_HOLDING,
    PACKET_PREFIX,
    TCP_FUNC_TRANSLATED,
    DongleTransport,
    compute_crc16,
)
from pylxpweb.transports.exceptions import TransportConnectionError

pytestmark = pytest.mark.skipif(not ssl.HAS_PSK, reason="OpenSSL lacks TLS-PSK support")

DONGLE = "BJ12345678"
INVERTER = "1234567890"
# Independent HMAC-SHA256(b"LuxPowerTek!", b"BJ12345678")[:16] vector.
PSK = bytes.fromhex("caa11b1a015da78f54be6ad437d567dd")


class EncServer:
    """TLS-only peer with protocol-5 replies and a known synthetic PSK."""

    def __init__(self) -> None:
        self.port = 0
        self.requests: list[tuple[int, int, int]] = []
        self.connections = 0
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("DHE-PSK-AES128-GCM-SHA256")
        # Public RFC 7919 group, generated with:
        # openssl genpkey -genparam -algorithm DH -pkeyopt group:ffdhe2048
        context.load_dh_params(Path(__file__).parents[2] / "fixtures" / "enc-dhparams.pem")
        context.set_psk_server_callback(
            lambda identity: PSK if identity == "Client_identity" else b""
        )
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0, ssl=context, ssl_handshake_timeout=0.2
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def close_connections(self) -> None:
        for writer in list(self._writers):
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        await self.close_connections()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._writers.add(writer)
        try:
            while True:
                header = await reader.readexactly(6)
                assert header[:2] == PACKET_PREFIX
                frame = header + await reader.readexactly(struct.unpack("<H", header[4:6])[0])
                assert frame[8:18] == DONGLE.encode("ascii")
                assert frame[22:32] == INVERTER.encode("ascii")
                assert compute_crc16(frame[20:-2]) == struct.unpack("<H", frame[-2:])[0]
                function = frame[21]
                register, count = struct.unpack("<HH", frame[32:36])
                self.requests.append((function, register, count))
                # ENC V3.03 was observed returning 80 holdings for a request
                # of 40. Keep the synthetic wire data free of device captures.
                returned_count = 80 if function == MODBUS_READ_HOLDING and count == 40 else count
                values = list(range(register, register + returned_count))
                data = bytes([0, function]) + INVERTER.encode("ascii")
                data += struct.pack("<H", register) + bytes([2 * len(values)])
                data += struct.pack(f"<{len(values)}H", *values)
                response = PACKET_PREFIX + struct.pack(
                    "<HHBB", 5, 16 + len(data), 1, TCP_FUNC_TRANSLATED
                )
                response += DONGLE.encode("ascii") + struct.pack("<H", len(data) + 2)
                response += data + struct.pack("<H", compute_crc16(data))
                # Fragment a full TLS application frame across writes.
                writer.write(response[:11])
                await writer.drain()
                await asyncio.sleep(0)
                writer.write(response[11:])
                await writer.drain()
        except (asyncio.IncompleteReadError, OSError):
            pass
        finally:
            self._writers.discard(writer)
            writer.close()


@pytest.fixture
async def enc_server() -> AsyncIterator[EncServer]:
    server = EncServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_cold_auto_probe_negotiates_tls_and_reconnects(enc_server: EncServer) -> None:
    """A cold health check recovers ENC, then remembers TLS on a fresh dial."""
    transport = DongleTransport("127.0.0.1", DONGLE, INVERTER, port=enc_server.port)
    try:
        assert await transport.check_link() is True
        assert transport._ssl_proven is True
        assert transport._ssl_active is True
        assert transport._link_probe_active is False
        assert transport._link_probe_tls is False
        await enc_server.close_connections()
        await transport._force_reconnect()
        assert await transport.check_link() is True
        assert transport._ssl_active is True
        assert enc_server.connections == 2
    finally:
        await transport.async_shutdown()


@pytest.mark.asyncio
async def test_enc_holding_reply_stays_inside_requested_range(enc_server: EncServer) -> None:
    """Consume a 197-byte reply but expose only the 40 requested holdings."""
    transport = DongleTransport("127.0.0.1", DONGLE, INVERTER, port=enc_server.port, use_ssl=True)
    try:
        assert await transport.check_link() is True
        assert await transport.read_parameters(0, 40) == dict(enumerate(range(40)))
        assert enc_server.connections == 1
        assert enc_server.requests[-1] == (MODBUS_READ_HOLDING, 0, 40)
    finally:
        await transport.async_shutdown()


@pytest.mark.asyncio
async def test_forced_plaintext_does_not_probe_tls(enc_server: EncServer) -> None:
    transport = DongleTransport("127.0.0.1", DONGLE, INVERTER, port=enc_server.port, use_ssl=False)
    try:
        assert await transport.check_link() is False
        assert transport._ssl_proven is False
        assert enc_server.connections == 0
    finally:
        await transport.async_shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_auto_probe_respects_negative_cache_expiry(
    enc_server: EncServer, expired: bool
) -> None:
    """Honor the negative TTL, but recover after a firmware upgrade and expiry."""
    transport = DongleTransport("127.0.0.1", DONGLE, INVERTER, port=enc_server.port)
    transport._ssl_unsupported_until = time.monotonic() + (-1 if expired else 60)
    try:
        assert await transport.check_link() is expired
        assert transport._ssl_proven is expired
        assert enc_server.connections == int(expired)
        if expired:
            assert transport._ssl_unsupported_until is None
    finally:
        await transport.async_shutdown()


@pytest.mark.asyncio
async def test_incorrect_serial_rejects_forced_tls(enc_server: EncServer) -> None:
    transport = DongleTransport(
        "127.0.0.1",
        "BJ99999999",
        INVERTER,
        port=enc_server.port,
        use_ssl=True,
        connection_retries=1,
    )
    try:
        with pytest.raises(TransportConnectionError):
            await transport.connect()
        assert transport._ssl_unsupported_until is None
        assert transport._ssl_active is False
        assert enc_server.connections == 0
    finally:
        await transport.async_shutdown()
