"""Opt-in local hardware reads; no cloud credentials or control operations.

Set PYLXPWEB_DONGLE_HOST, PYLXPWEB_DONGLE_SERIAL, PYLXPWEB_INVERTER_SERIAL,
and optionally PYLXPWEB_DONGLE_TLS=auto|true|false and PYLXPWEB_DONGLE_PORT.
Run: pytest tests/integration/test_live_dongle.py -m integration -s
Use only one local client per physical dongle while testing.
"""

from __future__ import annotations

import os

import pytest

from pylxpweb.transports.dongle import DongleTransport

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_read_only_live_dongle() -> None:
    host = os.getenv("PYLXPWEB_DONGLE_HOST")
    dongle = os.getenv("PYLXPWEB_DONGLE_SERIAL")
    inverter = os.getenv("PYLXPWEB_INVERTER_SERIAL")
    if not (host and dongle and inverter):
        pytest.skip("Set the PYLXPWEB_DONGLE_* hardware test environment variables")
    mode = os.getenv("PYLXPWEB_DONGLE_TLS", "auto")
    assert mode in {"auto", "true", "false"}
    transport = DongleTransport(
        host,
        dongle,
        inverter,
        port=int(os.getenv("PYLXPWEB_DONGLE_PORT", "8000")),
        use_ssl={"auto": None, "true": True, "false": False}[mode],
    )
    try:
        assert await transport.check_link() is True
        holdings = await transport.read_parameters(0, 40)
        assert set(holdings) == set(range(40))
        device_type = await transport.read_device_type()
        assert device_type > 0
        print(f"TLS={transport._ssl_active}; 40 holdings and device type decoded")
    finally:
        await transport.async_shutdown()
