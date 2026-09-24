import struct

from tests.conftest import make_registry
from token_daddy.estimate import (
    audio_seconds,
    estimate_input_tokens,
    pdf_pages,
    reservation_cost,
)
from token_daddy.settings import EstimationConfig, Family, Reservation
from token_daddy.types import File, Request, Text

CFG = EstimationConfig(safety_multiplier=1.0)


def fake_pdf(pages: int) -> bytes:
    body = b"".join(b"<< /Type /Page /Parent 2 0 R >>" for _ in range(pages))
    return b"%PDF-1.7 << /Type /Pages /Count " + str(pages).encode() + b" >>" + body


def wav(seconds: float, byte_rate: int = 32_000) -> bytes:
    header = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 16 + struct.pack("<I", byte_rate) + b"\0" * 12
    return header + b"\0" * int(seconds * byte_rate)


def test_pdf_pages_are_counted_without_the_page_tree():
    assert pdf_pages(fake_pdf(7)) == 7


def test_wav_duration_comes_from_the_header():
    assert abs(audio_seconds(wav(3.0), fallback_bytes_per_second=1) - 3.0) < 0.01


def test_attachments_are_priced_per_family():
    req = Request(model="m", parts=[Text("a" * 350), File(fake_pdf(2), "application/pdf")])
    assert estimate_input_tokens(req, Family.GPT, CFG) == 100 + 2 * 1500
    assert estimate_input_tokens(req, Family.GEMINI, CFG) == 100 + 2 * 560


def test_safety_multiplier_rounds_up():
    req = Request(model="m", parts=[Text("a" * 350)])
    assert estimate_input_tokens(req, Family.GPT, EstimationConfig(safety_multiplier=1.15)) == 115


def test_strict_reserves_the_full_output_cap_estimate_reserves_a_share():
    registry = make_registry()
    dep = registry.model("m").deployments[0]
    req = Request(model="m", parts=[Text("x")], max_output_tokens=8000)
    assert reservation_cost(req, dep, CFG, input_tokens=1000).output_tokens == 8000

    loose = registry.model("m").deployments[0].__class__(
        **{**dep.__dict__, "reservation": Reservation.ESTIMATE}
    )
    cost = reservation_cost(req, loose, CFG, input_tokens=1000)
    assert cost.output_tokens == 400  # 40% of input
    assert reservation_cost(req, loose, CFG, input_tokens=10).output_tokens == 256  # floor
