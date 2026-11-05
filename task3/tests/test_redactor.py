# Tests for the streaming redactor: correct redaction, no leaks at any chunk boundary, prompt output,
# and a bounded hold-back buffer.

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redactor import REDACTED, StreamRedactor, luhn_valid, redact_text

SECRETS = ["jane.doe+work@mail.example.co.uk", "123-45-6789", "4111 1111 1111 1111", "5500-0000-0000-0004"]
SAMPLE = (
    "Hi! Reach me at jane.doe+work@mail.example.co.uk today. My SSN is 123-45-6789 and my card is "
    "4111 1111 1111 1111, backup 5500-0000-0000-0004. Order 1234567812345678 shipped in 2024."
)


def stream(text: str, sizes) -> str:
    r = StreamRedactor()
    out, i = [], 0
    for size in sizes:
        if i >= len(text):
            break
        out.append(r.feed(text[i:i + size]))
        i += size
    if i < len(text):
        out.append(r.feed(text[i:]))
    out.append(r.flush())
    return "".join(out)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail bob@example.com now", f"mail {REDACTED} now"),
        ("a.b-c_d%e+f@sub.domain.example.io", REDACTED),
        ("ssn 123-45-6789.", f"ssn {REDACTED}."),
        ("ssn 123 45 6789 ok", f"ssn {REDACTED} ok"),
        ("visa 4111111111111111", f"visa {REDACTED}"),
        ("amex 3782 822463 10005", f"amex {REDACTED}"),
        ("mc 5500-0000-0000-0004!", f"mc {REDACTED}!"),
        ("two: a@b.co and c@d.org", f"two: {REDACTED} and {REDACTED}"),
        ("(john@example.com)", f"({REDACTED})"),
    ],
)
def test_redacts(text, expected):
    assert redact_text(text) == expected
    assert stream(text, [1] * len(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Order 1234567812345678 is a Luhn-invalid number",
        "SSN 000-12-3456 and 666-12-3456 and 900-12-3456 are invalid",
        "SSN 123-00-4567 and 123-45-0000 are invalid",
        "mixed separators 123-45 6789",
        "phone 555-123-4567",
        "year 2024, version 3.11.2, price $1,299.99",
        "user@localhost has no TLD",
        "not an email: @example.com or user@",
        "digits ١٢٣-٤٥-٦٧٨٩ are not ASCII",
        "123-45-67890 is too long",
        "",
    ],
)
def test_leaves_non_pii_alone(text):
    assert redact_text(text) == text
    assert stream(text, [1] * len(text)) == text


def test_every_two_way_split_matches_batch():
    expected = redact_text(SAMPLE)
    for cut in range(len(SAMPLE) + 1):
        assert stream(SAMPLE, [cut, len(SAMPLE)]) == expected, cut


def test_every_three_way_split_around_secrets():
    text = "x jane@example.com y 123-45-6789 z 4111111111111111 w"
    expected = redact_text(text)
    for a in range(len(text) + 1):
        for b in range(a, len(text) + 1):
            assert stream(text, [a, b - a, len(text)]) == expected, (a, b)


@pytest.mark.parametrize("seed", range(200))
def test_random_chunking_matches_batch_and_never_leaks(seed):
    rnd = random.Random(seed)
    sizes = [rnd.randint(1, 12) for _ in range(len(SAMPLE))]
    out = stream(SAMPLE, sizes)
    assert out == redact_text(SAMPLE)
    for secret in SECRETS:
        assert secret not in out
        assert secret.replace(" ", "").replace("-", "") not in out.replace(" ", "").replace("-", "")


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("378282246310005")
    assert not luhn_valid("4111111111111112")


def test_plain_text_is_released_immediately():
    r = StreamRedactor()
    assert r.feed("Hello there, friend. ") == "Hello there, friend. "
    assert r.pending == ""
    assert r.feed("The answer is 42 because") == "The answer is 42 "
    assert r.pending == "because"


def test_possible_match_is_held_then_resolved():
    r = StreamRedactor()
    assert r.feed("mail john@ex") == "mail "
    assert r.feed("ample.com") == ""
    assert r.feed(".au thanks") == f"{REDACTED} "
    assert r.flush() == "thanks"


@pytest.mark.parametrize("prefix, rest", [
    ("id " + "7" * 70, "4111111111111111 end"),
    ("tag " + "q" * 70, "bob@example.com end"),
])
def test_lookbehind_sees_already_emitted_text(prefix, rest):
    r = StreamRedactor()
    first = r.feed(prefix)
    assert first == prefix
    assert first + r.feed(rest) + r.flush() == redact_text(prefix + rest) == prefix + rest


def test_flush_redacts_match_at_end_of_stream():
    r = StreamRedactor()
    assert r.feed("send to bob@example.com") == "send to "
    assert r.flush() == REDACTED
    assert r.flush() == ""


def test_hold_back_stays_small_on_long_streams():
    rnd = random.Random(1)
    words = ["lorem", "ipsum", "a@b.co", "123-45-6789", "4111111111111111", "x" * 300, "2024", "foo.bar"]
    r = StreamRedactor()
    longest = 0
    text = " ".join(rnd.choice(words) for _ in range(20_000))
    i = 0
    while i < len(text):
        n = rnd.randint(1, 20)
        r.feed(text[i:i + n])
        i += n
        longest = max(longest, len(r.pending))
    assert longest < 400


def test_long_run_without_spaces_is_not_held():
    r = StreamRedactor()
    blob = "A" * 5000
    assert r.feed(blob) == blob
