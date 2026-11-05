# Streaming PII redactor. Replaces emails, SSNs and credit card numbers with [REDACTED] in text that
# arrives in chunks, holding back only the short tail that could still turn into a match.

from __future__ import annotations

import regex

REDACTED = "[REDACTED]"

_CARD = r"(?P<card>(?<![0-9-])[0-9](?:[ -]?[0-9]){12,18}(?![0-9]))"
_SSN = r"(?P<ssn>(?<![0-9-])(?P<area>[0-9]{3})(?P<sep>[- ])(?P<group>[0-9]{2})(?P=sep)(?P<serial>[0-9]{4})(?![0-9-]))"
_EMAIL = (
    r"(?P<email>(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@"
    r"[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24})"
)
PATTERN = regex.compile("|".join([_CARD, _SSN, _EMAIL]))
SEPARATORS = regex.compile(r"[ -]")


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def is_sensitive(match: regex.Match) -> bool:
    if match.group("card"):
        digits = SEPARATORS.sub("", match.group("card"))
        return 13 <= len(digits) <= 19 and luhn_valid(digits)
    if match.group("ssn"):
        area, group, serial = match.group("area"), match.group("group"), match.group("serial")
        return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"
    return True


class StreamRedactor:
    def __init__(self) -> None:
        self._pending = ""
        self._context = ""

    @property
    def pending(self) -> str:
        return self._pending

    def feed(self, text: str) -> str:
        return self._drain(self._pending + text, final=False)

    def flush(self) -> str:
        return self._drain(self._pending, final=True)

    def _drain(self, buf: str, final: bool) -> str:
        scan = self._context + buf
        start = len(self._context)
        emit_from = start
        hold_from = len(scan)
        out = []
        for m in PATTERN.finditer(scan, pos=start, partial=not final):
            if not final and (m.partial or PATTERN.fullmatch(scan, pos=m.start(), partial=True)):
                hold_from = m.start()
                break
            out.append(scan[emit_from : m.start()])
            out.append(REDACTED if is_sensitive(m) else m.group())
            emit_from = m.end()
        out.append(scan[emit_from:hold_from])
        self._pending = scan[hold_from:]
        self._context = scan[hold_from - 1 : hold_from] if hold_from else ""
        return "".join(out)


def redact_text(text: str) -> str:
    return StreamRedactor()._drain(text, final=True)
