#!/usr/bin/env python3
"""Guard against the PW#11 Windows PowerShell 5.1 parser-failure regression.

Background
----------
Windows PowerShell 5.1 (the default shell on Windows 10 / 11) reads
BOM-less ``.ps1`` files as Windows-1252, not UTF-8. PowerShell 7+ tolerates
BOM-less UTF-8, which is why dev-box runs (where ``pwsh`` is typically
the active shell) never surfaced the issue locally.

In May 2026 the three installer ``.ps1`` files were saved as BOM-less
UTF-8 containing em-dashes, right-arrows, and section signs. PowerShell
5.1 mis-decoded each multi-byte UTF-8 sequence into Windows-1252 garbage
(``E2 80 94`` em-dash rendered as ``â€"``), breaking quote/brace pairing
and producing parser errors before the script body ran. Fixed at PW
``077caf2`` by prepending UTF-8 BOM (``EF BB BF``) to every file.

This script enforces the rule going forward: any ``installer/*.ps1``
file that contains a non-ASCII byte MUST start with the UTF-8 BOM.
Files containing only ASCII bytes are exempt (PowerShell 5.1 handles
pure ASCII without ambiguity).

Usage
-----
    python3 installer/check_ps1_bom.py

Exits 0 if all .ps1 files are clean; exits 1 with a list of offenders
otherwise. Designed to run as a release.yml preflight job.

Refs: PW issue #11, OQ-2026-05-27-7.
"""

from __future__ import annotations

import sys
from pathlib import Path

BOM = b"\xef\xbb\xbf"


def has_non_ascii(data: bytes) -> bool:
    """Return True if any byte in ``data`` is >= 0x80."""
    return any(b >= 0x80 for b in data)


def check_file(path: Path) -> str | None:
    """Return a one-line failure message, or None if the file is clean."""
    data = path.read_bytes()
    if not has_non_ascii(data):
        return None
    if data.startswith(BOM):
        return None
    # Find the first offending byte for a precise diagnostic.
    offset = next(i for i, b in enumerate(data) if b >= 0x80)
    snippet = data[max(0, offset - 8) : offset + 8].decode(
        "utf-8", errors="replace"
    )
    return (
        f"{path}: non-ASCII byte 0x{data[offset]:02X} at offset {offset} "
        f"(context: {snippet!r}) — file is missing UTF-8 BOM. "
        f"Fix: prepend 0xEF 0xBB 0xBF to the file."
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    installer_dir = repo_root / "installer"
    if not installer_dir.is_dir():
        print(
            f"ERROR: installer/ directory not found at {installer_dir}",
            file=sys.stderr,
        )
        return 2

    ps1_files = sorted(installer_dir.glob("*.ps1"))
    if not ps1_files:
        # No .ps1 files to check — nothing can go wrong.
        return 0

    failures = []
    for path in ps1_files:
        msg = check_file(path)
        if msg is not None:
            failures.append(msg)

    if failures:
        print(
            f"FAIL: {len(failures)} of {len(ps1_files)} installer/*.ps1 files "
            f"contain non-ASCII bytes without a UTF-8 BOM.",
            file=sys.stderr,
        )
        print(
            "Windows PowerShell 5.1 (Windows 10 / 11 default shell) reads "
            "BOM-less UTF-8 as Windows-1252 and mis-decodes multi-byte "
            "sequences, breaking the parser before any code runs.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "To fix: open the file in an editor that supports 'UTF-8 with "
            "BOM', save with that encoding. Or prepend the BOM directly:",
            file=sys.stderr,
        )
        print(
            "  printf '\\xef\\xbb\\xbf' | cat - file.ps1 > tmp && mv tmp file.ps1",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {len(ps1_files)} installer/*.ps1 files checked, all "
        f"non-ASCII-containing files start with UTF-8 BOM."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
