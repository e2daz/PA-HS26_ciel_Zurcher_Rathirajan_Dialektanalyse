"""
Rewrite "%" as "Prozent" in the DIT and DAT transcript columns of every transcript TSV (DIAL-32).

Whisper (DIT) writes "51%", while the FHNW model (DAT) writes "51 Prozent", so the two
transcripts read differently in the side-by-side tables of the web tool. This normalizes the
committed transcripts in place with preprocessing.spell_out_percent.

Only the transcript column is touched: every other field, the BOM and the line endings stay
byte-identical, so the git diff shows just the changed transcripts. The reference sentences
(`sentence`) are left as they are in the corpora. Running the script twice changes nothing.

Run: python experiments/transcriptions/spell_out_percent.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from preprocessing import spell_out_percent  # noqa: E402

TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
TRANSCRIPT_COLUMNS = ("whisper_large_v2_transcript", "fhnw_transcript")
BOM = "﻿"


def rewrite_file(path: Path) -> int:
    """Spell out "%" in the file's transcript column; return the number of changed rows."""
    raw = path.read_bytes().decode("utf-8")
    has_bom = raw.startswith(BOM)
    lines = raw.removeprefix(BOM).split("\n")  # "\r" stays on each line, so CRLF is kept

    header = lines[0].rstrip("\r").split("\t")
    column = next(c for c in TRANSCRIPT_COLUMNS if c in header)
    idx = header.index(column)

    changed = 0
    for n, line in enumerate(lines[1:], start=1):
        if "%" not in line:
            continue
        fields = line.split("\t")
        if len(fields) != len(header):
            raise ValueError(f"{path}:{n + 1}: expected {len(header)} fields, got {len(fields)}")
        new_value = spell_out_percent(fields[idx])
        if new_value != fields[idx]:
            fields[idx] = new_value
            lines[n] = "\t".join(fields)
            changed += 1

    if changed:
        path.write_bytes(((BOM if has_bom else "") + "\n".join(lines)).encode("utf-8"))
    return changed


def main() -> None:
    for path in sorted(TRANSCRIPTS_DIR.rglob("*.tsv")):
        print(f"{rewrite_file(path):>6} rows changed  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
