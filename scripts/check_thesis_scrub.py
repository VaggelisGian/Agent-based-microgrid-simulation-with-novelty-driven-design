"""Scrub checks for the thesis sources and the built .docx.

Checks (a) the markdown sources of the submission and (b) every part of a .docx for: terms from
the scrub term list, em and en dashes, emoji and other symbol-plane characters, comments and
tracked changes, and tool-identifying metadata in the document properties and in the embedded
PNG and SVG payloads.

    python scripts/check_thesis_scrub.py --src <src dir> --docx <file.docx> \
        --terms docs/verification/scrub_terms.txt --author "<expected author name>"

The term list is a separate file, one "section: term" per line, so that this script carries no
vendor or tooling names of its own. Its default location is docs/verification/scrub_terms.txt.

Exit code 1 if anything is found, if an input is missing, or if there is nothing to check.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

DEFAULT_TERMS = Path("docs/verification/scrub_terms.txt")
SKIP_SOURCES = ("FORMAT.md", "WRITER_BRIEF.md")

DASHES = {"–": "EN DASH", "—": "EM DASH", "‒": "FIGURE DASH", "―": "HORIZONTAL BAR"}
ALLOWED_SYMBOLS = set("·×÷±≤≥≈≠∞√∑∏∂∫∈∉∀∃→←↔°µ€£§¶†‡•…‹›«»“”‘’„‚′″‰∙−")
ARROWS = "→←↔"


def load_terms(path: Path) -> dict[str, list[str]]:
    """Read the term list. Missing or empty file is a hard error: a scrub gate that checks
    nothing must never report success."""
    if not path.is_file():
        sys.exit(f"term list not found: {path} (pass --terms explicitly)")
    terms: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            sys.exit(f"malformed term line in {path}: {line!r}")
        section, term = line.split(":", 1)
        term = term.strip().lower()
        if term:
            terms.setdefault(section.strip().lower(), []).append(term)
    if not terms:
        sys.exit(f"term list {path} holds no terms")
    return terms


def term_pattern(term: str) -> str:
    """Substring match, but with a word boundary wherever the term's own edge is alphanumeric,
    so a short term cannot fire inside an unrelated identifier."""
    pat = re.escape(term)
    if term[0].isalnum():
        pat = r"(?<![0-9A-Za-zͰ-Ͽἀ-῿])" + pat
    if term[-1].isalnum():
        pat = pat + r"(?![0-9A-Za-zͰ-Ͽἀ-῿])"
    return pat


def scan_text(text: str, where: str, terms: dict[str, list[str]], allow_arrow: bool = False) -> list[str]:
    hits = []
    low = text.lower()
    for section, words in terms.items():
        for w in words:
            for m in re.finditer(term_pattern(w), low):
                ctx = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
                hits.append(f"{where}: {section} term ({len(w)} chars): ...{ctx}...")
    for ch, name in DASHES.items():
        for m in re.finditer(ch, text):
            ctx = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
            hits.append(f"{where}: {name}: ...{ctx}...")
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp < 0x2000:
            continue
        if ch in ALLOWED_SYMBOLS or ch in DASHES:
            continue
        if allow_arrow and ch in ARROWS:
            continue
        cat = unicodedata.category(ch)
        if cp >= 0x1F000 or cat in ("So", "Sk") or 0x2600 <= cp <= 0x27BF or 0x2190 <= cp <= 0x21FF \
                or 0x2700 <= cp <= 0x27BF or 0xFE00 <= cp <= 0xFE0F:
            ctx = text[max(0, i - 20): i + 20].replace("\n", " ")
            hits.append(f"{where}: symbol/emoji U+{cp:04X} ({unicodedata.name(ch, '?')}): ...{ctx}...")
    return hits


def check_sources(src: Path, terms: dict[str, list[str]]) -> list[str]:
    if not src.is_dir():
        sys.exit(f"source directory not found: {src}")
    files = [f for f in sorted(src.glob("*.md")) if f.name not in SKIP_SOURCES]
    if not files:
        sys.exit(f"no markdown sources under {src}")
    hits = []
    for f in files:
        hits += scan_text(f.read_text(encoding="utf-8"), f.name, terms)
    print(f"  scanned {len(files)} source file(s) under {src}")
    return hits


def png_metadata_hits(name: str, data: bytes) -> list[str]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return []
    hits, pos = [], 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        if ctype in (b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf"):
            payload = data[pos + 8:pos + 8 + min(length, 80)]
            hits.append(f"{name}: PNG {ctype.decode('ascii', 'replace')} chunk present: {payload!r}")
        pos += 12 + length
    return hits


def check_docx(docx: Path, terms: dict[str, list[str]], author: str | None) -> list[str]:
    if not docx.is_file():
        sys.exit(f"docx not found: {docx}")
    hits = []
    with zipfile.ZipFile(docx) as z:
        names = z.namelist()
        for n in names:
            if n.startswith("word/comments") or n.startswith("word/people"):
                hits.append(f"{n}: comments/people part present")
            if n.endswith(".xml") or n.endswith(".rels"):
                raw = z.read(n).decode("utf-8", errors="replace")
                for marker in ("<w:ins ", "<w:ins>", "<w:del ", "<w:del>", "<w:rPrChange", "<w:pPrChange",
                               "<w:moveFrom", "<w:moveTo", "<w:sectPrChange", "<w:tblPrChange"):
                    if marker in raw:
                        hits.append(f"{n}: revision markup {marker.strip('<')} present")
                if "<w:commentRangeStart" in raw or "<w:commentReference" in raw:
                    hits.append(f"{n}: comment anchors present")
                # the whole XML is scanned, not only the text: attribution can hide in an attribute
                hits += scan_text(re.sub(r"<[^>]+>", " ", raw), n, terms)
                hits += scan_text(raw, f"{n} (raw xml)", terms, allow_arrow=True)
            elif n.endswith(".svg"):
                raw = z.read(n).decode("utf-8", errors="replace")
                hits += scan_text(raw, n, terms, allow_arrow=True)
                if "<metadata" in raw.lower():
                    hits.append(f"{n}: svg metadata element present (generator, creator or build date)")
            elif n.endswith(".png"):
                hits += png_metadata_hits(n, z.read(n))
        core = z.read("docProps/core.xml").decode("utf-8")
        app = z.read("docProps/app.xml").decode("utf-8")

        def prop(xml: str, tag: str) -> str | None:
            m = re.search(rf"<{tag}\s*/>", xml)
            if m:
                return ""
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S)
            return m.group(1) if m else None

        for tag in ("dc:creator", "cp:lastModifiedBy"):
            value = prop(core, tag)
            print(f"  {tag} = {value if value is not None else '(absent)'}")
            if author is not None and (value or "") != author:
                hits.append(f"docProps/core.xml: {tag} is {value!r}, expected {author!r}")
        for tag, expected in (("Application", "Microsoft Office Word"), ("Company", ""), ("Manager", None)):
            value = prop(app, tag)
            print(f"  app:{tag} = {value if value is not None else '(absent)'}")
            if expected is not None and (value or "") != expected:
                hits.append(f"docProps/app.xml: {tag} is {value!r}, expected {expected!r}")
        print(f"  app:Template = {prop(app, 'Template')}")
        if "docProps/custom.xml" in names:
            hits.append("docProps/custom.xml present: " + z.read("docProps/custom.xml").decode("utf-8")[:200])
        print(f"  scanned {len(names)} docx part(s)")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="directory of submission markdown sources")
    ap.add_argument("--docx", help="built .docx to inspect")
    ap.add_argument("--terms", default=str(DEFAULT_TERMS), help="term list file")
    ap.add_argument("--author", help="expected docProps creator and lastModifiedBy")
    args = ap.parse_args()
    if not args.src and not args.docx:
        ap.error("nothing to check: pass --src, --docx, or both")
    terms = load_terms(Path(args.terms))
    hits = []
    if args.src:
        hits += check_sources(Path(args.src), terms)
    if args.docx:
        hits += check_docx(Path(args.docx), terms, args.author)
    for h in hits:
        print(h)
    print(f"{len(hits)} hit(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
