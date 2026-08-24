"""Extract the visible text of a .docx paragraph by paragraph (body, tables, headers, footers),
one paragraph per line, with field codes omitted and their cached results kept. Used by the
verification gates to grep the built thesis.

    python scripts/dump_docx_text.py docs/thesis/submission/thesis.docx > thesis_text.txt
"""

from __future__ import annotations

import sys
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def ptext(p) -> str:
    out = []
    skip = 0
    for e in p.iter():
        tag = etree.QName(e).localname
        ns = etree.QName(e).namespace
        if tag == "fldChar":
            t = e.get(f"{{{W}}}fldCharType")
            if t == "begin":
                skip += 1
            elif t in ("separate", "end"):
                # "end" must also clear the skip: a field with no result (a hidden SEQ reset,
                # for instance) carries no "separate", and without this the rest of the
                # paragraph would silently vanish from the dump the gates read
                skip = max(0, skip - 1)
            continue
        if skip:
            continue
        if tag == "t":
            out.append(e.text or "")
        elif tag == "tab":
            out.append("\t")
        elif tag == "br":
            out.append("\n")
        elif tag == "oMath":
            out.append("{EQUATION: " + "".join(x.text or "" for x in e.iter(f"{{{M}}}t")) + "}")
    return "".join(out)


def main(path: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        doc = etree.fromstring(z.read("word/document.xml"))
        for p in doc.iter(f"{{{W}}}p"):
            if any(etree.QName(a).localname == "oMath" for a in p.iterancestors()):
                continue
            t = ptext(p)
            if t.strip():
                print(t)
        for n in sorted(names):
            if n.startswith("word/header") or n.startswith("word/footer"):
                part = etree.fromstring(z.read(n))
                for p in part.iter(f"{{{W}}}p"):
                    t = ptext(p)
                    if t.strip():
                        print(f"[{n[5:]}] {t}")


if __name__ == "__main__":
    main(sys.argv[1])
