"""Dump a .docx as a paragraph-by-paragraph structure listing (style, numbering, text, fields,
section breaks with their header/footer targets), so the built thesis can be checked against the
template and the regulations without opening Word.

    python scripts/dump_docx_structure.py docs/thesis/submission/thesis.docx > out.txt
"""

from __future__ import annotations

import sys
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W, "r": R}


def ptext(p):
    out = []
    for e in p.iter():
        tag = etree.QName(e).localname
        if tag == "t":
            out.append(e.text or "")
        elif tag == "tab":
            out.append("\t")
        elif tag == "br":
            out.append(" | ")
        elif tag == "instrText":
            out.append("{" + (e.text or "").strip() + "}")
        elif tag == "drawing":
            out.append("{DRAWING}")
        elif tag == "oMath":
            out.append("{EQUATION}")
    return "".join(out)


def main(path):
    sys.stdout.reconfigure(encoding="utf-8")
    with zipfile.ZipFile(path) as z:
        doc = etree.fromstring(z.read("word/document.xml"))
        rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
        relmap = {r.get("Id"): r.get("Target") for r in rels}
        hf_text = {}
        for n in z.namelist():
            if n.startswith("word/header") or n.startswith("word/footer"):
                t = etree.fromstring(z.read(n))
                hf_text[n[5:]] = " // ".join(ptext(p) for p in t.findall(".//w:p", NS))
        settings = z.read("word/settings.xml").decode("utf-8")
        print("evenAndOddHeaders:", "<w:evenAndOddHeaders" in settings)
    body = doc.find("w:body", NS)
    i = 0
    for child in body:
        i += 1
        tag = etree.QName(child).localname
        if tag == "p":
            ppr = child.find("w:pPr", NS)
            style = ppr.find("w:pStyle", NS).get(f"{{{W}}}val") if ppr is not None and ppr.find("w:pStyle", NS) is not None else "Normal"
            num = ""
            if ppr is not None and ppr.find("w:numPr", NS) is not None:
                nid = ppr.find("w:numPr/w:numId", NS)
                num = f" numId={nid.get(f'{{{W}}}val')}" if nid is not None else ""
            pb = " [pageBreakBefore]" if ppr is not None and ppr.find("w:pageBreakBefore", NS) is not None else ""
            txt = ptext(child).strip()
            sect = ppr.find("w:sectPr", NS) if ppr is not None else None
            if sect is not None:
                refs = []
                for ref in sect:
                    ln = etree.QName(ref).localname
                    if ln in ("headerReference", "footerReference"):
                        target = relmap.get(ref.get(f"{{{R}}}id"), "?")
                        refs.append(f"{ln[:6]}/{ref.get(f'{{{W}}}type')}={target}:'{hf_text.get(target, '')}'")
                    elif ln in ("type", "pgNumType"):
                        refs.append(f"{ln}{dict((etree.QName(a).localname, v) for a, v in ref.attrib.items())}")
                    elif ln == "pgMar":
                        refs.append("pgMar" + str({etree.QName(a).localname: v for a, v in ref.attrib.items()}))
                print(f"{i:5d} <<SECTION BREAK>> " + " ; ".join(refs))
                if not txt:
                    continue
            if txt or style != "Normal":
                print(f"{i:5d} [{style}{num}]{pb} {txt[:200]}")
        elif tag == "tbl":
            rows = child.findall("w:tr", NS)
            first = " | ".join(ptext(c).strip() for c in rows[0].findall("w:tc", NS)) if rows else ""
            print(f"{i:5d} <TABLE rows={len(rows)} cols={len(rows[0].findall('w:tc', NS)) if rows else 0}> {first[:150]}")
        elif tag == "sdt":
            print(f"{i:5d} <SDT> {ptext(child)[:120]}")
        elif tag == "sectPr":
            refs = []
            for ref in child:
                ln = etree.QName(ref).localname
                if ln in ("headerReference", "footerReference"):
                    target = relmap.get(ref.get(f"{{{R}}}id"), "?")
                    refs.append(f"{ln[:6]}/{ref.get(f'{{{W}}}type')}={target}:'{hf_text.get(target, '')}'")
            print(f"{i:5d} <<FINAL SECTPR>> " + " ; ".join(refs))


if __name__ == "__main__":
    main(sys.argv[1])
