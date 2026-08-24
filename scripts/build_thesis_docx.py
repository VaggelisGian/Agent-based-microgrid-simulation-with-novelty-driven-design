"""Build the thesis .docx on top of the department template.

The template (a .dotx) is opened as a zip, its document.xml is rewritten around the
front-matter skeleton it already carries (cover, second sheet, dedication, Πρόλογος,
Περίληψη, Abstract, Ευχαριστίες, Περιεχόμενα, lists, Συντομογραφίες), and the body
chapters, bibliography and appendices are generated from a small markdown dialect
(docs/thesis/submission/src/FORMAT.md). Styles, numbering, page setup, headers and
footers come from the template itself; nothing is recreated from scratch.

Usage:
    python scripts/build_thesis_docx.py \
        --template "docs/Πρότυπο-Συγγραφής-Διπλωματικών-Ιούνιος-2020-v1.1.dotx" \
        --src docs/thesis/submission/src \
        --out docs/thesis/submission/thesis.docx

The TOC, the lists of figures and tables and the page numbers are Word fields; run
scripts/finalize_thesis_docx.ps1 afterwards to update them and record page counts.
"""

from __future__ import annotations

import argparse
import copy
import io
import re
import struct
import sys
import zipfile
from pathlib import Path

import yaml
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W, "r": R}

TEXT_WIDTH_TWIPS = 9060  # template text width (A4 minus 2 x 2.5 cm), as its tables use
EMU_PER_TWIP = 635
MAX_FIG_HEIGHT_EMU = int(20.0 / 2.54 * 914400)  # 20 cm

GREEK_LETTERS = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
TEMPLATE_PROPERTY_NAME = "Πρότυπο-Συγγραφής-Διπλωματικών Ιούνιος 2020 v1.1"


def xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def set_property(xml: str, tag: str, value: str, drop: bool = False) -> str:
    """Set a docProps element, escaping the value and handling the self-closing form Word
    writes for empty properties. Returns the xml unchanged only if the tag is absent."""
    body = "" if drop else f"<{tag}>{xml_escape(value)}</{tag}>"
    pair = re.compile(rf"<{re.escape(tag)}(\s[^>]*)?>.*?</{re.escape(tag)}>", re.S)
    if pair.search(xml):
        return pair.sub(body, xml, count=1)
    empty = re.compile(rf"<{re.escape(tag)}(\s[^>]*)?/>")
    if empty.search(xml):
        return empty.sub(body, xml, count=1)
    return xml


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    return "{%s}%s" % ({"w": W, "r": R, "wp": WP, "a": A, "pic": PIC, "m": M, "asvg": ASVG}[prefix], local)


def el(tag: str, attrib: dict | None = None, text: str | None = None) -> etree._Element:
    e = etree.Element(qn(tag))
    for k, v in (attrib or {}).items():
        e.set(qn(k) if ":" in k else k, str(v))
    if text is not None:
        e.text = text
    return e


def sub(parent: etree._Element, tag: str, attrib: dict | None = None, text: str | None = None) -> etree._Element:
    e = el(tag, attrib, text)
    parent.append(e)
    return e


# ----------------------------------------------------------------------------
# Inline markup
# ----------------------------------------------------------------------------

INLINE_RE = re.compile(
    r"(\*\*(?P<bold>.+?)\*\*)"
    r"|(\*(?P<ital>[^*]+?)\*)"
    r"|(`(?P<code>[^`]+?)`)"
    r"|(\[(?P<cite>@[A-Za-z0-9_:\-]+(?:\s*;\s*@[A-Za-z0-9_:\-]+)*)\])"
    r"|(@(?P<ref>(fig|tbl|eq):[A-Za-z0-9_\-]+))"
)


class Context:
    """Cross-document state: citation order, figure/table/equation numbers."""

    def __init__(self, bib: dict[str, str]):
        self.bib = bib
        self.cite_order: list[str] = []
        self.labels: dict[str, str] = {}  # fig:key -> "4.1"
        self.bookmark_id = 1000
        self.docpr_id = 100
        self.missing_refs: list[str] = []

    def cite(self, key: str) -> int:
        if key not in self.bib:
            raise SystemExit(f"citation key not in bibliography.md: {key}")
        if key not in self.cite_order:
            self.cite_order.append(key)
        return self.cite_order.index(key) + 1

    def next_bookmark(self) -> int:
        self.bookmark_id += 1
        return self.bookmark_id

    def next_docpr(self) -> int:
        self.docpr_id += 1
        return self.docpr_id


def make_run(text: str, bold=False, italic=False, code=False, lang: str | None = None,
             size: int | None = None, superscript=False, noproof=False) -> etree._Element:
    r = el("w:r")
    rpr = el("w:rPr")
    if code:
        sub(rpr, "w:rFonts", {"w:ascii": "Consolas", "w:hAnsi": "Consolas", "w:cs": "Consolas"})
    if bold:
        sub(rpr, "w:b")
        sub(rpr, "w:bCs")
    if italic:
        sub(rpr, "w:i")
        sub(rpr, "w:iCs")
    if noproof:
        sub(rpr, "w:noProof")
    if size:
        sub(rpr, "w:sz", {"w:val": size})
        sub(rpr, "w:szCs", {"w:val": size})
    if code and not size:
        sub(rpr, "w:sz", {"w:val": 20})
        sub(rpr, "w:szCs", {"w:val": 20})
    if superscript:
        sub(rpr, "w:vertAlign", {"w:val": "superscript"})
    if lang:
        sub(rpr, "w:lang", {"w:val": lang})
    if len(rpr):
        r.append(rpr)
    t = sub(r, "w:t", text=text)
    if text != text.strip() or "  " in text:
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


def inline_runs(text: str, ctx: Context, lang: str | None = None, base_size: int | None = None) -> list[etree._Element]:
    runs: list[etree._Element] = []
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            runs.append(make_run(text[pos:m.start()], lang=lang, size=base_size))
        if m.group("bold") is not None:
            runs.extend(inline_runs_simple(m.group("bold"), ctx, bold=True, lang=lang, size=base_size))
        elif m.group("ital") is not None:
            runs.extend(inline_runs_simple(m.group("ital"), ctx, italic=True, lang=lang, size=base_size))
        elif m.group("code") is not None:
            runs.append(make_run(m.group("code"), code=True, lang="en-US", size=base_size))
        elif m.group("cite") is not None:
            keys = [k.strip().lstrip("@") for k in m.group("cite").split(";")]
            nums = [ctx.cite(k) for k in keys]
            runs.append(make_run(", ".join(f"[{n}]" for n in nums), lang=lang, size=base_size))
        elif m.group("ref") is not None:
            label = m.group("ref")
            kind = label.split(":")[0]
            num = ctx.labels.get(label)
            if num is None:
                ctx.missing_refs.append(label)
                num = "?"
            if kind == "fig":
                txt = f"Σχήμα {num}"
            elif kind == "tbl":
                txt = f"Πίνακας {num}"
            else:
                txt = f"({num})"
            runs.append(make_run(txt, lang=lang, size=base_size))
        pos = m.end()
    if pos < len(text):
        runs.append(make_run(text[pos:], lang=lang, size=base_size))
    return runs


def inline_runs_simple(text: str, ctx: Context, bold=False, italic=False, lang=None, size=None):
    # nested code/citations inside bold or italic spans
    out = []
    pos = 0
    for m in re.finditer(r"`([^`]+?)`|\[(@[A-Za-z0-9_:\-; ]+)\]", text):
        if m.start() > pos:
            out.append(make_run(text[pos:m.start()], bold=bold, italic=italic, lang=lang, size=size))
        if m.group(1) is not None:
            out.append(make_run(m.group(1), code=True, bold=bold, lang="en-US", size=size))
        else:
            keys = [k.strip().lstrip("@") for k in m.group(2).split(";")]
            out.append(make_run(", ".join(f"[{ctx.cite(k)}]" for k in keys), bold=bold, italic=italic, lang=lang, size=size))
        pos = m.end()
    if pos < len(text):
        out.append(make_run(text[pos:], bold=bold, italic=italic, lang=lang, size=size))
    return out


# ----------------------------------------------------------------------------
# Block builders
# ----------------------------------------------------------------------------

def para(style: str | None, runs: list[etree._Element], *, jc: str | None = None, keep_next=False,
         page_break_before=False, num_id: int | None = None, ilvl: int | None = None,
         spacing: dict | None = None, tabs: list[tuple[str, int]] | None = None,
         ind: dict | None = None, shading: str | None = None, keep_lines=False) -> etree._Element:
    p = el("w:p")
    ppr = el("w:pPr")
    if style:
        sub(ppr, "w:pStyle", {"w:val": style})
    if keep_next:
        sub(ppr, "w:keepNext")
    if keep_lines:
        sub(ppr, "w:keepLines")
    if page_break_before:
        sub(ppr, "w:pageBreakBefore")
    if num_id is not None:
        npr = sub(ppr, "w:numPr")
        sub(npr, "w:ilvl", {"w:val": ilvl if ilvl is not None else 0})
        sub(npr, "w:numId", {"w:val": num_id})
    if shading:
        sub(ppr, "w:shd", {"w:val": "clear", "w:color": "auto", "w:fill": shading})
    if tabs:
        t = sub(ppr, "w:tabs")
        for kind, pos in tabs:
            sub(t, "w:tab", {"w:val": kind, "w:pos": pos})
    if spacing:
        sub(ppr, "w:spacing", {f"w:{k}": v for k, v in spacing.items()})
    if ind:
        sub(ppr, "w:ind", {f"w:{k}": v for k, v in ind.items()})
    if jc:
        sub(ppr, "w:jc", {"w:val": jc})
    p.append(ppr)
    for r in runs:
        p.append(r)
    return p


def field_runs(instr: str, cached: str) -> list[etree._Element]:
    out = []
    r = el("w:r"); sub(r, "w:fldChar", {"w:fldCharType": "begin"}); out.append(r)
    r = el("w:r"); it = sub(r, "w:instrText", text=f" {instr} ")
    it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); out.append(r)
    r = el("w:r"); sub(r, "w:fldChar", {"w:fldCharType": "separate"}); out.append(r)
    out.append(make_run(cached, noproof=True))
    r = el("w:r"); sub(r, "w:fldChar", {"w:fldCharType": "end"}); out.append(r)
    return out


def caption_para(kind: str, chapter_label: str, seq_num: int, text: str, ctx: Context,
                 bookmark: str, above: bool, is_appendix: bool) -> etree._Element:
    """kind: 'Σχήμα' or 'Πίνακας'. Mirrors the template's caption field structure."""
    p = el("w:p")
    ppr = sub(p, "w:pPr")
    sub(ppr, "w:pStyle", {"w:val": "Caption"})
    if above:
        sub(ppr, "w:keepNext")
    bid = ctx.next_bookmark()
    sub(p, "w:bookmarkStart", {"w:id": bid, "w:name": bookmark})
    p.append(make_run(f"{kind} "))
    if is_appendix:
        p.append(make_run(chapter_label))
    else:
        for r in field_runs("STYLEREF 1 \\s", chapter_label):
            p.append(r)
    p.append(make_run("."))
    for r in field_runs(f"SEQ {kind} \\* ARABIC \\s 1", str(seq_num)):
        p.append(r)
    sub(p, "w:bookmarkEnd", {"w:id": bid})
    p.append(make_run(": "))
    for r in inline_runs(text, ctx):
        p.append(r)
    return p


def strip_image_metadata(data: bytes, ext: str) -> bytes:
    """Drop generator metadata from embedded images: PNG text/time chunks (matplotlib writes a
    tEXt Software chunk) and the SVG <metadata> element (matplotlib writes dc:creator and a
    build timestamp there). Pixel and vector payloads are untouched."""
    if ext == "png" and data[:8] == b"\x89PNG\r\n\x1a\n":
        out = bytearray(data[:8])
        pos = 8
        drop = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf"}
        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            end = pos + 12 + length
            if ctype not in drop:
                out += data[pos:end]
            pos = end
        return bytes(out)
    if ext == "svg":
        text = data.decode("utf-8")
        text = re.sub(r"<metadata[^>]*>.*?</metadata>", "", text, flags=re.DOTALL)
        return text.encode("utf-8")
    return data


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("only PNG figures are supported")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def figure_para(rid_png: str, rid_svg: str | None, px_w: int, px_h: int, width_frac: float, ctx: Context,
                name: str) -> etree._Element:
    cx = int(TEXT_WIDTH_TWIPS * EMU_PER_TWIP * width_frac)
    cy = int(cx * px_h / px_w)
    if cy > MAX_FIG_HEIGHT_EMU:
        cy = MAX_FIG_HEIGHT_EMU
        cx = int(cy * px_w / px_h)
    did = ctx.next_docpr()
    p = el("w:p")
    ppr = sub(p, "w:pPr")
    sub(ppr, "w:keepNext")
    sub(ppr, "w:spacing", {"w:before": 120, "w:after": 0})
    sub(ppr, "w:jc", {"w:val": "center"})
    r = sub(p, "w:r")
    rpr = sub(r, "w:rPr"); sub(rpr, "w:noProof")
    drawing = sub(r, "w:drawing")
    inline = sub(drawing, "wp:inline", {"distT": 0, "distB": 0, "distL": 0, "distR": 0})
    sub(inline, "wp:extent", {"cx": cx, "cy": cy})
    sub(inline, "wp:effectExtent", {"l": 0, "t": 0, "r": 0, "b": 0})
    sub(inline, "wp:docPr", {"id": did, "name": name})
    cnv = sub(inline, "wp:cNvGraphicFramePr")
    sub(cnv, "a:graphicFrameLocks", {"noChangeAspect": 1})
    graphic = sub(inline, "a:graphic")
    gdata = sub(graphic, "a:graphicData", {"uri": PIC})
    pic = sub(gdata, "pic:pic")
    nv = sub(pic, "pic:nvPicPr")
    sub(nv, "pic:cNvPr", {"id": did, "name": name})
    cnvp = sub(nv, "pic:cNvPicPr")
    sub(cnvp, "a:picLocks", {"noChangeAspect": 1})
    bf = sub(pic, "pic:blipFill")
    blip = sub(bf, "a:blip", {"r:embed": rid_png})
    if rid_svg:
        ext_lst = sub(blip, "a:extLst")
        ext = sub(ext_lst, "a:ext", {"uri": "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"})
        sub(ext, "asvg:svgBlip", {"r:embed": rid_svg})
    sub(sub(bf, "a:stretch"), "a:fillRect")
    sppr = sub(pic, "pic:spPr")
    xfrm = sub(sppr, "a:xfrm")
    sub(xfrm, "a:off", {"x": 0, "y": 0})
    sub(xfrm, "a:ext", {"cx": cx, "cy": cy})
    sub(sub(sppr, "a:prstGeom", {"prst": "rect"}), "a:avLst")
    return p


def table_block(header: list[str], rows: list[list[str]], ctx: Context) -> etree._Element:
    ncol = len(header)
    size = 20 if ncol <= 5 else 18
    # column widths proportional to the longest cell, with a floor
    lens = [max(len(h), *(len(r[i]) if i < len(r) else 0 for r in rows)) for i, h in enumerate(header)]
    lens = [max(6, min(l, 40)) for l in lens]
    total = sum(lens)
    widths = [int(TEXT_WIDTH_TWIPS * l / total) for l in lens]
    widths[-1] += TEXT_WIDTH_TWIPS - sum(widths)
    tbl = el("w:tbl")
    tpr = sub(tbl, "w:tblPr")
    sub(tpr, "w:tblStyle", {"w:val": "TableGrid"})
    sub(tpr, "w:tblW", {"w:w": TEXT_WIDTH_TWIPS, "w:type": "dxa"})
    sub(tpr, "w:jc", {"w:val": "center"})
    sub(tpr, "w:tblLayout", {"w:type": "fixed"})
    sub(tpr, "w:tblLook", {"w:val": "04A0", "w:firstRow": 1, "w:lastRow": 0, "w:firstColumn": 1,
                           "w:lastColumn": 0, "w:noHBand": 0, "w:noVBand": 1})
    grid = sub(tbl, "w:tblGrid")
    for wdt in widths:
        sub(grid, "w:gridCol", {"w:w": wdt})

    def row(cells: list[str], is_header: bool) -> etree._Element:
        tr = el("w:tr")
        trpr = sub(tr, "w:trPr")
        sub(trpr, "w:cantSplit")
        if is_header:
            sub(trpr, "w:tblHeader")
        for i in range(ncol):
            txt = cells[i] if i < len(cells) else ""
            tc = sub(tr, "w:tc")
            tcpr = sub(tc, "w:tcPr")
            sub(tcpr, "w:tcW", {"w:w": widths[i], "w:type": "dxa"})
            sub(tcpr, "w:vAlign", {"w:val": "center"})
            numeric = bool(re.fullmatch(r"[-+]?[0-9][0-9.,e+\-% ]*", txt.strip())) and not is_header
            runs = inline_runs(txt, ctx, base_size=size)
            if is_header:
                for r_ in runs:
                    rpr = r_.find(qn("w:rPr"))
                    if rpr is None:
                        rpr = el("w:rPr"); r_.insert(0, rpr)
                    at = 1 if (len(rpr) and etree.QName(rpr[0]).localname == "rFonts") else 0
                    rpr.insert(at, el("w:b"))
            tc.append(para(None, runs, jc="center" if (is_header or numeric) else "left",
                           spacing={"after": 0, "line": 240, "lineRule": "auto"}))
        return tr

    tbl.append(row(header, True))
    for r_ in rows:
        tbl.append(row(r_, False))
    return tbl


def equation_para(latex: str, chapter_label: str, seq_num: int, ctx: Context, is_appendix: bool,
                  xslt: etree.XSLT) -> etree._Element:
    import latex2mathml.converter as l2m
    mml = l2m.convert(latex)
    omath = xslt(etree.fromstring(mml)).getroot()
    p = el("w:p")
    ppr = sub(p, "w:pPr")
    sub(ppr, "w:pStyle", {"w:val": "functionstyle"})
    r = sub(p, "w:r"); sub(r, "w:tab")
    p.append(omath)
    r = sub(p, "w:r"); sub(r, "w:tab")
    p.append(make_run("("))
    if is_appendix:
        p.append(make_run(chapter_label))
    else:
        for fr in field_runs("STYLEREF 1 \\s", chapter_label):
            p.append(fr)
    p.append(make_run("."))
    for fr in field_runs("SEQ ( \\* ARABIC \\s 1", str(seq_num)):
        p.append(fr)
    p.append(make_run(")"))
    return p


def code_paras(lines: list[str]) -> list[etree._Element]:
    out = []
    for i, line in enumerate(lines):
        runs = [make_run(line.replace("\t", "    ") if line else " ", code=True, lang="en-US", size=17)]
        out.append(para("CodeBlock", runs, keep_next=(i < len(lines) - 1), keep_lines=True))
    return out


# ----------------------------------------------------------------------------
# Markdown dialect parser
# ----------------------------------------------------------------------------

class Block:
    def __init__(self, kind: str, **kw):
        self.kind = kind
        self.__dict__.update(kw)


def parse_markdown(text: str) -> tuple[str | None, list[Block]]:
    lines = text.replace("\r\n", "\n").split("\n")
    title = None
    blocks: list[Block] = []
    i = 0
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            blocks.append(Block("p", text=" ".join(s.strip() for s in buf)))
            buf = []

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("# ") and title is None and not blocks:
            title = s[2:].strip(); i += 1; continue
        if s.startswith("### "):
            flush(); blocks.append(Block("h3", text=s[4:].strip())); i += 1; continue
        if s.startswith("## "):
            flush(); blocks.append(Block("h2", text=s[3:].strip())); i += 1; continue
        if s.startswith("# "):
            raise SystemExit(f"second '# ' heading found: {s}")
        if s == "\\newpage":
            flush(); blocks.append(Block("pagebreak")); i += 1; continue
        if s.startswith("```"):
            flush()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip()); i += 1
            i += 1
            blocks.append(Block("code", lines=code)); continue
        if s.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:]
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("- ") \
                        and lines[i].startswith("  "):
                    item += " " + lines[i].strip(); i += 1
                items.append(item)
            blocks.append(Block("list", items=items)); continue
        m = re.match(r"!\[(?P<cap>.*)\]\((?P<path>[^)]+)\)\s*(\{(?P<attrs>[^}]*)\})?\s*$", s)
        if m:
            flush()
            attrs = m.group("attrs") or ""
            key = re.search(r"#(fig:[A-Za-z0-9_\-]+)", attrs)
            wm = re.search(r"width=([0-9.]+)", attrs)
            blocks.append(Block("fig", caption=m.group("cap"), path=m.group("path"),
                                key=key.group(1) if key else None, width=float(wm.group(1)) if wm else 0.85))
            i += 1; continue
        m = re.match(r"Table:\s*(?P<cap>.*?)\s*(\{#(?P<key>tbl:[A-Za-z0-9_\-]+)\})?\s*$", s)
        if m:
            flush()
            i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells); i += 1
            if len(rows) < 2:
                raise SystemExit(f"table without rows after caption: {s}")
            header = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            if not body:
                raise SystemExit(f"table with a header but no body rows: {s}")
            blocks.append(Block("table", caption=m.group("cap"), key=m.group("key"), header=header, rows=body))
            continue
        m = re.match(r"\$\$(?P<tex>.+)\$\$\s*(\{#(?P<key>eq:[A-Za-z0-9_\-]+)\})?\s*$", s)
        if m:
            flush(); blocks.append(Block("eq", tex=m.group("tex").strip(), key=m.group("key"))); i += 1; continue
        buf.append(line); i += 1
    flush()
    return title, blocks


# ----------------------------------------------------------------------------
# Document assembly
# ----------------------------------------------------------------------------

class Package:
    def __init__(self, template: Path):
        self.parts: dict[str, bytes] = {}
        with zipfile.ZipFile(template) as z:
            for n in z.namelist():
                self.parts[n] = z.read(n)
        self.rels = etree.fromstring(self.parts["word/_rels/document.xml.rels"])
        self.ctypes = etree.fromstring(self.parts["[Content_Types].xml"])
        self.next_rid = 1 + max(int(r.get("Id")[3:]) for r in self.rels)
        self.next_part = {"header": 100, "footer": 100, "image": 100}

    def add_rel(self, rtype: str, target: str) -> str:
        rid = f"rId{self.next_rid}"; self.next_rid += 1
        e = etree.SubElement(self.rels, "{%s}Relationship" % PKG_REL)
        e.set("Id", rid); e.set("Type", f"http://schemas.openxmlformats.org/officeDocument/2006/relationships/{rtype}")
        e.set("Target", target)
        return rid

    def add_override(self, part: str, ctype: str):
        e = etree.SubElement(self.ctypes, "{%s}Override" % CT)
        e.set("PartName", part); e.set("ContentType", ctype)

    def ensure_default(self, ext: str, ctype: str):
        for d in self.ctypes.findall("{%s}Default" % CT):
            if d.get("Extension") == ext:
                return
        e = etree.SubElement(self.ctypes, "{%s}Default" % CT)
        e.set("Extension", ext); e.set("ContentType", ctype)

    def add_image(self, data: bytes, ext: str) -> str:
        self.next_part["image"] += 1
        name = f"media/thesis_image{self.next_part['image']}.{ext}"
        self.parts["word/" + name] = strip_image_metadata(data, ext)
        if ext == "svg":
            self.ensure_default("svg", "image/svg+xml")
        return self.add_rel("image", name)

    def add_header_footer(self, kind: str, xml: bytes) -> str:
        self.next_part[kind] += 1
        name = f"{kind}{self.next_part[kind]}.xml"
        self.parts["word/" + name] = xml
        self.add_override(f"/word/{name}", f"application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml")
        return self.add_rel(kind, name)

    def write(self, out: Path, document_xml: bytes):
        self.parts["word/document.xml"] = document_xml
        self.parts["word/_rels/document.xml.rels"] = etree.tostring(self.rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        # a .docx, not a .dotx
        for o in self.ctypes.findall("{%s}Override" % CT):
            if o.get("PartName") == "/word/document.xml":
                o.set("ContentType", "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml")
        self.parts["[Content_Types].xml"] = etree.tostring(self.ctypes, xml_declaration=True, encoding="UTF-8", standalone=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", self.parts["[Content_Types].xml"])
            for n, data in self.parts.items():
                if n == "[Content_Types].xml":
                    continue
                z.writestr(n, data)


def header_xml(text: str, right: bool) -> bytes:
    hdr = etree.Element(qn("w:hdr"), nsmap={"w": W, "r": R})
    p = sub(hdr, "w:p")
    ppr = sub(p, "w:pPr"); sub(ppr, "w:pStyle", {"w:val": "Header"})
    if right:
        sub(ppr, "w:jc", {"w:val": "right"})
    p.append(make_run(text))
    return etree.tostring(hdr, xml_declaration=True, encoding="UTF-8", standalone=True)


def par_text(p: etree._Element) -> str:
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def set_par_text(p: etree._Element, text: str):
    ts = list(p.iter(qn("w:t")))
    if not ts:
        r = sub(p, "w:r"); ts = [sub(r, "w:t")]
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in ts[1:]:
        t.text = ""


def set_sect_type(sectpr: etree._Element, kind: str):
    t = sectpr.find(qn("w:type"))
    if t is None:
        t = el("w:type"); sectpr.insert(0, t)
    t.set(qn("w:val"), kind)


def sectpr_for(pkg: Package, base: etree._Element, even_text: str, odd_text: str, start_page: int | None,
               page_fmt: str | None = None) -> etree._Element:
    sp = copy.deepcopy(base)
    for ref in list(sp):
        if etree.QName(ref).localname in ("headerReference", "footerReference", "pgNumType", "type"):
            sp.remove(ref)
    refs = [
        el("w:headerReference", {"w:type": "even", "r:id": pkg.add_header_footer("header", header_xml(even_text, False))}),
        el("w:headerReference", {"w:type": "default", "r:id": pkg.add_header_footer("header", header_xml(odd_text, True))}),
        el("w:footerReference", {"w:type": "even", "r:id": "rId31"}),   # template page-number footers
        el("w:footerReference", {"w:type": "default", "r:id": "rId17"}),
        el("w:type", {"w:val": "nextPage"}),
    ]
    for i, r_ in enumerate(refs):
        sp.insert(i, r_)
    if start_page is not None or page_fmt:
        pn = el("w:pgNumType")
        if page_fmt:
            pn.set(qn("w:fmt"), page_fmt)
        if start_page is not None:
            pn.set(qn("w:start"), str(start_page))
        idx = [i for i, c in enumerate(sp) if etree.QName(c).localname == "pgMar"][0] + 1
        sp.insert(idx, pn)
    return sp


class Builder:
    def __init__(self, template: Path, src: Path, repo: Path, out: Path):
        self.pkg = Package(template)
        self.src = src
        self.repo = repo
        self.out = out
        self.meta = yaml.safe_load((src / "meta.yaml").read_text(encoding="utf-8"))
        self.bib = self.load_bib(src / "bibliography.md")
        self.ctx = Context(self.bib)
        self.doc = etree.fromstring(self.pkg.parts["word/document.xml"])
        self.body = self.doc.find(qn("w:body"))
        self.kids = list(self.body)
        self.xslt = etree.XSLT(etree.parse(str(self.meta["mml2omml_xsl"])))
        self.figure_log: list[dict] = []
        self.stats = {"chapters": [], "figures": 0, "tables": 0, "equations": 0, "citations": 0}

    @staticmethod
    def load_bib(path: Path) -> dict[str, str]:
        bib = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[@([A-Za-z0-9_:\-]+)\]\s+(.*)", line.strip())
            if m:
                bib[m.group(1)] = m.group(2).strip()
        return bib

    # --- template anchors --------------------------------------------------
    def find_par(self, startswith: str, after: int = 0) -> int:
        for i in range(after, len(self.kids)):
            k = self.kids[i]
            if etree.QName(k).localname == "p" and par_text(k).strip().startswith(startswith):
                return i
        raise SystemExit(f"template anchor not found: {startswith!r}")

    def find_sect_par(self, after: int) -> int:
        for i in range(after, len(self.kids)):
            k = self.kids[i]
            if etree.QName(k).localname == "p" and k.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None:
                return i
        raise SystemExit("section paragraph not found")

    # --- content ------------------------------------------------------------
    def render_blocks(self, blocks: list[Block], chapter_label: str, is_appendix: bool, lang: str | None = None,
                      numbered_headings=True) -> list[etree._Element]:
        out: list[etree._Element] = []
        counters = {"fig": 0, "tbl": 0, "eq": 0, "h2": 0, "h3": 0}
        for b in blocks:
            if b.kind == "p":
                out.append(para(None, inline_runs(b.text, self.ctx, lang=lang)))
            elif b.kind == "h2":
                if is_appendix:
                    # appendix sections carry a literal letter label (Α.1, Α.2 ...): the Heading2
                    # list numbering would continue the last chapter's counter (9.7, 9.8 ...)
                    counters["h2"] += 1
                    counters["h3"] = 0
                    label = make_run(f"{chapter_label}.{counters['h2']} ")
                    out.append(para("Heading2", [label] + inline_runs(b.text, self.ctx, lang=lang), num_id=0))
                else:
                    out.append(para("Heading2", inline_runs(b.text, self.ctx, lang=lang),
                                    num_id=None if numbered_headings else 0))
            elif b.kind == "h3":
                if is_appendix:
                    counters["h3"] += 1
                    label = make_run(f"{chapter_label}.{counters['h2']}.{counters['h3']} ")
                    out.append(para("Heading3", [label] + inline_runs(b.text, self.ctx, lang=lang), num_id=0))
                else:
                    out.append(para("Heading3", inline_runs(b.text, self.ctx, lang=lang)))
            elif b.kind == "list":
                for it in b.items:
                    out.append(para("ListParagraph", inline_runs(it, self.ctx, lang=lang)))
            elif b.kind == "pagebreak":
                p = el("w:p"); r = sub(p, "w:r"); sub(r, "w:br", {"w:type": "page"}); out.append(p)
            elif b.kind == "code":
                out.extend(code_paras(b.lines))
                out.append(para(None, [], spacing={"after": 0, "line": 120, "lineRule": "auto"}))
            elif b.kind == "fig":
                counters["fig"] += 1
                num = f"{chapter_label}.{counters['fig']}"
                if b.key:
                    self.ctx.labels[b.key] = num
                path = self.repo / b.path
                data = path.read_bytes()
                w, h = png_size(data)
                rid = self.pkg.add_image(data, "png")
                rid_svg = None
                svg = path.with_suffix(".svg")
                if svg.exists():
                    rid_svg = self.pkg.add_image(svg.read_bytes(), "svg")
                out.append(figure_para(rid, rid_svg, w, h, b.width, self.ctx, path.name))
                out.append(caption_para("Σχήμα", chapter_label, counters["fig"], b.caption, self.ctx,
                                        f"_Ref_{(b.key or 'fig').replace(':', '_')}", False, is_appendix))
                self.figure_log.append({"number": f"Σχήμα {num}", "file": b.path, "caption": b.caption, "svg": bool(rid_svg)})
                self.stats["figures"] += 1
            elif b.kind == "table":
                counters["tbl"] += 1
                num = f"{chapter_label}.{counters['tbl']}"
                if b.key:
                    self.ctx.labels[b.key] = num
                out.append(caption_para("Πίνακας", chapter_label, counters["tbl"], b.caption, self.ctx,
                                        f"_Ref_{(b.key or 'tbl').replace(':', '_')}", True, is_appendix))
                out.append(table_block(b.header, b.rows, self.ctx))
                out.append(para(None, [], spacing={"after": 120, "line": 120, "lineRule": "auto"}))
                self.stats["tables"] += 1
            elif b.kind == "eq":
                counters["eq"] += 1
                num = f"{chapter_label}.{counters['eq']}"
                if b.key:
                    self.ctx.labels[b.key] = num
                out.append(equation_para(b.tex, chapter_label, counters["eq"], self.ctx, is_appendix, self.xslt))
                self.stats["equations"] += 1
        return out

    def prelabel(self, blocks: list[Block], chapter_label: str):
        """Assign figure/table/equation numbers before rendering so forward references resolve."""
        counters = {"fig": 0, "tbl": 0, "eq": 0}
        for b in blocks:
            if b.kind in ("fig", "table", "eq"):
                k = {"fig": "fig", "table": "tbl", "eq": "eq"}[b.kind]
                counters[k] += 1
                if b.key:
                    self.ctx.labels[b.key] = f"{chapter_label}.{counters[k]}"

    def read_md(self, name: str) -> tuple[str | None, list[Block]]:
        return parse_markdown((self.src / name).read_text(encoding="utf-8"))

    # --- build ----------------------------------------------------------------
    def build(self):
        k = self.kids
        meta = self.meta
        # ---- cover and second sheet ----
        set_par_text(k[self.find_par("«ΘΕΜΑ»")], meta["title_el"])
        cover_tbl = [c for c in k if etree.QName(c).localname == "tbl"][1]
        for p in cover_tbl.iter(qn("w:p")):
            t = par_text(p)
            if t.startswith("Τ.… φοιτητ"):
                set_par_text(p, meta["student_label_el"])
            elif t.startswith("…………"):
                set_par_text(p, meta["author_el"])
            elif t.startswith("Αρ. Μητρώου"):
                set_par_text(p, f"Αρ. Μητρώου: {meta['student_id']}")
            elif t.startswith("Ονοματεπώνυμο"):
                set_par_text(p, meta["supervisor_el"])
            elif t.startswith("Βαθμίδα"):
                set_par_text(p, meta["supervisor_rank_el"])
        set_par_text(k[self.find_par("Τίτλος Δ.Ε.")], f"Τίτλος Δ.Ε.: {meta['title_el']}")
        set_par_text(k[self.find_par("Κωδικός Δ.Ε.")], f"Κωδικός Δ.Ε.: {meta['thesis_code']}")
        set_par_text(k[self.find_par("Ονοματεπώνυμο φοιτητή")], f"Ονοματεπώνυμο φοιτητή: {meta['author_el']}")
        set_par_text(k[self.find_par("Ονοματεπώνυμο εισηγητή")], f"Ονοματεπώνυμο εισηγητή: {meta['supervisor_el']}")
        set_par_text(k[self.find_par("Ημερομηνία ανάληψης")], f"Ημερομηνία ανάληψης Δ.Ε.: {meta['date_assigned']}")
        set_par_text(k[self.find_par("Ημερομηνία περάτωσης")], f"Ημερομηνία περάτωσης Δ.Ε.: {meta['date_completed']}")
        ip = k[self.find_par("Η παρούσα εργασία αποτελεί πνευματική ιδιοκτησία")]
        set_par_text(ip, par_text(ip).replace("τ__  φοιτητ__  _________________________________", meta["author_genitive_el"])
                     .replace("που την εκπόνησε/αν", "που την εκπόνησε"))
        cover_sect = self.find_sect_par(0)
        dedic_sect = self.find_sect_par(cover_sect + 1)
        front_sect = self.find_sect_par(dedic_sect + 1)
        for i in (cover_sect, dedic_sect, front_sect):
            set_sect_type(k[i].find(qn("w:pPr") + "/" + qn("w:sectPr")), "nextPage")

        new_body: list[etree._Element] = []
        cover = k[: dedic_sect + 1]  # cover, second sheet, dedication
        # the long title needs the vertical slack the template leaves above the date line
        i_date = self.find_par("Ημερομηνία ……")
        drop = [k[i_date - 1], k[i_date - 2], k[i_date - 3]]
        drop = [d for d in drop if etree.QName(d).localname == "p" and not par_text(d).strip()]
        new_body.extend(c for c in cover if c not in drop)

        # ---- front matter ----
        def heading_with_break(idx: int) -> etree._Element:
            p = k[idx]
            ppr = p.find(qn("w:pPr"))
            if ppr.find(qn("w:pageBreakBefore")) is None:
                ppr.insert(1, el("w:pageBreakBefore"))
            return p

        i_pro = self.find_par("Πρόλογος", dedic_sect)
        i_per = self.find_par("Περίληψη", i_pro + 1)
        i_ent = self.find_par("«Τίτλος Διπλωματικής Εργασίας»", i_per + 1)
        i_abs = self.find_par("Abstract", i_ent + 1)
        i_ack = self.find_par("Ευχαριστίες", i_abs + 1)
        i_toc = self.find_par("Περιεχόμενα", i_ack + 1)
        i_lof = self.find_par("Κατάλογος Σχημάτων", i_toc + 1)
        i_lot = self.find_par("Κατάλογος Πινάκων", i_lof + 1)
        i_abb = self.find_par("Συντομογραφίες", i_lot + 1)

        new_body.append(k[i_pro])
        new_body.extend(self.render_blocks(self.read_md("00_prologos.md")[1], "0", False, numbered_headings=False))
        new_body.append(heading_with_break(i_per))
        new_body.extend(self.render_blocks(self.read_md("00_perilipsi.md")[1], "0", False, numbered_headings=False))
        # English title and author block before the Abstract
        ent = heading_with_break(i_ent)
        set_par_text(ent, meta["title_en"])
        new_body.append(ent)
        aut = k[self.find_par("«Όνομα & Επώνυμο Φοιτητή»", i_ent)]
        set_par_text(aut, meta["author_en"])
        new_body.append(aut)
        new_body.append(k[i_abs])
        new_body.extend(self.render_blocks(self.read_md("00_abstract.md")[1], "0", False, lang="en-US", numbered_headings=False))
        new_body.append(heading_with_break(i_ack))
        new_body.extend(self.render_blocks(self.read_md("00_efxaristies.md")[1], "0", False, numbered_headings=False))
        # contents, lists: keep the template's field paragraphs, drop its empty spacer paragraphs
        # (with real list contents a trailing empty paragraph can spill onto its own page before
        # the next pageBreakBefore heading, leaving a blank numbered page)
        def field_pars_only(seq):
            kept = []
            for c in seq:
                if etree.QName(c).localname != "p":
                    kept.append(c)
                    continue
                if (par_text(c).strip()
                        or c.find(".//" + qn("w:fldChar")) is not None
                        or c.find(".//" + qn("w:instrText")) is not None
                        or c.find(".//" + qn("w:drawing")) is not None
                        or c.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None):
                    kept.append(c)
            return kept

        new_body.append(heading_with_break(i_toc))
        new_body.extend(field_pars_only(k[i_toc + 1: i_lof]))
        new_body.append(heading_with_break(i_lof))
        new_body.extend(field_pars_only(k[i_lof + 1: i_lot]))
        new_body.append(k[i_lot])
        new_body.extend(field_pars_only(k[i_lot + 1: i_abb]))
        new_body.append(heading_with_break(i_abb))
        for line in (self.src / "00_syntomografies.md").read_text(encoding="utf-8").splitlines():
            if "|" not in line:
                continue
            abbr, expl = [s.strip() for s in line.split("|", 1)]
            r1 = make_run(abbr); r2 = el("w:r"); sub(r2, "w:tab")
            new_body.append(para(None, [r1, r2] + inline_runs(expl, self.ctx), tabs=[("left", 2268)],
                                 spacing={"after": 40}, jc="left"))
        new_body.append(k[front_sect])

        # ---- body chapters ----
        base_sect = copy.deepcopy(k[self.find_sect_par(front_sect + 1)].find(qn("w:pPr") + "/" + qn("w:sectPr")))
        chapter_files = sorted(p for p in self.src.glob("[0-9][0-9]_*.md") if not p.name.startswith("00_"))
        if not chapter_files:
            raise SystemExit(f"no chapter sources (NN_*.md) under {self.src}")
        for n, f in enumerate(chapter_files, 1):
            title, blocks = self.read_md(f.name)
            label = str(n)
            self.prelabel(blocks, label)
            new_body.append(para("Heading1", inline_runs(title, self.ctx)))
            new_body.extend(self.render_blocks(blocks, label, False))
            sp = sectpr_for(self.pkg, base_sect, f"Κεφάλαιο {n}", title, 1 if n == 1 else None,
                            "decimal" if n == 1 else None)
            new_body.append(para(None, [], spacing={"after": 0}) )
            new_body[-1].find(qn("w:pPr")).append(sp)
            self.stats["chapters"].append({"n": n, "title": title, "file": f.name})
        # ---- appendices ----
        app_files = sorted(p for p in self.src.glob("[A-Z]_*.md"))
        if not app_files:
            raise SystemExit(f"no appendix sources (X_*.md) under {self.src}")
        if len(app_files) > len(GREEK_LETTERS):
            raise SystemExit(f"more appendices than Greek letters: {len(app_files)}")
        app_body: list[etree._Element] = []  # rendered before the bibliography so their citations get numbers
        final_sect = copy.deepcopy(self.body.find(qn("w:sectPr")))
        for n, f in enumerate(app_files):
            title, blocks = self.read_md(f.name)
            letter = GREEK_LETTERS[n]
            self.prelabel(blocks, letter)
            # static Greek letter: Word has no Greek-letter list format, and the regulation asks for Α, Β, Γ
            head = para("Appendix", [make_run(f"ΠΑΡΑΡΤΗΜΑ {letter}: {title}")], num_id=0)
            # hidden SEQ resets: the caption sequences reset on the Heading 1 style only, and the
            # Appendix style does not count, so without these Παράρτημα Γ tables continued Α's counter
            for ident in ("Σχήμα", "Πίνακας", "("):
                for r in field_runs(f"SEQ {ident} \\r 0 \\h", ""):
                    head.append(r)
            app_body.append(head)
            app_body.extend(self.render_blocks(blocks, letter, True))
            if n < len(app_files) - 1:
                # the break paragraph must live in the SAME buffer as the appendix content:
                # putting it in new_body stranded three empty sections before the bibliography
                sp = sectpr_for(self.pkg, base_sect, "Παραρτήματα", f"Παράρτημα {letter}", None)
                app_body.append(para(None, [], spacing={"after": 0}))
                app_body[-1].find(qn("w:pPr")).append(sp)
            self.stats["chapters"].append({"n": f"Παράρτημα {letter}", "title": title, "file": f.name})
        last_sp = sectpr_for(self.pkg, base_sect, "Παραρτήματα", f"Παράρτημα {GREEK_LETTERS[len(app_files) - 1]}", None)
        # the final sectPr keeps the template's page geometry; swap header/footer refs and type
        for c in list(final_sect):
            if etree.QName(c).localname in ("headerReference", "footerReference", "type", "pgNumType"):
                final_sect.remove(c)
        for i, c in enumerate([c for c in last_sp if etree.QName(c).localname in ("headerReference", "footerReference", "type")]):
            final_sect.insert(i, c)

        # ---- bibliography ----
        new_body.append(para("Heading1", [make_run("ΒΙΒΛΙΟΓΡΑΦΙΑ")], num_id=0))
        cite_numbers = list(self.ctx.cite_order)
        uncited = [kk for kk in self.bib if kk not in cite_numbers]
        if uncited:
            print("WARNING uncited bibliography entries:", uncited, file=sys.stderr)
        for i, key in enumerate(cite_numbers, 1):
            runs = [make_run(f"[{i}]", lang="en-US"), (lambda r_: (sub(r_, "w:tab"), r_)[1])(el("w:r"))]
            runs += inline_runs(self.bib[key], self.ctx, lang="en-US")
            new_body.append(para(None, runs, ind={"left": 567, "hanging": 567}, tabs=[("left", 567)],
                                 spacing={"after": 100}))
        self.stats["citations"] = len(cite_numbers)
        sp = sectpr_for(self.pkg, base_sect, "Βιβλιογραφία", "Βιβλιογραφία", None)
        new_body.append(para(None, [], spacing={"after": 0}))
        new_body[-1].find(qn("w:pPr")).append(sp)
        new_body.extend(app_body)
        for c in list(self.body):
            self.body.remove(c)
        for c in new_body:
            self.body.append(c)
        self.body.append(final_sect)

        if self.ctx.missing_refs:
            raise SystemExit(f"unresolved cross-references: {sorted(set(self.ctx.missing_refs))}")

        self.add_code_style()
        self.write_metadata()
        xml = etree.tostring(self.doc, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.pkg.write(self.out, xml)

    def add_code_style(self):
        styles = etree.fromstring(self.pkg.parts["word/styles.xml"])
        if any(s.get(qn("w:styleId")) == "CodeBlock" for s in styles.findall(qn("w:style"))):
            return
        s = el("w:style", {"w:type": "paragraph", "w:customStyle": "1", "w:styleId": "CodeBlock"})
        sub(s, "w:name", {"w:val": "Code Block"})
        sub(s, "w:basedOn", {"w:val": "Normal"})
        sub(s, "w:qFormat")
        ppr = sub(s, "w:pPr")
        sub(ppr, "w:shd", {"w:val": "clear", "w:color": "auto", "w:fill": "F2F2F2"})
        sub(ppr, "w:spacing", {"w:after": 0, "w:line": 240, "w:lineRule": "auto"})
        sub(ppr, "w:ind", {"w:left": 284})
        sub(ppr, "w:jc", {"w:val": "left"})
        rpr = sub(s, "w:rPr")
        sub(rpr, "w:rFonts", {"w:ascii": "Consolas", "w:hAnsi": "Consolas", "w:cs": "Consolas"})
        sub(rpr, "w:sz", {"w:val": 17}); sub(rpr, "w:szCs", {"w:val": 17})
        sub(rpr, "w:lang", {"w:val": "en-US"})
        styles.append(s)
        self.pkg.parts["word/styles.xml"] = etree.tostring(styles, xml_declaration=True, encoding="UTF-8", standalone=True)

    def write_metadata(self):
        core = self.pkg.parts["docProps/core.xml"].decode("utf-8")
        core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{self.meta['title_el']}</dc:title>", core, flags=re.S)
        core = re.sub(r"<dc:creator>.*?</dc:creator>", f"<dc:creator>{self.meta['author_el']}</dc:creator>", core, flags=re.S)
        core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", f"<cp:lastModifiedBy>{self.meta['author_el']}</cp:lastModifiedBy>", core, flags=re.S)
        self.pkg.parts["docProps/core.xml"] = core.encode("utf-8")
        app = self.pkg.parts["docProps/app.xml"].decode("utf-8")
        app = re.sub(r"<Company>.*?</Company>", "<Company></Company>", app, flags=re.S)
        app = re.sub(r"<TitlesOfParts>.*?</TitlesOfParts>", "", app, flags=re.S)
        app = re.sub(r"<HeadingPairs>.*?</HeadingPairs>", "", app, flags=re.S)
        self.pkg.parts["docProps/app.xml"] = app.encode("utf-8")


def postprocess(docx: Path, meta: dict) -> None:
    """After Word has updated the fields and saved: restore the author metadata Word
    overwrote (lastModifiedBy becomes the Windows account name, Template becomes
    Normal.dotm) and confirm no comments or tracked changes exist."""
    tmp = docx.with_suffix(".tmp")
    dropped: set[str] = set()
    with zipfile.ZipFile(docx) as zin:
        for name in zin.namelist():
            if name.startswith("word/comments") or name.startswith("word/people"):
                dropped.add(name)
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if dropped and item.filename.endswith(".rels"):
                # a dropped part must lose its relationship too, or Word offers to repair
                rels = data.decode("utf-8")
                for name in dropped:
                    target = name.split("/")[-1]
                    rels = re.sub(rf'<Relationship[^>]*Target="{re.escape(target)}"[^>]*/>', "", rels)
                data = rels.encode("utf-8")
            if dropped and item.filename == "[Content_Types].xml":
                types = data.decode("utf-8")
                for name in dropped:
                    types = re.sub(rf'<Override[^>]*PartName="/{re.escape(name)}"[^>]*/>', "", types)
                data = types.encode("utf-8")
            if item.filename == "docProps/core.xml":
                core = data.decode("utf-8")
                core = set_property(core, "dc:creator", meta["author_el"])
                core = set_property(core, "cp:lastModifiedBy", meta["author_el"])
                core = set_property(core, "dc:title", meta["title_el"])
                data = core.encode("utf-8")
            elif item.filename == "docProps/app.xml":
                app = data.decode("utf-8")
                app = set_property(app, "Company", "")
                app = set_property(app, "Template", TEMPLATE_PROPERTY_NAME)
                app = set_property(app, "Manager", "", drop=True)
                data = app.encode("utf-8")
            elif item.filename in dropped:
                continue
            zout.writestr(item, data)
    tmp.replace(docx)
    if dropped:
        print("dropped comment/people parts:", ", ".join(sorted(dropped)))
    print("postprocessed metadata in", docx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", help="the department .dotx; required unless --post")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--post", action="store_true", help="only rewrite metadata of an existing --out (after Word)")
    args = ap.parse_args()
    if args.post:
        meta = yaml.safe_load((Path(args.src) / "meta.yaml").read_text(encoding="utf-8"))
        postprocess(Path(args.out), meta)
        return
    if not args.template:
        ap.error("--template is required for a build (only --post may omit it)")
    b = Builder(Path(args.template), Path(args.src), Path(args.repo), Path(args.out))
    b.build()
    print("built", args.out)
    print("stats", b.stats)
    log = Path(args.out).with_suffix(".figures.txt")
    log.write_text("\n".join(f"{f['number']}\t{f['file']}\tsvg={f['svg']}\t{f['caption']}" for f in b.figure_log), encoding="utf-8")


if __name__ == "__main__":
    main()
