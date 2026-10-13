"""Format-specific parsers. Each returns a list of Blocks in reading order.
Tables become one block per row, rendered 'Header: value; Header: value' so every row
is self-describing even when it lands in a chunk on its own."""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("veltrix.ingest")


@dataclass
class Block:
    text: str
    heading: Optional[str]  # nearest section heading
    page: int = 1
    kind: str = "text"  # text | table_row | heading


def _clean(s: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", s or "").strip()


def render_row(header: list[str], row: list[str]) -> str:
    row = [_clean(c) for c in row]
    if not any(row):
        return ""
    if header and len(header) == len(row) and any(header):
        return "; ".join(f"{h}: {v}" if h else v for h, v in zip(header, row) if v)
    return " | ".join(c for c in row if c)


def _rows_to_blocks(rows, heading, page) -> list[Block]:
    rows = [[_clean(str(c or "")) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [Block(" | ".join(header), heading, page, "table_row")]
    return [Block(t, heading, page, "table_row") for t in (render_row(header, r) for r in body) if t]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not (2 < len(line) <= 90) or not re.search(r"[A-Za-z]", line):
        return False
    if line.endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\.?\s", line):
        return False
    return len(line.split()) <= 12


# ---------------------------------------------------------------- PDF
def parse_pdf(path: Path, merge_lines: bool = True) -> list[Block]:
    import fitz  # PyMuPDF

    blocks: list[Block] = []
    heading: Optional[str] = None
    for pno, page in enumerate(fitz.open(path), start=1):
        if len(page.get_text("text").strip()) < 25:  # no text layer -> OCR
            ocr, heading = _ocr_page(page, pno, heading)
            blocks.extend(ocr)
            continue
        table_boxes, table_items = [], []
        try:
            for tab in page.find_tables().tables:
                table_boxes.append(fitz.Rect(tab.bbox))
                for b in _rows_to_blocks(tab.extract(), None, pno):
                    table_items.append((tab.bbox[1], b))
        except Exception as e:  # best effort
            logger.info("table_detect_failed", extra={"fields": {"file": path.name, "err": str(e)[:100]}})
        lines, sizes = [], []
        for b in page.get_text("dict").get("blocks", []):
            for ln in b.get("lines", []):
                spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                r = fitz.Rect(ln["bbox"])
                mid = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
                if any(tb.contains(mid) for tb in table_boxes):
                    continue
                size = max(s["size"] for s in spans)
                bold = all(("bold" in s.get("font", "").lower()) or (s.get("flags", 0) & 16) for s in spans)
                lines.append((r.y0, _clean("".join(s["text"] for s in spans)), size, bold))
                sizes.append(size)
        body = statistics.median(sizes) if sizes else 10
        items = [(y, Block(t, t, pno, "heading") if (sz >= body * 1.15 or bold) and _looks_like_heading(t)
                  else Block(t, None, pno, "text")) for y, t, sz, bold in lines]
        items.extend(table_items)
        items.sort(key=lambda t: t[0])
        for _, b in items:
            if b.kind == "heading":
                heading = b.text
            else:
                b.heading = heading
            blocks.append(b)
    return _merge_text_lines(blocks) if merge_lines else blocks

def _find_tesseract():
    """Locate tesseract: env var, PATH, then common Windows folders."""
    import os
    import shutil

    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _ocr_page(page, pno: int, heading):
    import io

    import pytesseract
    from PIL import Image

    cmd = _find_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        
    img = Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))
    text = pytesseract.image_to_string(img)
    logger.info("ocr_page", extra={"fields": {"page": pno, "chars": len(text)}})
    out = []
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        good = sum(ch.isalnum() or ch.isspace() for ch in line)
        if len(line) < 4 or good / len(line) < 0.8:
            continue  # OCR noise, e.g. signatures or logos
        is_head = _looks_like_heading(line) and len(line.split()) <= 8 and (
            line.isupper() or line.istitle() or re.match(r"^\d+(\.\d+)*\.?\s", line))
        if is_head:
            heading = line
            out.append(Block(line, line, pno, "heading"))
        else:
            out.append(Block(line, heading, pno, "text"))
    return out, heading


def _merge_text_lines(blocks: list[Block]) -> list[Block]:
    """Join wrapped text lines sharing heading+page into paragraphs."""
    out: list[Block] = []
    for b in blocks:
        if (out and b.kind == "text" and out[-1].kind == "text" and out[-1].heading == b.heading
                and out[-1].page == b.page and not re.match(r"^([-•*]|\d+[.)])\s", b.text)
                and len(out[-1].text) < 600):
            out[-1].text += " " + b.text
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------- DOCX
def parse_docx(path: Path) -> list[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    blocks: list[Block] = []
    heading = None
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = _clean(p.text)
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            runs = [r for r in p.runs if r.text.strip()]
            if style.lower().startswith(("heading", "title")) or (runs and all(r.bold for r in runs)
                                                                   and _looks_like_heading(text)):
                heading = text
                blocks.append(Block(text, text, 1, "heading"))
            else:
                blocks.append(Block(text, heading, 1, "text"))
        elif tag == "tbl":
            rows = []
            for r in Table(el, d).rows:
                cells, prev = [], None
                for c in r.cells:  # merged cells repeat
                    if prev is not None and c._tc is prev:
                        continue
                    prev = c._tc
                    cells.append(c.text)
                rows.append(cells)
            blocks.extend(_rows_to_blocks(rows, heading, 1))
    return blocks


# ---------------------------------------------------------------- HTML
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre", "blockquote", "dt", "dd", "div",
               "section"]


def parse_html(path: Path) -> list[Block]:
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    blocks: list[Block] = []
    heading = None
    for el in (soup.body or soup).find_all(_BLOCK_TAGS):
        name = el.name
        if name != "table" and el.find_parent("table"):
            continue
        if name in ("div", "section") and el.find(_BLOCK_TAGS):
            continue
        if name in ("p", "li", "dd", "dt", "blockquote", "pre") and el.find_parent(["li", "p", "blockquote"]):
            continue
        if name == "table":
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            blocks.extend(_rows_to_blocks(rows, heading, 1))
            continue
        text = _clean(el.get_text(" ", strip=True))
        if not text:
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading = text
            blocks.append(Block(text, text, 1, "heading"))
        else:
            blocks.append(Block(text, heading, 1, "text"))
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):  # comments are content too (kept as data)
        if _clean(c):
            blocks.append(Block(_clean(c), heading, 1, "text"))
    return blocks


# ---------------------------------------------------------------- Markdown
def parse_markdown(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: list[Block] = []
    heading = None
    para: list[str] = []

    def flush():
        if para:
            blocks.append(Block(_clean(" ".join(para)), heading, 1, "text"))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            heading = _clean(m.group(2).strip("# "))
            blocks.append(Block(heading, heading, 1, "heading"))
        elif line.strip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            blocks.extend(_rows_to_blocks(rows, heading, 1))
            continue
        elif not line.strip():
            flush()
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            flush()
            para.append(re.sub(r"^\s*([-*+]|\d+[.)])\s+", "- ", line))
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return blocks


def parse_any(path: Path, merge_lines: bool = True) -> list[Block]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, merge_lines)
    if ext == ".docx":
        return parse_docx(path)
    if ext in (".html", ".htm"):
        return parse_html(path)
    if ext in (".md", ".markdown", ".txt"):
        return parse_markdown(path)
    raise ValueError(f"unsupported format: {ext}")
