from docx import Document
import re
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

CODE_FONTS = ["courier new", "consolas"]
CODE_FILL_COLORS = {"1e1e1e", "1e1e2e", "f8f0ff"}
FENCE = "`" * 3

HEADING_STYLES = ["heading 1", "heading 2", "heading 3"]
MIN_HEADING_PT = 12
MIN_SECTION_CHARS = 100
MAX_CHUNK_CHARS = 800

HANDS_ON_PATTERN = re.compile(r"\b(hands[- ]on|labs?|exercises?|practice problems?)\b", re.IGNORECASE)

SKIP_TITLE = re.compile(
    r"\bpre-?\s?reads?\b"
    r"|quick reference"
    r"|^(\d+(\.\d+)*\.?\s*)?(reference material|optional deep dives?|further reading|additional reading)"
    r"|^contents?\s*(\(part \d+\))?\s*$",
    re.I,
)

def iter_block_items(doc):
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def table_to_text(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    if len(rows) == 1 or len(rows[0]) == 1:
        return "\n".join(" | ".join(c for c in r if c) for r in rows)

    header = rows[0]
    lines = []
    for r in rows[1:]:
        parts = [f"{h}: {v}" for h, v in zip(header, r) if v]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def get_fill(table):
    tcPr = table.rows[0].cells[0]._tc.tcPr
    shd = tcPr.find(qn("w:shd")) if tcPr is not None else None
    fill = shd.get(qn("w:fill")) if shd is not None else None
    return fill.lower() if fill else None


def runs_are_monospace(runs):
    runs = [r for r in runs if r.text.strip() != ""]
    if not runs:
        return False
    return all((r.font.name or "").lower() in CODE_FONTS for r in runs)


def is_code_paragraph(paragraph):
    return runs_are_monospace(paragraph.runs)


def is_code_box(table):
    if len(table.rows) != 1 or len(table.columns) != 1:
        return False
    if get_fill(table) in CODE_FILL_COLORS:
        return True
    cell = table.rows[0].cells[0]
    runs = [r for p in cell.paragraphs for r in p.runs]
    return runs_are_monospace(runs)


def is_bold_sized_heading(paragraph, stripped_text):
    if len(stripped_text) >= 100:
        return False
    runs = [r for r in paragraph.runs if r.text.strip() != ""]
    if not runs or not all(r.bold for r in runs):
        return False
    sizes = [r.font.size.pt for r in runs if r.font.size is not None]
    if not sizes:
        return False
    return max(sizes) >= MIN_HEADING_PT


def flush_code(code_buffer, current_text):
    if code_buffer:
        current_text.append(FENCE + "\n" + "\n".join(code_buffer) + "\n" + FENCE)
        code_buffer.clear()


def make_section(lecture_name, title, lines, default_content_type):
    content_type = "hands_on" if HANDS_ON_PATTERN.search(title) else default_content_type
    return {
        "lecture": lecture_name,
        "section": title,
        "text": "\n".join(lines).strip(),
        "content_type": content_type,
    }


def is_reading_section(section_title):
    return bool(SKIP_TITLE.search(section_title.strip()))


def merge_tiny_sections(sections, min_chars=MIN_SECTION_CHARS):
    merged = []
    carry = ""
    for sec in sections:
        text = sec["text"]
        if carry:
            text = carry + "\n" + text
            carry = ""
        if len(text) < min_chars:
            carry = text
            continue
        merged.append({**sec, "text": text})
    if carry:
        if merged:
            merged[-1]["text"] = merged[-1]["text"] + "\n" + carry
        else:
            merged.append({**sections[-1], "text": carry})
    return merged


def parse_lecture_docx(file_path, lecture_name, default_content_type="theory"):
    doc = Document(file_path)
    heading_pattern = re.compile(r'^\d+\.\s')

    sections = []
    parent_title = "Introduction"
    current_title = "Introduction"
    current_text = []
    code_buffer = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            raw = block.text
            text = raw.strip()

            if text == "" or "▬▬▬" in text:
                continue

            if is_code_paragraph(block):
                code_buffer.append(raw.rstrip())
                continue

            flush_code(code_buffer, current_text)

            style_name = (block.style.name or "").lower() if block.style is not None else ""
            is_style_heading = style_name in HEADING_STYLES
            is_numbered_heading = (raw == raw.lstrip()) and bool(heading_pattern.match(text)) and len(text) < 100
            is_sub_heading = (
                not is_style_heading
                and not is_numbered_heading
                and is_bold_sized_heading(block, text)
            )

            if is_style_heading or is_numbered_heading or is_sub_heading:
                if current_text:
                    sections.append(make_section(lecture_name, current_title, current_text, default_content_type))
                if is_sub_heading:
                    current_title = parent_title + " > " + text
                else:
                    parent_title = text
                    current_title = text
                current_text = []
            else:
                current_text.append(text)

        elif isinstance(block, Table):
            flush_code(code_buffer, current_text)
            table_text = table_to_text(block)
            if table_text.strip() == "" or "▬▬▬" in table_text:
                continue
            if is_code_box(block):
                table_text = FENCE + "\n" + table_text + "\n" + FENCE
            current_text.append(table_text)

    flush_code(code_buffer, current_text)
    if current_text:
        sections.append(make_section(lecture_name, current_title, current_text, default_content_type))

    return merge_tiny_sections(sections)


def text_to_units(text):
    units = []
    in_code = False
    code_lines = []
    for line in text.split("\n"):
        if line.strip() == FENCE:
            if in_code:
                code_lines.append(line)
                units.append("\n".join(code_lines))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = [line]
        elif in_code:
            code_lines.append(line)
        else:
            units.append(line)
    if code_lines:
        units.append("\n".join(code_lines))
    return units


def split_long_text(text, max_chars):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            parts.append(current)
            current = sentence
        elif current:
            current = current + " " + sentence
        else:
            current = sentence
    if current:
        parts.append(current)
    return parts


def split_big_unit(unit, max_chars):
    if len(unit) <= max_chars:
        return [unit]

    if not unit.startswith(FENCE):
        return split_long_text(unit, max_chars)

    lines = unit.split("\n")[1:-1]
    room = max_chars - (2 * len(FENCE) + 2)

    parts = []
    current = ""
    for line in lines:
        if current and len(current) + 1 + len(line) > room:
            parts.append(current)
            current = line
        elif current:
            current = current + "\n" + line
        else:
            current = line
    if current:
        parts.append(current)

    return [FENCE + "\n" + p + "\n" + FENCE for p in parts]


def split_section(sec, max_chars=MAX_CHUNK_CHARS):
    if len(sec["text"]) <= max_chars:
        return [sec]

    units = []
    for unit in text_to_units(sec["text"]):
        units.extend(split_big_unit(unit, max_chars))

    chunks = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        elif current:
            current = current + "\n" + unit
        else:
            current = unit
    if current:
        chunks.append(current)

    if len(chunks) == 1:
        return [sec]

    return [
        {**sec, "section": f'{sec["section"]} (part {n})', "text": c}
        for n, c in enumerate(chunks, start=1)
    ]


def split_oversized_sections(sections, max_chars=MAX_CHUNK_CHARS):
    result = []
    for sec in sections:
        result.extend(split_section(sec, max_chars))
    return result


if __name__ == "__main__":
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import LECTURES_DIR

    if not os.path.isdir(LECTURES_DIR):
        print(f"No '{LECTURES_DIR}' folder found. Move your .docx lecture files there first.")
    else:
        docx_files = [
            f for f in os.listdir(LECTURES_DIR)
            if f.lower().endswith(".docx") and not f.startswith("~$")
        ]
        if not docx_files:
            print(f"'{LECTURES_DIR}' exists but has no .docx files in it yet.")

        total_chunks = 0
        total_over = 0

        for filename in sorted(docx_files):
            path = os.path.join(LECTURES_DIR, filename)
            lecture_name = os.path.splitext(filename)[0]

            sections = parse_lecture_docx(path, lecture_name)
            chunks = split_oversized_sections(sections)
            over = [c for c in chunks if len(c["text"]) > MAX_CHUNK_CHARS]

            total_chunks += len(chunks)
            total_over += len(over)

            print("=" * 80)
            print("FILE:", filename)
            print("Sections:", len(sections), "| Chunks:", len(chunks), "| Largest:", max(len(c["text"]) for c in chunks))
            print("Code blocks:", sum(c["text"].count(FENCE) // 2 for c in chunks), "| hands_on:", sum(1 for s in sections if s["content_type"] == "hands_on"))
            if over:
                print("Chunks over", MAX_CHUNK_CHARS, ":", len(over))
                for c in over[:5]:
                    print(f"    {len(c['text']):>5} | {c['section']} | {c['text'][:80]!r}")

        print("=" * 80)
        print("TOTAL chunks:", total_chunks, "| TOTAL over limit:", total_over)