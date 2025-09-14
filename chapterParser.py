import json, re
from pathlib import Path
from langchain_core.documents import Document

HEADING_RE = re.compile(r'^\s{0,3}#\s+(.*)$')   # H1 = Chapter

def _flush_paragraph(buf_lines: list[str]) -> str | None:
    text = "\n".join(buf_lines).strip()
    return text if text else None

def _parse_table_block(lines: list[str], start_idx: int) -> tuple[list[dict], int]:
    """
    Very simple GitHub-style pipe table parser.
    Returns (rows_list, next_index).
    """
    i = start_idx
    block = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        block.append(lines[i].rstrip("\n"))
        i += 1
    if not block:
        return [], start_idx

    header = [c.strip() for c in block[0].strip("|").split("|")]
    if len(block) >= 2 and set(block[1].replace(" ", "")) <= {"|", ":", "-"}:
        data_rows = block[2:]
    else:
        data_rows = block[1:]

    rows = []
    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        d = {}
        for idx, val in enumerate(cells):
            key = header[idx] if idx < len(header) and header[idx] else f"col_{idx}"
            d[key] = val
        rows.append(d)
    return rows, i

IMG_INLINE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def parse_markdown_to_documents_with_chapter(md_path: str) -> list[Document]:
    """
    Scan the markdown file line-by-line, track current H1 as `chapter`,
    and emit Documents for:
      - paragraph blocks
      - each table row (as JSON string, like your previous helper)
      - image tags (alt text as content, plus image_url metadata)
    """
    docs: list[Document] = []
    p = Path(md_path)
    lines = p.read_text(encoding="utf8").splitlines()

    current_chapter = p.stem
    paragraph_buf: list[str] = []
    i = 0

    def push_paragraph():
        txt = _flush_paragraph(paragraph_buf)
        paragraph_buf.clear()
        if txt:
            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": md_path, "chapter": current_chapter}
                )
            )

    while i < len(lines):
        line = lines[i]

        # Chapter (H1) heading
        m = HEADING_RE.match(line)
        if m:
            push_paragraph()
            current_chapter = m.group(1).strip() or current_chapter
            i += 1
            continue

        if line.strip().startswith("|"):
            push_paragraph()
            rows, next_i = _parse_table_block(lines, i)
            for row in rows:
                docs.append(
                    Document(
                        page_content=json.dumps(row, ensure_ascii=False),
                        metadata={"source": md_path, "chapter": current_chapter}
                    )
                )
            i = next_i
            continue

        for alt_text, rel_path in IMG_INLINE_RE.findall(line):
            img_path = (p.parent / rel_path).resolve()
            docs.append(
                Document(
                    page_content=alt_text or "",
                    metadata={
                        "source": md_path,
                        "image_url": img_path.as_uri(),
                        "chapter": current_chapter
                    }
                )
            )

        if line.strip() == "":
            push_paragraph()
        else:
            paragraph_buf.append(line)

        i += 1

    push_paragraph()

    return docs
