#!/usr/bin/env python3
"""Crop build.pdf to its ink bounding box and write fig_provenance_graph.pdf.

Fallback cropper for environments without standalone.cls / pdfcrop.
Adds a small uniform margin around the detected content.
"""
import sys
import fitz  # PyMuPDF

MARGIN_PT = 5.0  # ~1.8mm border around the figure

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "build.pdf"
    dst = sys.argv[2] if len(sys.argv) > 2 else "fig_provenance_graph.pdf"
    doc = fitz.open(src)
    page = doc[0]
    # Union of all drawing + text bounding boxes = ink bbox.
    rect = None
    for d in page.get_drawings():
        r = d["rect"]
        rect = r if rect is None else (rect | r)
    text = page.get_text("rawdict")
    for block in text.get("blocks", []):
        if "bbox" in block:
            r = fitz.Rect(block["bbox"])
            rect = r if rect is None else (rect | r)
    # Also union word-level boxes: some standalone/rotated text nodes are
    # missed by the block bbox pass and would otherwise be clipped.
    for w in page.get_text("words"):
        r = fitz.Rect(w[:4])
        rect = r if rect is None else (rect | r)
    if rect is None:
        raise SystemExit("No content found on page 1")
    rect = fitz.Rect(rect.x0 - MARGIN_PT, rect.y0 - MARGIN_PT,
                     rect.x1 + MARGIN_PT, rect.y1 + MARGIN_PT)
    rect &= page.rect

    # Build a fresh document whose page IS the crop region, so the final
    # PDF has a tight mediabox (not just a cropbox). This keeps the figure
    # margin-free regardless of whether a downstream tool honors cropbox.
    out = fitz.open()
    newpage = out.new_page(width=rect.width, height=rect.height)
    newpage.show_pdf_page(newpage.rect, doc, 0, clip=rect)
    out.save(dst, garbage=4, deflate=True)
    print(f"Cropped {src} -> {dst}: "
          f"{rect.width:.1f} x {rect.height:.1f} pt")

if __name__ == "__main__":
    main()
