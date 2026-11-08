"""Excel report: one row per document, plus a sheet with every receipt line item."""
from __future__ import annotations

import io

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from docsort.service import Result

HEADER_FILL = PatternFill("solid", fgColor="1F2937")


def safe_cell(value):
    """Stop spreadsheet formula injection (OWASP "CSV injection"): text that starts like a formula
    — e.g. a receipt that says '=HYPERLINK(...)' — is stored as plain text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _sheet(wb: Workbook, title: str, headers: list[str], rows: list[list], first: bool = False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
    for row in rows:
        ws.append([safe_cell(v) for v in row])
    for i, h in enumerate(headers, 1):
        longest = max([len(str(h))] + [len(str(r[i - 1])) for r in rows if r[i - 1] is not None])
        ws.column_dimensions[get_column_letter(i)].width = min(60, longest + 2)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def build_report(results: list[Result]) -> bytes:
    wb = Workbook()
    docs, items = [], []
    for r in results:
        f = r.fields or {}
        c = r.classification
        docs.append([r.filename, c.label, round(c.confidence * 100, 1), "yes" if c.unsure else "",
                     f.get("date"), f.get("merchant") or f.get("seller") or f.get("sender"),
                     f.get("invoice_number"), f.get("total"), f.get("currency"),
                     f.get("summary") or f.get("title"), r.extraction_error])
        for it in f.get("items") or []:
            items.append([r.filename, f.get("merchant"), it.get("name"), it.get("quantity"), it.get("price")])
    _sheet(wb, "Documents", ["File", "Type", "Confidence %", "Check manually", "Date", "From",
                             "Invoice no.", "Total", "Currency", "Summary", "Note"], docs, first=True)
    _sheet(wb, "Receipt items", ["File", "Merchant", "Item", "Quantity", "Price"], items)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
