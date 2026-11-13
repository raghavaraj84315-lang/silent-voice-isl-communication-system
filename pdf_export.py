"""Session transcript report generation (PDF, CSV, TXT)."""
import io
import os
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics, ttfonts
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_NAME = "Helvetica"
# Attempt to register Nirmala font for Indian Script support
NIRMALA_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
if os.path.exists(NIRMALA_PATH):
    try:
        pdfmetrics.registerFont(ttfonts.TTFont("Nirmala", NIRMALA_PATH, subfontIndex=0))
        pdfmetrics.registerFont(ttfonts.TTFont("Nirmala-Bold", NIRMALA_PATH, subfontIndex=1))
        FONT_NAME = "Nirmala"
    except Exception as e:
        print(f"[PDF Export] Could not register Nirmala font: {e}")


def generate_transcript_pdf(transcripts: List[Dict[str, Any]], username: str = "User") -> bytes:
    """Generate a clean PDF report for a session transcript."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName=f"{FONT_NAME}-Bold" if FONT_NAME == "Nirmala" else "Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=f"{FONT_NAME}-Bold" if FONT_NAME == "Nirmala" else "Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    signer_style = ParagraphStyle(
        "SignerCell",
        parent=cell_style,
        textColor=colors.HexColor("#2563eb"),
        fontName=f"{FONT_NAME}-Bold" if FONT_NAME == "Nirmala" else "Helvetica-Bold"
    )

    hearing_style = ParagraphStyle(
        "HearingCell",
        parent=cell_style,
        textColor=colors.HexColor("#059669"),
        fontName=f"{FONT_NAME}-Bold" if FONT_NAME == "Nirmala" else "Helvetica-Bold"
    )

    story = []

    # Title & Subtitle with Logo
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "logo_icon.png")
    if os.path.exists(logo_path):
        try:
            logo_img = RLImage(logo_path, width=44, height=44)
            header_table = Table(
                [[logo_img, [
                    Paragraph("Silent Voice — Two-Way Session Transcript Report", title_style),
                    Paragraph("Sign → Text → Voice | Indian Sign Language AI Hub", subtitle_style)
                ]]],
                colWidths=[54, 710]
            )
            header_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(header_table)
        except Exception:
            story.append(Paragraph("Silent Voice — Two-Way Session Transcript Report", title_style))
            story.append(Paragraph("Sign → Text → Voice | Indian Sign Language AI Hub", subtitle_style))
    else:
        story.append(Paragraph("Silent Voice — Two-Way Session Transcript Report", title_style))
        story.append(Paragraph("Sign → Text → Voice | Indian Sign Language AI Hub", subtitle_style))

    # Metadata Block
    now_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    meta_data = [
        [
            Paragraph(f"<b>Logged-in User:</b> {username}", meta_style),
            Paragraph(f"<b>Report Generated:</b> {now_str}", meta_style),
            Paragraph(f"<b>Total Statements:</b> {len(transcripts)}", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[200, 300, 200])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Transcripts Table
    table_data = [
        [
            Paragraph("#", th_style),
            Paragraph("Timestamp", th_style),
            Paragraph("Speaker", th_style),
            Paragraph("English (Polished / Sign)", th_style),
            Paragraph("Hindi (हिन्दी)", th_style),
            Paragraph("Kannada (ಕನ್ನಡ)", th_style)
        ]
    ]

    for idx, item in enumerate(transcripts, start=1):
        spk = item.get("speaker", "Signer")
        spk_style = hearing_style if "hearing" in spk.lower() else signer_style

        time_val = item.get("timestamp", "")
        # Shorten timestamp if full datetime
        if len(time_val) > 10:
            time_val = time_val.split(" ")[-1]

        en_text = item.get("english", "")
        if item.get("is_smoothed"):
            en_text += " [AI Polished]"

        table_data.append([
            Paragraph(str(idx), cell_style),
            Paragraph(time_val, cell_style),
            Paragraph(spk, spk_style),
            Paragraph(en_text, cell_style),
            Paragraph(item.get("hindi") or "—", cell_style),
            Paragraph(item.get("kannada") or "—", cell_style)
        ])

    if len(transcripts) == 0:
        table_data.append([
            Paragraph("—", cell_style),
            Paragraph("—", cell_style),
            Paragraph("—", cell_style),
            Paragraph("No session history recorded yet.", cell_style),
            Paragraph("—", cell_style),
            Paragraph("—", cell_style)
        ])

    col_widths = [25, 65, 85, 230, 180, 180]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t)

    doc.build(story)
    return buffer.getvalue()


def generate_transcript_csv(transcripts: List[Dict[str, Any]], username: str = "User") -> bytes:
    """Generate a clean CSV export with UTF-8 BOM for Excel/Sheets compatibility."""
    import codecs
    import csv

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Timestamp", "Speaker", "English Statement", "Hindi (हिन्दी)", "Kannada (ಕನ್ನಡ)", "AI Polished"])

    for idx, item in enumerate(transcripts, start=1):
        writer.writerow([
            idx,
            item.get("timestamp", ""),
            item.get("speaker", "Signer"),
            item.get("english", ""),
            item.get("hindi", ""),
            item.get("kannada", ""),
            "Yes" if item.get("is_smoothed") else "No"
        ])

    return codecs.BOM_UTF8 + output.getvalue().encode("utf-8")


def generate_transcript_txt(transcripts: List[Dict[str, Any]], username: str = "User") -> str:
    """Generate a plain text session transcript."""
    lines = [
        "=" * 70,
        "  SILENT VOICE — TWO-WAY SESSION TRANSCRIPT",
        f"  User: {username}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        ""
    ]

    for idx, item in enumerate(transcripts, start=1):
        spk = item.get("speaker", "Signer")
        ts = item.get("timestamp", "")
        en = item.get("english", "")
        hi = item.get("hindi", "")
        kn = item.get("kannada", "")
        smoothed = " [AI Polished]" if item.get("is_smoothed") else ""

        lines.append(f"[{idx}] {ts} — {spk.upper()}:")
        lines.append(f"    English: {en}{smoothed}")
        if hi:
            lines.append(f"    Hindi:   {hi}")
        if kn:
            lines.append(f"    Kannada: {kn}")
        lines.append("-" * 50)

    if not transcripts:
        lines.append("No transcript entries logged in this session.")

    return "\n".join(lines)

