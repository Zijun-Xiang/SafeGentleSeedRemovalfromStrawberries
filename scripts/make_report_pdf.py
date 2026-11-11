"""Converts report/REPORT.md to report/REPORT.html and report/REPORT.pdf.

Needs pandoc, and Chrome/Chromium or wkhtmltopdf for the PDF step.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "report"

CSS = """
body { font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #1b1f23; max-width: 900px; margin: 0 auto; padding: 0 18px; }
h1 { font-size: 20pt; margin-bottom: 4px; }
h2 { font-size: 14.5pt; margin-top: 26px; border-bottom: 1px solid #d0d7de; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 20px; page-break-after: avoid; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
table { border-collapse: collapse; margin: 10px 0; font-size: 9.5pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f3f5f7; }
code { font-family: "JetBrains Mono", Consolas, monospace; font-size: 9pt; background: #f3f5f7; padding: 1px 3px; border-radius: 3px; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 22px 0; }
@page { size: A4; margin: 15mm 13mm; }
"""


def main():
    md = REP / "REPORT.md"
    html = REP / "REPORT.html"
    css = REP / "report.css"
    css.write_text(CSS)
    subprocess.run(["pandoc", str(md), "-f", "gfm", "-t", "html5", "-s", "--metadata", "title=Safe, Gentle Seed Removal: simulation report",
                    "--css", "report.css", "-o", str(html)], check=True, cwd=REP)
    text = html.read_text()
    # pandoc repeats the metadata title as a header block; the document has its own first heading
    start = text.find('<header id="title-block-header">')
    if start >= 0:
        end = text.find("</header>", start) + len("</header>")
        text = text[:start] + text[end:]
        html.write_text(text)
    pdf = REP / "REPORT.pdf"
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser", "chrome") if shutil.which(c)), None)
    if chrome:
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    elif shutil.which("wkhtmltopdf"):
        subprocess.run(["wkhtmltopdf", "--enable-local-file-access", str(html), str(pdf)], check=True)
    else:
        print("no PDF tool found; wrote", html)
        return 0
    print("wrote", pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
