import os
import argparse
import sys
import requests
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# This file was written with the help of ChatGPT (GPT-5)

load_dotenv()

# I'm using GitHub Models to access Openai API for "free" (there are usage limits of course)
# More info: https://github.blog/changelog/2025-05-19-github-models-built-into-your-repository-is-in-public-preview/
CUSTOM_BASE_URL = "https://models.github.ai/inference"
CUSTOM_API_KEY = os.getenv("MY_API_KEY")


def install_and_import(package, import_name=None):
    """Helper to install a package if missing, then import it."""
    import importlib
    try:
        return importlib.import_module(import_name or package)
    except ImportError:
        print(f"📦 Installing missing package: {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(import_name or package)


def load_source(source_path):
    """Load text content from a source (URL, pdf, docx, csv, or text file)."""
    text = ""
    try:
        if source_path.startswith("http://") or source_path.startswith("https://"):
            response = requests.get(source_path, timeout=10)
            response.raise_for_status()
            text = response.text

        elif source_path.endswith(".pdf"):
            PyPDF2 = install_and_import("PyPDF2")
            with open(source_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif source_path.endswith(".docx"):
            docx = install_and_import("python-docx", "docx")
            doc = docx.Document(source_path)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif source_path.endswith(".csv") or source_path.endswith(".txt"):
            with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        else:
            with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print(f"Warning: Could not load source '{source_path}'. Error: {e}")
    return text


def save_output(result_text, output_path):
    """Save output into txt, pdf, docx, or csv formats (auto-installs dependencies)."""
    ext = os.path.splitext(output_path)[1].lower()

    try:
        if ext == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_text)

        elif ext == ".pdf":
            reportlab = install_and_import("reportlab")
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            for line in result_text.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 12))

            doc.build(story)

        elif ext == ".docx":
            docx = install_and_import("python-docx", "docx")
            Document = docx.Document
            doc = Document()
            for line in result_text.split("\n"):
                doc.add_paragraph(line)
            doc.save(output_path)

        elif ext == ".csv":
            import csv
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for line in result_text.split("\n"):
                    row = [col.strip() for col in line.split(",")] if "," in line else [line]
                    writer.writerow(row)

        else:
            raise ValueError(f"Unsupported output format: {ext}")

        print(f"\n✅ Output saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error saving output to {output_path}: {e}")


def summarize_text(client, text, query=None):
    """Helper to summarize or answer a query on a given text chunk."""
    try:
        full_prompt = f"Context:\n{text}\n\nQuestion: {query or 'Summarize the content'}\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="A simple command-line utility to query OpenAI LLMs with data sources.",
        epilog="Example: python assignment-4.py notes.txt file.pdf -q \"Summarize these files\" -o summary.pdf"
    )
    parser.add_argument("sources", nargs='+', help="Data sources (URL, pdf, docx, csv, or text file path).")
    parser.add_argument("-q", "--query", type=str, help="The query. If not given, will summarize content.")
    parser.add_argument("-o", "--output", type=str, help="File path to save the output.")

    args = parser.parse_args()

    client = OpenAI(
        base_url=CUSTOM_BASE_URL,
        api_key=CUSTOM_API_KEY
    )

    # --- MAP STEP: Summarize each file individually ---
    summaries = []
    for source in args.sources:
        print(f"-> Processing: {source}")
        file_text = load_source(source)
        if not file_text.strip():
            print(f"⚠️ No text found in {source}, skipping.")
            continue

        # Truncate very long files to stay safe
        MAX_CHARS = 12000
        if len(file_text) > MAX_CHARS:
            print(f"⚠️ {source} is too long ({len(file_text)} chars). Truncating to {MAX_CHARS}.")
            file_text = file_text[:MAX_CHARS]

        summary = summarize_text(client, file_text, "Summarize this file in detail.")
        summaries.append(f"--- Summary of {source} ---\n{summary}\n")

    if not summaries:
        print("No usable summaries produced. Exiting.")
        sys.exit(1)

    # --- REDUCE STEP: Combine summaries into final answer ---
    combined_text = "\n".join(summaries)
    final_answer = summarize_text(client, combined_text, args.query)

    print("\n--- Final Result ---")
    print(final_answer)
    print("--------------------")

    # Save if needed
    if args.output:
        save_output(final_answer, args.output)


if __name__ == "__main__":
    main()
