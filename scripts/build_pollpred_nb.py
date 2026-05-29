"""One-off builder: consolidate scripts/ into notebooks/Pollpred.ipynb."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NB_PATH = ROOT / "notebooks" / "Pollpred.ipynb"

DATA_CSV = r"C:\Users\Lucky\Desktop\WG1\subjects_AND_sampling_metadata_anonymized_full.csv"
MARGINS_CSV = str(ROOT / "data" / "trump_vs_harris_margins.csv").replace("\\", "\\\\")
DATA_DIR = str(ROOT / "data").replace("\\", "\\\\")

CONFIG_CELL = textwrap.dedent(f"""
    from pathlib import Path
    import os

    PROJECT_ROOT = Path(r"{ROOT}")
    DATA_CSV = r"{DATA_CSV}"
    MARGINS_CSV = r"{ROOT / 'data' / 'trump_vs_harris_margins.csv'}"
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["ELECTION_CSV"] = DATA_CSV
""").strip()


def read_script(rel: str) -> str:
    return (SCRIPTS / rel).read_text(encoding="utf-8")


def patch_paths(code: str) -> str:
    replacements = [
        (r'file_path = r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"', "file_path = DATA_CSV"),
        (r'file_path = r"E:\\Csci_3\\subjects_AND_sampling_metadata_anonymized_full.csv"', "file_path = DATA_CSV"),
        (r'pd.read_csv("E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"', "pd.read_csv(DATA_CSV"),
        (r'pd.read_csv(r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"', "pd.read_csv(DATA_CSV"),
        ('FILE_IN = "subjects_AND_sampling_metadata_anonymized_full.csv"', "FILE_IN = DATA_CSV"),
        ('file_path = "subjects_AND_sampling_metadata_anonymized_full.csv"', "file_path = DATA_CSV"),
        (r'r"E:\Csci_3\US_Election\data\trump_vs_harris_margins.csv"', "MARGINS_CSV"),
        (r'r"E:\Csci_3\US_Election\Peilin\trump_vs_harris_margins.csv"', "MARGINS_CSV"),
    ]
    for old, new in replacements:
        code = code.replace(old, new)
    code = code.replace('file_path = r"DATA_CSV"', "file_path = DATA_CSV")
    code = code.replace('to_csv("comments_with_topics.csv"', "to_csv(DATA_DIR / 'comments_with_topics.csv'")
    code = code.replace('to_csv("topic_summary.csv"', "to_csv(DATA_DIR / 'topic_summary.csv'")
    code = code.replace('to_csv("state_topic_counts.csv"', "to_csv(DATA_DIR / 'state_topic_counts.csv'")
    code = code.replace('pd.read_csv("state_topic_counts.csv")', "pd.read_csv(DATA_DIR / 'state_topic_counts.csv')")
    code = code.replace('pd.read_csv("comments_with_topics.csv")', "pd.read_csv(DATA_DIR / 'comments_with_topics.csv')")
    code = code.replace('to_csv("swing_state_topic_counts_only.csv"', "to_csv(DATA_DIR / 'swing_state_topic_counts_only.csv'")
    code = code.replace('to_csv("swing_state_topic_counts.csv"', "to_csv(DATA_DIR / 'swing_state_topic_counts.csv'")
    code = code.replace('FILE_OUT = "subjects_cleaned_basic.csv"', "FILE_OUT = DATA_DIR / 'subjects_cleaned_basic.csv'")
    return code


def _poll_prediction_section() -> str:
    marker = "# ================================\n# 10. LOAD FINAL ELECTION"
    full = read_script("analysis/Methods_difindif.py")
    # First copy only (sections 10–14); second copy in Methods_difindif is duplicate
    block = full.split(marker, 1)[1].split(marker, 1)[0]
    trajectory = read_script("pollpred_extras/poll_trajectory.py")
    return marker + block + "\n" + trajectory


def patch_test_missing_state(code: str) -> str:
    code = code.replace(
        "DEFAULT_CSV = os.environ.get(\n    \"ELECTION_CSV\",\n    r\"E:\\Csci_3\\subjects_AND_sampling_metadata_anonymized_full.csv\",\n)",
        "DEFAULT_CSV = os.environ.get(\"ELECTION_CSV\", DATA_CSV)",
    )
    code = code.replace(
        "out_dir = Path(__file__).resolve().parent / \"test_missing_state_output\"",
        "out_dir = DATA_DIR / \"test_missing_state_output\"",
    )
    code = code.replace("if __name__ == \"__main__\":\n    sys.exit(main())", "main()")
    return code


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def build() -> dict:
    workflow = read_script("Project_workflow.py").strip().strip("'''").strip()
    cells = [
        md("# Pollpred — US Election Twitter Analysis\n\nConsolidated notebook from `scripts/`."),
        md("## Project workflow\n\n```\n" + workflow + "\n```"),
        code(CONFIG_CELL),
        md("## Exploration — standard file reading"),
        code(patch_paths(read_script("exploration/Standard_filreading_procedure.py"))),
        md("## Missingness — chunk-based column missing %"),
        code(patch_paths(read_script("missingness/MissingVals.py"))),
        md("## Missingness — MCAR / MAR / MNAR classification"),
        code(patch_paths(read_script("missingness/Missingdata.py"))),
        md("## Missingness — swing-state coverage diagnostics"),
        code(patch_paths(patch_test_missing_state(read_script("missingness/test_missing_state.py")))),
        md("## Data cleaning"),
        code(patch_paths(read_script("cleaning/ana_clean.py"))),
        md("## Feature engineering — swing-state tweets (chunked)"),
        code(patch_paths(read_script("analysis/Methods_fixed.py"))),
        md("## Topic modeling (LDA)"),
        code(patch_paths(read_script("topic_modeling/Basic_view.py"))),
        md("## Topic modeling — swing states only"),
        code(patch_paths(read_script("topic_modeling/swingonly.py") + "\n\n" + read_script("topic_modeling/swingstates.py"))),
        md("## Difference-in-differences & daily panel"),
        code(patch_paths(read_script("analysis/Methods_difindif.py").split("# ================================\n# 10. LOAD FINAL ELECTION")[0].rstrip())),
        md("## Poll prediction & trajectory"),
        code(patch_paths(_poll_prediction_section())),
    ]

    extras_dir = SCRIPTS / "pollpred_extras"
    extras = [
        ("## Bias audit — engagement concentration (panel)", "cell2.py"),
        ("## Bias audit — representation", "cell3.py"),
        ("## Bias audit — viral spike inequality", "cell4.py"),
        ("## Bias audit — missingness by state", "cell5.py"),
        ("## Bias audit — amplification & DiD spike sensitivity", "cell6.py"),
    ]
    for title, fname in extras:
        path = extras_dir / fname
        if path.is_file():
            cells.append(md(title))
            cells.append(code(path.read_text(encoding="utf-8")))

    did = read_script("analysis/Methods_difindif.py")
    start = did.index("# ##################################################################")
    cells.append(md("## Systematic bias audit (Methods_difindif sections 16–20)"))
    cells.append(code(patch_paths(did[start:])))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NB_PATH} ({len(nb['cells'])} cells)")
