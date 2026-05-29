"""
Diagnostic script for Method_Exploration.ipynb: why the final sns.lineplot
may show only two state lines.

Hypotheses tested:
1) Only a subset of swing states appear in clean...state_simple after filters.
2) State labels in the CSV do not exactly match the swing_states list (string mismatch).
3) Some states have so few rows or so few distinct dates that they are effectively absent
   from state_daily (or collapse visually).

Run: set ELECTION_CSV to your CSV path, or edit DEFAULT_CSV below.

Uses only the Python standard library for CSV + stats (no pandas), so MSYS2 base Python works.
Optional: matplotlib + seaborn for PNG figures; otherwise CSV summaries are written.

MSYS2 plotting: pacman -S mingw-w64-x86_64-python-matplotlib mingw-w64-x86_64-python-seaborn
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    mdates = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]
    HAS_PLOTTING = False

# ---------------------------------------------------------------------------
# Config (same swing states and column names as Peilin/Method_Exploration.ipynb)
# ---------------------------------------------------------------------------
DEFAULT_CSV = os.environ.get(
    "ELECTION_CSV",
    r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv",
)
CHUNK_SIZE = 100_000
SWING_STATES = [
    "Arizona",
    "Georgia",
    "Michigan",
    "Nevada",
    "North Carolina",
    "Pennsylvania",
    "Wisconsin",
]

COLS = frozenset(
    [
        "clean...state_simple",
        "created_at.users",
        "sampling_tweet",
    ]
)

CUTOFF_2024 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def parse_utc_datetime(s: str | None) -> datetime | None:
    """Parse created_at.users to timezone-aware UTC (best-effort, stdlib only)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    s_iso = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s_iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def normalize_utc_day(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def iter_csv_chunks(
    csv_path: str,
    max_chunks: int | None = None,
) -> Iterator[list[dict[str, str]]]:
    """Yield list-of-dicts chunks with only COLS keys (same idea as pandas usecols)."""
    n_chunks = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header_row = next(reader)
        except StopIteration:
            return
        header = [h.strip().lstrip("\ufeff") for h in header_row]
        idx = {name: header.index(name) for name in COLS if name in header}
        if len(idx) != len(COLS):
            missing = COLS - set(idx.keys())
            raise ValueError(f"CSV missing required columns: {missing}")

        buf: list[dict[str, str]] = []
        for row in reader:
            rec = {c: (row[idx[c]] if idx[c] < len(row) else "") for c in COLS}
            buf.append(rec)
            if len(buf) >= CHUNK_SIZE:
                yield buf
                buf = []
                n_chunks += 1
                if max_chunks is not None and n_chunks >= max_chunks:
                    return
        if buf:
            yield buf


def process_chunk_swing_rows(
    chunk: list[dict[str, str]],
) -> list[tuple[str, datetime]]:
    """Apply notebook filters; return list of (state, utc_datetime) for swing states."""
    out: list[tuple[str, datetime]] = []
    for rec in chunk:
        st = (rec.get("clean...state_simple") or "").strip()
        tw = (rec.get("sampling_tweet") or "").strip()
        created = rec.get("created_at.users")
        if not st or not tw or not (created and str(created).strip()):
            continue
        if st not in SWING_STATES:
            continue
        dt = parse_utc_datetime(created)
        if dt is None:
            continue
        out.append((st, dt))
    return out


def count_rows_per_state(
    csv_path: str, max_chunks: int | None = None
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for chunk in iter_csv_chunks(csv_path, max_chunks=max_chunks):
        for state, _dt in process_chunk_swing_rows(chunk):
            counts[state] += 1
    return {s: int(counts.get(s, 0)) for s in SWING_STATES}


def collect_state_date_stream(
    csv_path: str, max_chunks: int | None = None
) -> tuple[dict[str, tuple[datetime, datetime]], dict[str, set[date]]]:
    """
    Full pass: per-state min/max datetime; per-state set of UTC calendar dates in 2024+.
    """
    bounds: dict[str, list[datetime]] = {s: [] for s in SWING_STATES}
    days_2024: dict[str, set[date]] = {s: set() for s in SWING_STATES}

    for chunk in iter_csv_chunks(csv_path, max_chunks=max_chunks):
        for state, dt in process_chunk_swing_rows(chunk):
            bounds[state].append(dt)
            nd = normalize_utc_day(dt)
            if nd >= CUTOFF_2024:
                days_2024[state].add(nd.date())

    minmax: dict[str, tuple[datetime | None, datetime | None]] = {}
    for s in SWING_STATES:
        xs = bounds[s]
        if not xs:
            minmax[s] = (None, None)
        else:
            minmax[s] = (min(xs), max(xs))

    return minmax, days_2024


def top_raw_state_labels(csv_path: str, first_chunks: int = 3) -> dict[str, int]:
    """Value counts of clean...state_simple before swing filter (first N chunks only)."""
    counts: Counter[str] = Counter()
    for chunk in iter_csv_chunks(csv_path, max_chunks=first_chunks):
        for rec in chunk:
            st = (rec.get("clean...state_simple") or "").strip()
            tw = (rec.get("sampling_tweet") or "").strip()
            created = rec.get("created_at.users")
            if not st or not tw or not (created and str(created).strip()):
                continue
            counts[st] += 1
    return dict(counts.most_common(40))


def states_with_positive_rows(row_counts: dict[str, int]) -> list[str]:
    return [s for s in SWING_STATES if row_counts.get(s, 0) > 0]


def _write_csv_dict(path: Path, data: dict[str, int], value_header: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", value_header])
        for k in sorted(data.keys()):
            w.writerow([k, data[k]])


def plot_diagnostics(
    row_counts: dict[str, int],
    distinct_days: dict[str, int],
    out_dir: Path,
    raw_top: dict[str, int] | None = None,
    date_bounds: dict[str, tuple[datetime | None, datetime | None]] | None = None,
) -> None:
    """Save figures (or CSV only if matplotlib is unavailable)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PLOTTING or plt is None or sns is None or mdates is None:
        _write_csv_dict(out_dir / "rows_per_swing_state.csv", row_counts, "row_count")
        _write_csv_dict(out_dir / "distinct_days_2024plus.csv", distinct_days, "distinct_days")
        if date_bounds:
            with open(out_dir / "date_bounds_per_state.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["state", "min_date", "max_date"])
                for s in SWING_STATES:
                    lo, hi = date_bounds.get(s, (None, None))
                    w.writerow([s, lo.isoformat() if lo else "", hi.isoformat() if hi else ""])
        if raw_top:
            with open(out_dir / "top_state_labels_sample.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["state", "rows_in_sample"])
                for k, v in sorted(raw_top.items(), key=lambda x: -x[1]):
                    w.writerow([k, v])
        print(
            "\nmatplotlib/seaborn not installed — skipped PNG figures; wrote CSV summaries to",
            out_dir,
        )
        print(
            "MSYS2: pacman -S mingw-w64-x86_64-python-matplotlib mingw-w64-x86_64-python-seaborn"
        )
        return

    sns.set_theme(style="whitegrid")
    states = SWING_STATES
    rc = [row_counts[s] for s in states]
    colors = ["#2ecc71" if row_counts[s] > 0 else "#e74c3c" for s in states]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(states, rc, color=colors)
    ax1.set_title("Tweet rows per swing state (after notebook filters, chunked count)")
    ax1.set_xlabel("clean...state_simple")
    ax1.set_ylabel("Row count")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()
    fig1.savefig(out_dir / "rows_per_swing_state.png", dpi=150)
    plt.close(fig1)

    dd = [distinct_days[s] for s in states]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(states, dd, color="steelblue")
    ax2.set_title("Distinct calendar days with data per state (date >= 2024-01-01)")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Distinct days")
    ax2.tick_params(axis="x", rotation=45)
    ax2.axhline(1, color="orange", linestyle="--", linewidth=1, label="1 day (single point line)")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "distinct_days_per_state.png", dpi=150)
    plt.close(fig2)

    if raw_top:
        labels = list(raw_top.keys())
        vals = [raw_top[k] for k in labels]
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.barh(labels[::-1], vals[::-1], color="gray")
        ax3.set_title("Top state strings in CSV (sample chunks, before swing filter)")
        ax3.set_xlabel("Row count (partial file sample)")
        fig3.tight_layout()
        fig3.savefig(out_dir / "top_state_labels_sample.png", dpi=150)
        plt.close(fig3)

    if date_bounds:
        db = {
            s: date_bounds[s]
            for s in SWING_STATES
            if date_bounds[s][0] is not None and date_bounds[s][1] is not None
        }
        if db:
            fig4, ax4 = plt.subplots(figsize=(11, 6))
            keys = list(db.keys())
            for i, state in enumerate(keys):
                lo, hi = db[state]
                assert lo is not None and hi is not None
                ax4.plot([lo, hi], [i, i], color="teal", linewidth=4, solid_capstyle="round")
            ax4.set_yticks(range(len(keys)))
            ax4.set_yticklabels(keys)
            ax4.axvline(
                CUTOFF_2024,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Notebook: date >= 2024-01-01",
            )
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax4.set_xlabel("created_at.users")
            ax4.set_title(
                "Date range of tweets per swing state (if max < cutoff, state drops from state_daily)"
            )
            ax4.legend(loc="lower right")
            fig4.autofmt_xdate()
            fig4.tight_layout()
            fig4.savefig(out_dir / "date_span_per_state.png", dpi=150)
            plt.close(fig4)


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    max_chunks_env = os.environ.get("MAX_CHUNKS")
    max_chunks = int(max_chunks_env) if max_chunks_env else None

    out_dir = Path(__file__).resolve().parent / "test_missing_state_output"

    if not Path(csv_path).is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        print("Set ELECTION_CSV or pass path as first argument.")
        return 1

    print(f"Using CSV: {csv_path}")
    if max_chunks:
        print(f"MAX_CHUNKS={max_chunks} (subset run for speed)")

    print("\n--- Top clean...state_simple values (first 3 chunks, before swing filter) ---")
    raw_top = top_raw_state_labels(csv_path, first_chunks=3)
    for k, v in sorted(raw_top.items(), key=lambda x: -x[1])[:40]:
        print(f"{k:25s} {v}")

    print("\n--- Row counts per swing state (after filters, isin swing_states) ---")
    row_counts = count_rows_per_state(csv_path, max_chunks=max_chunks)
    for s in SWING_STATES:
        print(f"{s:18s} {row_counts[s]}")

    present = states_with_positive_rows(row_counts)
    print(f"\nStates with >0 rows: {len(present)} -> {present}")

    print("\n--- Min/max dates and distinct 2024+ days (full pass over filtered rows) ---")
    date_bounds, days_2024 = collect_state_date_stream(csv_path, max_chunks=max_chunks)
    distinct_days = {s: len(days_2024[s]) for s in SWING_STATES}

    print("\n--- Min / max tweet date per state (all rows; compare to 2024-01-01 cutoff) ---")
    for s in SWING_STATES:
        lo, hi = date_bounds[s]
        print(f"{s:18s} min={lo} max={hi}")

    print("\n--- Distinct days with >=1 row per state (date >= 2024-01-01) ---")
    for s in SWING_STATES:
        print(f"{s:18s} {distinct_days[s]}")

    n_rows_any_state = sum(1 for s in SWING_STATES if row_counts[s] > 0)
    n_states_after_2024 = sum(1 for s in SWING_STATES if distinct_days[s] > 0)
    print("\n--- Interpretation vs Method_Exploration.ipynb lineplot ---")
    print(
        "The notebook applies state_daily = state_daily[state_daily['date'] >= '2024-01-01']. "
        "States with no rows on or after that date disappear from state_daily, so sns.lineplot "
        "has fewer hue levels than the seven swing states."
    )
    print(
        f"Chunked counts: {n_rows_any_state} swing state(s) have >0 tweet rows after basic filters; "
        f"after the 2024-01-01 filter, {n_states_after_2024} state(s) have >=1 distinct calendar day."
    )
    if n_states_after_2024 <= 2:
        print(
            "Finding: Most swing-state volume may fall entirely before 2024-01-01 (see date_span_per_state.png). "
            "That yields at most a couple of lines in the final plot even when raw row counts look large."
        )
    elif n_states_after_2024 < 7:
        print(
            "Finding: Some states have tweets only before 2024-01-01 (distinct_days=0 for those); "
            "they are excluded from state_daily and do not appear in the lineplot."
        )
    if n_rows_any_state < 7:
        print(
            "Also check string mismatch: see top_state_labels_sample.png if some swing states never appear."
        )

    plot_diagnostics(
        row_counts,
        distinct_days,
        out_dir,
        raw_top=raw_top,
        date_bounds=date_bounds,
    )
    print(f"\nFigures saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
