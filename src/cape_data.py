"""
Utilities for building the CAPE behaviour dataset in a reusable way.

This module mirrors the logic from the behaviour baseline notebook:
  * load public_labels.csv
  * match rows to JSON reports in public_small_reports/
  * extract API-like tokens from behaviour + static imports
  * normalise tokens
  * apply selection + filtering (missing/unknown family, broken JSON, etc.)
  * enforce MIN_API_LEN
  * return a clean DataFrame with:
        sha256, family, api_seq, num_calls, api_text_full
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

def get_cape_paths(project_dir: Path) -> Tuple[Path, Path]:
    """
    Return (labels_path, reports_dir) for the CAPE dataset.

    Assumes the structure:
        project_dir /
            data /
                cape /
                    public_labels.csv
                    public_small_reports / <sha256>.json
    """
    data_dir = project_dir / "data" / "cape"
    labels_path = data_dir / "public_labels.csv"
    reports_dir = data_dir / "public_small_reports"
    if not labels_path.exists():
        raise FileNotFoundError(f"CAPE labels not found at: {labels_path}")
    if not reports_dir.exists():
        raise FileNotFoundError(f"CAPE reports dir not found at: {reports_dir}")
    return labels_path, reports_dir


# -------------------------------------------------------------------
# Normalisation + API sequence extraction
# -------------------------------------------------------------------

def normalise_api_name(name: str) -> str:
    """
    Normalise a raw API / function / token string.

    Steps:
      * keep only the part before the first '('
      * strip whitespace
      * lowercase
    """
    if not isinstance(name, str):
        return ""
    base = name.split("(", 1)[0]
    base = base.strip().lower()
    return base


def _collect_from_summary(node, out_seq: List[str]) -> None:
    """
    Recursive helper to pull string-like tokens out of behaviour.summary.

    Handles dicts, lists and strings.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            k_norm = normalise_api_name(k)
            if k_norm:
                out_seq.append(k_norm)
            _collect_from_summary(v, out_seq)
    elif isinstance(node, list):
        for item in node:
            _collect_from_summary(item, out_seq)
    elif isinstance(node, str):
        s_norm = normalise_api_name(node)
        if s_norm:
            out_seq.append(s_norm)


def extract_api_sequence(report: Dict) -> List[str]:
    """
    Extract a flat list of API-like tokens from a CAPE JSON report.

    Sources:
      * behaviour.processes[*].calls[*].api / .call
      * behaviour.apistats (expanded according to counts)
      * behaviour.summary (keys and string leaves)
      * static.pe_imports / static.imports (as a fallback)
    """
    seq: List[str] = []

    behavior = report.get("behavior") or {}

    # 1) processes[*].calls[*].api / call
    for proc in behavior.get("processes", []) or []:
        calls = proc.get("calls", []) or []
        for call in calls:
            if not isinstance(call, dict):
                continue
            api_raw = call.get("api") or call.get("call")
            api_norm = normalise_api_name(api_raw)
            if api_norm:
                seq.append(api_norm)

    # 2) apistats: behaviour["apistats"][process][api_name] = count
    apistats = behavior.get("apistats") or {}
    for _proc_name, stats in apistats.items():
        if not isinstance(stats, dict):
            continue
        for api_name, count in stats.items():
            api_norm = normalise_api_name(api_name)
            if not api_norm:
                continue
            # repeat according to count, but don't go crazy
            if isinstance(count, int) and count > 0:
                repeat = min(count, 50)
                seq.extend([api_norm] * repeat)

    # 3) summary: recursive walk over keys and strings
    summary = behavior.get("summary")
    if summary is not None:
        _collect_from_summary(summary, seq)

    # 4) static imports as a final fallback
    static = report.get("static") or {}
    for key in ("pe_imports", "imports"):
        imports = static.get(key)
        if isinstance(imports, list):
            for entry in imports:
                # In many CAPE reports, this is a list of dicts with "imports"
                if isinstance(entry, str):
                    nm_norm = normalise_api_name(entry)
                    if nm_norm:
                        seq.append(nm_norm)
                elif isinstance(entry, dict):
                    for sub in entry.get("imports", []) or []:
                        if isinstance(sub, dict):
                            nm_norm = normalise_api_name(sub.get("name"))
                            if nm_norm:
                                seq.append(nm_norm)
        elif isinstance(imports, dict):
            for nm in imports.keys():
                nm_norm = normalise_api_name(nm)
                if nm_norm:
                    seq.append(nm_norm)

    # Final clean-up: drop empty strings
    seq = [tok for tok in seq if tok]
    return seq


# -------------------------------------------------------------------
# Build cape_df with filtering
# -------------------------------------------------------------------

def build_cape_df(
    project_dir: Path,
    min_api_len: int = 10,
) -> pd.DataFrame:
    """
    Build the main CAPE DataFrame used throughout the behaviour + hybrid work.

    Filtering steps:
      * match labels to existing report JSONs
      * drop rows with missing/blank/"unknown" family
      * deduplicate by sha256
      * skip broken JSONs (JSONDecodeError)
      * extract API sequence and drop samples with len(api_seq) < min_api_len

    Returns a shuffled DataFrame with columns:
      sha256, family, api_seq, num_calls, api_text_full
    """
    labels_path, reports_dir = get_cape_paths(project_dir)
    labels_df = pd.read_csv(labels_path)

    rows = []
    seen_hashes = set()

    for row in labels_df.itertuples(index=False):
        sha256 = getattr(row, "sha256", None)
        family = getattr(row, "classification_family", None)

        # family sanity: non-empty, non-"unknown"
        if not isinstance(family, str):
            continue
        fam_clean = family.strip()
        if fam_clean == "" or fam_clean.lower() == "unknown":
            continue

        if not isinstance(sha256, str) or sha256.strip() == "":
            continue
        sha256 = sha256.strip()
        if sha256 in seen_hashes:
            continue

        report_path = reports_dir / f"{sha256}.json"
        if not report_path.exists():
            continue

        try:
            with report_path.open("r", encoding="utf-8") as f:
                report_json = json.load(f)
        except json.JSONDecodeError:
            # broken report
            continue

        api_seq = extract_api_sequence(report_json)
        if len(api_seq) < min_api_len:
            # too short / uninformative trace
            continue

        seen_hashes.add(sha256)
        api_text_full = " ".join(api_seq)
        rows.append(
            {
                "sha256": sha256,
                "family": fam_clean,
                "api_seq": api_seq,
                "num_calls": len(api_seq),
                "api_text_full": api_text_full,
            }
        )

    cape_df = pd.DataFrame(rows)
    # Shuffle to break any accidental ordering
    cape_df = cape_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return cape_df


def make_cape_text_splits(
    cape_df: pd.DataFrame,
    random_state: int = 42,
):
    """
    Convenience helper: 60/20/20 split on (api_text_full, family) with stratification.
    Returns:
        X_train_text, X_val_text, X_test_text, y_train, y_val, y_test
    """
    X = cape_df["api_text_full"].values
    y = cape_df["family"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test