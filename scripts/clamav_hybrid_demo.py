#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make src importable

from src.hybrid_scorer import HybridCAPEFamilyScorer


def run_clamscan(file_path: Path) -> str:
    """
    Run clamscan on a single file and return a simple verdict:
      - "FOUND" if malware detected
      - "CLEAN" otherwise
    """
    # Call clamscan --no-summary to keep output small
    result = subprocess.run(
        ["clamscan", "--no-summary", str(file_path)],
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    if "FOUND" in output:
        return "FOUND"
    elif "LibClamAV Error" in output or "ERROR:" in output:
        return "UNKNOWN"
    else:
        return "CLEAN"


def extract_api_text_from_file(file_path: Path) -> str:
    """
    Design level placeholder so read the file as text and treat it as api_text.
    In a real system this would extract CAPE-like API tokens from a report
    or from dynamic logs. For this prototype assume the file already
    contains a space-separated token sequence
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    # Very small normalisation: collapse whitespace
    text = " ".join(text.split())
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Tiny ClamAV + hybrid scorer demo (design-level)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File(s) or directory(ies) to scan. "
             "If a directory is given, all regular files inside it are scanned.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.8,
        help="Hybrid risk threshold on family confidence (default: 0.8).",
    )
    args = parser.parse_args()

    scorer = HybridCAPEFamilyScorer(alpha=0.0, static_conf_thresh=0.9)

    # Collect file list
    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file():
                    files.append(child)
        elif path.is_file():
            files.append(path)

    for file_path in files:
     clam_verdict = run_clamscan(file_path)

     hybrid_info = None

     # Only run hybrid scorer if ClamAV did NOT already find malware
     if clam_verdict != "FOUND":
         api_text = extract_api_text_from_file(file_path)
         if api_text:
             family, conf, proba_dict = scorer.score_text(api_text)
             hybrid_info = {
                 "predicted_family": family,
                 "confidence": conf,
                 "risk_label": (
                     "SUSPICIOUS_HYBRID"
                     if conf >= args.risk_threshold
                     else "LIKELY_CLEAN"
                 ),
             }

     record = {
         "file": str(file_path),
         "clamav_verdict": clam_verdict,
         "hybrid_used": hybrid_info is not None,
         "hybrid": hybrid_info,
     }

     print(json.dumps(record))


if __name__ == "__main__":
    main()