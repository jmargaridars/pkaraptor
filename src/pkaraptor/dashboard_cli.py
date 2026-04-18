import argparse
from pathlib import Path

import pandas as pd

from .dashboard import build_dashboard_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive protonation dashboard html from a pdb and csv."
    )
    parser.add_argument("--pdb", required=True, help="input pdb file")
    parser.add_argument(
        "--csv",
        default="protonation_environment_analysis.csv",
        help="csv file produced by `pkaraptor`",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH used for the analysis (for display in the title)",
    )
    parser.add_argument(
        "--out",
        default="protonation_dashboard.html",
        help="output html file",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    html_doc = build_dashboard_html(df, args.pdb, args.ph)
    out_path = Path(args.out)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Written HTML dashboard: {out_path}")