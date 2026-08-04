from __future__ import annotations

import sys

from . import __version__
from . import analysis_cli
from .workflow import finalize_main, run_main


def _print_help() -> None:
    print(
        """pKaRaptor - protein pKa and structural-environment analysis

Recommended workflow:
  pkaraptor run PDB [options]
      Analyse a structure and create an organized project with an HTML dashboard.

  pkaraptor finalize DECISIONS.json --project PROJECT [options]
      Apply explicit residue-state decisions, add hydrogens, validate, and report.

Legacy analysis command (retained for compatibility):
  pkaraptor --pdb PDB [options]
      Write the residue-level protonation and environment CSV directly.

Additional commands:
  pkaraptor-analysis, pkaraptor-dashboard, pkaraptor-apply-json,
  pkaraptor-protonate, pkaraptor-run, pkaraptor-finalize

Use `pkaraptor run --help`, `pkaraptor finalize --help`, or
`pkaraptor-analysis --help` for detailed options.
"""
    )


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return
    if sys.argv[1] == "--version":
        print(f"pKaRaptor {__version__}")
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run_main(sys.argv[2:])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "finalize":
        finalize_main(sys.argv[2:])
        return
    analysis_cli.main()
