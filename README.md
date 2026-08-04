[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20111134.svg)](https://doi.org/10.5281/zenodo.20111134)

<p align="center">
  <img src="https://raw.githubusercontent.com/jmargaridars/pkaraptor/main/src/assets/pkaraptor-logo.png" alt="pKaRaptor logo" width="500">
</p>

# pKaRaptor

pKaRaptor is a Python toolkit for integrating protein pKa predictions with residue-level structural environments. It produces an interactive HTML dashboard in which protonation and tautomer assignments remain explicit user decisions.

## Features

- Integrates local PROPKA results with optional PyPka and DeepKa predictions.
- Calculates residue-level structural and solvent-exposure descriptors.
- Supports chain-aware membrane annotations from user-supplied OPM/PPM information.
- Displays searchable residue data, titration curves, three-dimensional structure context, and two-dimensional interaction maps.
- Exports explicit residue assignments as JSON.
- Applies compatible assignments, adds hydrogens with OpenMM, and preserves detected disulfides.
- Generates JSON and HTML preparation reports.

## Installation

Python 3.10 or later is required.

```bash
pip install pkaraptor
```

For an editable source installation:

```bash
git clone https://github.com/jmargaridars/pkaraptor.git
cd pkaraptor
python -m pip install -e .
```

Confirm the installation with:

```bash
pkaraptor --version
pkaraptor --help
```

## Recommended workflow

Run the analysis and create a structured project:

```bash
pkaraptor run protein.pdb --ph 7.0
```

The project contains separate `input`, `analysis`, `dashboard`, `decisions`, `structures`, and `report` directories.

External predictions and an oriented membrane structure can be supplied in the same command:

```bash
pkaraptor run protein.pdb \
  --ph 7.0 \
  --pypka-csv pypka.csv \
  --deepka-csv deepka.csv \
  --opm-pdb protein-ppm.pdb \
  --opm-id PPM3 \
  --opm-embedded "A:48-66,69,426,428-458" \
  --opm-embedded "B:48-66,69,426,428-458,461" \
  --no-open
```

`--opm-embedded` is repeatable. The original PDB is retained for final preparation, while the oriented PDB is used for environment analysis and dashboard visualization.

Open the generated dashboard, review the evidence, select histidine tautomers and any other intended overrides, and export the decisions JSON. Finalize the project with:

```bash
pkaraptor finalize decisions.json --project protein_pkaraptor
```

If a protein residue is covalently linked to a heterogen or glycan that is unsupported by the selected OpenMM force field, pKaRaptor stops with an explanatory error. A protein-only output can be requested explicitly:

```bash
pkaraptor finalize decisions.json \
  --project protein_pkaraptor \
  --remove-heterogens
```

This option removes heterogens, including glycans, ligands, and ions. Water is also removed unless `--keep-water` is supplied.

## Assignment policy

- pKaRaptor does not recommend or automatically infer residue assignments.
- Unchosen non-histidine residues preserve their canonical input state.
- Histidines require an explicit `HID`, `HIE`, or `HIP` selection.
- Detected disulfide cysteines are preserved as `CYX`.
- Explicit states that are incompatible with the selected hydrogenation route produce an error instead of being silently replaced.

The two-dimensional interaction map and the three-dimensional viewer present structural evidence. Changing a selected output state does not recalculate the input structure or its measured contacts.

## Individual commands

The recommended interface is `pkaraptor run` followed by `pkaraptor finalize`. Individual tools remain available:

```text
pkaraptor-analysis
pkaraptor-dashboard
pkaraptor-apply-json
pkaraptor-protonate
pkaraptor-run
pkaraptor-finalize
```

The original analysis interface is retained for compatibility:

```bash
pkaraptor --pdb protein.pdb --ph 7.0 --out residues.csv
```

Use `--help` with any command to inspect its options.

## Outputs

Depending on the selected workflow, pKaRaptor can generate:

- residue-level CSV analysis;
- an interactive, self-contained HTML dashboard;
- JSON residue decisions;
- a hydrogenated PDB structure;
- a histidine hydrogen report;
- JSON and HTML preparation reports.

The preparation report summarizes assignments, estimated side-chain charge, and residue-mapping validation. Charge estimates exclude termini, ligands, ions, and cofactors.

## Citation

Use the citation metadata in `CITATION.cff`. The concept DOI for all pKaRaptor versions is [10.5281/zenodo.20111134](https://doi.org/10.5281/zenodo.20111134).

## License

pKaRaptor is distributed under the BSD 3-Clause License.
