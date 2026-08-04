# Changelog

## [0.2.0] - 2026-08-04

### Added

- Unified `pkaraptor run` command for residue analysis and dashboard generation.
- `pkaraptor finalize` command for applying user decisions, adding hydrogens, validating the output, and generating a preparation report.
- Structured project output containing inputs, analysis results, dashboard, decisions, structures, and reports.
- Chain-aware membrane annotation through repeatable `--opm-embedded` arguments.
- Support for a separate PPM/OPM-oriented structure in the dashboard.
- Interactive 2D residue and contact representations.
- Searchable and filterable residue table.
- Colour-blind-friendly titration curves using colour, line style, and marker shape.
- JSON and HTML preparation reports containing assignment, charge, and residue-mapping summaries.

### Changed

- Residue states are controlled explicitly by the user.
- Unchosen non-histidine residues preserve their canonical input state.
- Histidines require an explicit `HID`, `HIE`, or `HIP` selection.
- Detected disulfide cysteines are preserved as `CYX`.
- Membrane embedding and SASA exposure are reported separately.
- Project metadata uses relative paths so generated projects can be moved or shared.

### Fixed

- Added command routing for `pkaraptor run` and `pkaraptor finalize`.
- Ensured that dashboards use the PPM/OPM-oriented structure when `--opm-pdb` is provided.
- Corrected residue-table searching and repeated 2D residue selection.
- Prevented unsupported protonation states from being passed silently to OpenMM.
- Added a clear diagnostic for protein residues covalently linked to unsupported heterogens or glycans.
- Preserved canonical residue states during export instead of assigning unsupported deprotonated fallbacks.
