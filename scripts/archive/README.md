# Scripts archive

This directory keeps historical/legacy tooling out of the active `scripts/`
surface while preserving it in the repository for provenance.

| Subdirectory | Contents |
|---|---|
| `legacy_windows_tools/` | One-off tools that hard-coded a local Windows path or read V2/V3-era inputs. They are not part of the current V4 workflow. |
| `legacy_review_tools/` | Old blind-review and V4 candidate generation/revision helpers whose inputs have been superseded by `experiments/v4/augmentation_candidates/` and the current review packets. |
| `legacy_lab_scripts/` | Old AutoDL/lab SSH automation with hard-coded server paths. Use the env-driven scripts in `scripts/` for new lab work. |
| `legacy_backend_tools/` | Superseded Hu Tao/Genshin dataset, intent-cache and dual-3090 helpers removed from the active backend surface during finalization. |

Do not use these scripts for formal experiments. For current operations use
`scripts/README.md` and the V4 gate command documented there.

`INDEX.json` records every archived path, its former location, byte size and SHA-256. Archived code is provenance only: it is compiled for syntax during release checks but is not imported by the application.
