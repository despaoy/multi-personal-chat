# Legacy backend tools

These scripts were moved from `backend/scripts/` during the 2026-08-28 finalization audit because no active entrypoint or test referenced them and their assumptions no longer match the V4/Kisaki system.

| Script | Historical scope |
| --- | --- |
| `augment_patterns.py` | Template expansion for the old Genshin intent cache |
| `convert_dataset.py` | Hu Tao dialogue conversion |
| `dataset_manager.py` | Pre-canonical dataset copy/index helper |
| `find_boundary_cases.py` | Old Genshin boundary scan with a personal Windows path |
| `inject_labels.py` | Manual labels for the old intent cache |
| `local_infer_7b.py` | Hu Tao 4-bit local LoRA inference |
| `deploy_3090.py` | Superseded dual-3090 deployment assumptions |

Do not use these files for current deployment, canonical data changes or formal experiments. Current service tools are indexed in [`../../../backend/scripts/README.md`](../../../backend/scripts/README.md).
