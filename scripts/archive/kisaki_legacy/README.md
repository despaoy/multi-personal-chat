# Archived Kisaki Execution Scripts

These scripts reproduce superseded prompt-v1, prompt-v2, E1/E2, and E2''
workflows. No active script or documentation entry point may call them.

The archive also contains superseded dataset builders, Gold v2 tools, direct
vLLM launchers, and AutoDL helpers. They may be inspected for provenance but
must not be imported or invoked by active code.

Use these active entry points instead:

- `scripts/validate_kisaki_v4_training_gate.py`
- `scripts/run_kisaki_experiment.py` (it remains blocked until the gate passes)
- `scripts/lab-run-kisaki-r2.sh`
- `scripts/lab-run-kisaki-r3.sh`
