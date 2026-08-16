# Security

## Supported scope

This is a single-machine research prototype. Security fixes are accepted for
the FastAPI backend, Next.js proxy, AstrBot gateway plugin, and the documented
Docker deployment path.

## Reporting

Do not report secrets, private keys, or real user data in a public issue.
Open an issue with a minimal reproduction and the affected component, or
contact the repository owner privately.

## Secrets policy

- Real `JWT_SECRET`, `ENCRYPTION_KEY`, API keys, database passwords, and SSH
  keys must never be committed.
- Use `.env.example` as the template and inject values from the deployment
  environment.
- If a secret is committed, rotate it immediately and remove it from Git
  history before disclosure.

## Important boundaries

- `vLLM` should not be exposed to the public network directly.
- LoRA root and symlink targets stay inside `LORA_PATH` / `LORA_ALLOWED_REAL_ROOTS`.
- The built-in integration token (`ASTRBOT_INTEGRATION_TOKEN`) is for the
  AstrBot gateway, not for public clients.
