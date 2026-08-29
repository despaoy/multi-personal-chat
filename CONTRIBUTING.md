# Contributing

Thank you for improving MultiPersonal Chat System. Changes should preserve the
project's evidence, compatibility and reproducibility guarantees.

## Development setup

Use Python 3.12, Node.js 22 and pnpm 10. Python source must remain importable on
Python 3.10 because the laboratory embedding environment still uses that
version; Ruff therefore treats Python 3.10 as the syntax compatibility floor.
Install backend development tools with:

```bash
python -m pip install -r backend/requirements-dev.txt
pnpm install --frozen-lockfile
```

Copy `.env.example` to the location documented for the selected deployment
mode. Never commit secrets, model weights, databases, generated indexes or
private experiment output.

## Naming conventions

### Python

- Packages and modules use short `snake_case` names.
- Classes and exceptions use `PascalCase`.
- Functions, methods, parameters and local variables use `snake_case`.
- Constants use `UPPER_SNAKE_CASE`.
- A leading underscore marks a non-public symbol; do not export internal
  builders or compatibility helpers from package `__init__.py` files.
- Test names describe observable behavior, for example
  `test_runtime_rejects_unknown_domain`.

### TypeScript and React

- Types, interfaces, enums and React components use `PascalCase`.
- Functions, properties and local variables use `camelCase`.
- Use complete, descriptive words; avoid unfamiliar abbreviations and
  Hungarian prefixes such as `IUser`.
- React component files may use `PascalCase.tsx`; route files keep the Next.js
  reserved lowercase names. Other TypeScript modules use lowercase
  kebab-case filenames.
- Export a symbol only when another module consumes it.

### Configuration and APIs

- Environment variables use `UPPER_SNAKE_CASE` and a feature prefix, such as
  `CHARACTER_RAG_INDEX_ROOT`.
- Python-internal fields use `snake_case`; existing JSON APIs use `camelCase`.
- Public API fields are not silently renamed. Add the replacement, deprecate
  the old field, document the migration and remove it only in a major version.
- Version numbers describe compatibility boundaries. Index-format versions are
  not product or service names.

## Formatting and quality checks

Python uses Ruff as the formatter, import sorter and linter. TypeScript uses
the strict compiler configuration and ESLint.

```bash
python -m ruff check backend/api/ask.py backend/api/generate.py \
  backend/api/knowledge.py backend/db/schemas.py \
  backend/knowledge/multiscale_rag \
  backend/knowledge/retrieval_core \
  backend/knowledge/grounded_answer \
  backend/scripts/build_character_rag_index.py
python -m ruff format --check backend/api/ask.py backend/api/generate.py \
  backend/api/knowledge.py backend/db/schemas.py \
  backend/knowledge/multiscale_rag \
  backend/knowledge/retrieval_core \
  backend/knowledge/grounded_answer \
  backend/scripts/build_character_rag_index.py
python -m pytest backend/tests -q
pnpm ts-check
pnpm lint
```

Use UTF-8, LF line endings and a final newline. `.editorconfig` contains the
editor-independent whitespace policy.

## Change requirements

- Keep each change focused and preserve unrelated working-tree edits.
- Add or update tests for behavior changes.
- Update README, architecture, operations and API documentation in the same
  change when their facts change.
- Keep historical experiment reports immutable; create a new report for a new
  condition.
- Record user-visible additions, changes, deprecations and removals in
  `CHANGELOG.md`.
- Run `git diff --check` before submitting.

## Reference standards

The repository conventions are adapted to this codebase from:

- [PEP 8](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Microsoft TypeScript Coding Guidelines](https://github.com/microsoft/TypeScript/wiki/Coding-guidelines)
- [FastAPI: Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [Semantic Versioning 2.0.0](https://semver.org/)

When an upstream guide conflicts with an established project convention,
prefer the documented repository convention and migrate deliberately instead
of combining styles within one module.
