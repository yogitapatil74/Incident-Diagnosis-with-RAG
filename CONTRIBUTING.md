# Contributing

Thanks for your interest in improving this project. Correctness and clarity matter more
than raw feature count — small, well-tested changes are very welcome.

## Getting started

```bash
git clone https://github.com/yogitapatil74/Incident-Diagnosis-with-RAG.git
cd Incident-Diagnosis-with-RAG

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install ruff  # linter used in CI

cp config/config.example.yaml config/config.yaml
cp .env.example .env
```

See [RUNNING_THE_POC.md](RUNNING_THE_POC.md) for the full walkthrough (indexing the
sample knowledge base, running a diagnosis, starting the API, running evaluation).

## Before you open a PR

```bash
ruff check src scripts tests   # lint
pytest tests/ -v                # tests
```

Both run in CI on every PR; please make sure they pass locally first.

## Making changes

- Keep pull requests focused — one logical change per PR is easier to review than a
  bundle of unrelated fixes.
- Add or update tests for behavior you change. `tests/` mocks all external LLM/embedding
  calls, so tests should run offline and deterministically.
- Match the existing style: type hints, `structlog` for logging, dataclasses for
  structured data. `ruff check` enforces import ordering and catches unused imports.
- If you change the chunking, retrieval, or diagnosis pipeline, please explain the
  reasoning in the PR description — these are core to the system's behavior, so changes
  are worth documenting clearly.

## Reporting bugs / requesting features

Please use the issue templates under `.github/ISSUE_TEMPLATE/`. For bugs, a minimal
reproduction (sample document/incident + expected vs. actual output) is the single most
useful thing you can include.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating,
you're expected to uphold it.
