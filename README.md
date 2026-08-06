# PYWR Utils

A command-line utility package for PYWR water resource modeling.

## Installation

```bash
pip install -e .
```

## Usage

### Create Synthetic Model

Create a synthetic PYWR model with specified zones and transfers:

```bash
pywr-utils create-synthetic-model --zones 5 --transfers 10
```

#### Options

- `--zones N`: Number of zones to create in the synthetic model (required)
- `--transfers M`: Number of transfers to create in the synthetic model (required)

## Development

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Run the CLI: `pywr-utils --help`
4. One-time, so the pre-push gitleaks hook actually runs: `pip install pre-commit && pre-commit install` (needs the `gitleaks` binary on PATH — `brew install gitleaks` or see the install step in `.github/workflows/gitleaks.yml`). CI runs the same scan on every push/PR regardless, this just catches a leak before it leaves your machine.

## License

MIT License
