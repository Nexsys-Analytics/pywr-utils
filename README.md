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

## License

MIT License
