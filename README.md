# RAQA

**R**etrieval-**A**ugmented **Q**uestion-**A**nswering

Pip-installable, Streamlit-based Q&A over your documents, powered by the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

## Quick start

```bash
pip install raqa
raqa
```

Open the URL shown in your terminal (default `http://localhost:8501`). A sample dataset is bundled — you can start asking questions immediately.

## Data

### Custom documents

Place `.md`, `.txt`, or `.csv` files in a `data/` folder in the working directory:

```
data/
├── my_docs.md
├── notes.txt
└── spreadsheet.csv
```

Override the directory with the `RAQA_DATA_DIR` environment variable:

```bash
RAQA_DATA_DIR=/path/to/docs raqa
```

### Bundle sample data for publishing

The `data/` folder at the repo root is your source of truth. When publishing:

```bash
make publish
```

The `build` target copies `data/*` into the package before creating the wheel, so sample data ships with the installed package. If no custom `data/` is found at runtime, the bundled sample data is used as fallback.

## Configuration

### API key

Create a `.env` file in the working directory:

```
OPENAI_API_KEY=sk-...
```

RAQA reads it automatically via `python-dotenv`. Alternatively, set the `OPENAI_API_KEY` environment variable directly.

## Development

```bash
git clone https://github.com/JordiCarreraVentura/raqa
cd raqa

# Create a branch-specific virtual environment
make env

# Launch the app
make run
```

### Makefile targets

| Target      | Description                                     |
|-------------|-------------------------------------------------|
| `env`       | Create virtualenv for current branch            |
| `install`   | Pip-install the package in editable mode        |
| `run`       | Start the Streamlit app                         |
| `build`     | Sync `data/` into package and build wheel+tarball |
| `publish`   | Build and upload to PyPI                        |
| `clean`     | Remove cache and build artifacts                |

## Deployment

RAQA is a standard Streamlit app and can be deployed on any platform that runs Python web apps.

### RunPod

```bash
pip install raqa
streamlit run $(python -c "import raqa; print(raqa.__file__)")._app.py \
  --server.address 0.0.0.0 --server.port 8501
```

Open port 8501 in the firewall. See [RunPod docs](https://docs.runpod.io/) for serverless or pod deployment.

### DigitalOcean Droplet

```bash
pip install raqa
raqa
```

Then expose the app:

```bash
sudo ufw allow 8501/tcp
```

Point a domain or IP to `http://<droplet-ip>:8501`. For production, pair with Nginx as a reverse proxy for HTTPS.

## Project structure

```
src/raqa/
  __init__.py     # Package marker
  __main__.py     # Enables python -m raqa
  _app.py         # Streamlit chat UI
  _agent.py       # OpenAI agent + search tool
  _data.py        # Document loader (.md, .txt, .csv)
  _sample/        # Bundled sample data (synced from data/ at build time)
  cli.py          # CLI entry point
```

## Related

- [PyRAG](https://pypi.org/project/PyRAG) — focuses on SingleStore.
- [ragger-simple](https://pypi.org/project/ragger-simple) — requires Qdrant + API key. RAQA only needs an LLM API key.
