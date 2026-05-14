import sys
from pathlib import Path


def main():
    """Launch the RAQA Streamlit app."""
    app_path = Path(__file__).parent / "_app.py"
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())
