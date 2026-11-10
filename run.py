"""Start the dashboard from anywhere:  python run.py
(runs from this folder so .streamlit/config.toml - the light theme - is applied)."""
import os
import sys
from pathlib import Path

from streamlit.web import cli

os.chdir(Path(__file__).resolve().parent)
sys.argv = ["streamlit", "run", "dashboard.py", *sys.argv[1:]]
sys.exit(cli.main())
