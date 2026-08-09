import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import DOCS_DIR
from app.rag.pipeline import ingest_directory

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="recursive",
                    choices=["fixed","recursive","sentence","parent_child"])
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

print(ingest_directory(DOCS_DIR, args.strategy, args.reset))
