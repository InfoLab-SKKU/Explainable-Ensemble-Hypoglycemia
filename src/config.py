"""Paths and global settings shared by all experiment scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ENCODED = DATA / 'encoded_features'
RESULTS = ROOT / 'results'
CACHE = ROOT / 'cache'          # per-split features selected by the neural selector (re-created if missing)
SEEDS = list(range(10))         # 10 stratified 70/30 splits (random seeds 0-9)
for d in (RESULTS, CACHE):
    d.mkdir(exist_ok=True)
