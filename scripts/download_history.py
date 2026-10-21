"""Download/refresh 1-minute candle history for all traded and reference symbols.

Usage: python -m scripts.download_history [--days 30]
"""
import argparse
import logging

from src.config import ROOT, settings
from src.market_data import update_history
from src.wallex_client import WallexClient

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--universe", action="store_true", help="use data/universe.txt (research universe)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = WallexClient()
    syms = list(settings.symbols + settings.reference_symbols)
    if args.universe:
        syms += [s for s in (ROOT / "data" / "universe.txt").read_text().split() if s not in syms]
    for sym in syms:
        df = update_history(client, sym, days=args.days)
        span = f"{df.index[0]} -> {df.index[-1]}" if len(df) else "empty"
        print(f"{sym:10s} {len(df):7d} bars  {span}")
