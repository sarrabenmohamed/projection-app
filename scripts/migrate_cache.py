"""Converts the old per-language JSON caches (data/flores_lang_*.json) to the
folder format of backend/embedding_store.py.

Each conversion is read back and compared number by number; a folder that
doesn't match is removed and reported. The JSON files are only deleted with
--delete, and only after their own conversion checked out.

Usage:
    python scripts/migrate_cache.py [--data-dir data] [--delete]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.embedding_store import EmbeddingStore  # noqa: E402


def _size(path: Path) -> int:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--delete", action="store_true", help="remove each JSON file once its conversion is verified")
    args = parser.parse_args()

    store = EmbeddingStore(args.data_dir)
    before = after = failed = deleted = 0
    for legacy in sorted(args.data_dir.glob("flores_lang_*.json")):
        folder = legacy.with_suffix("")
        try:
            if folder.exists():
                if not args.delete:
                    print(f"skip     {legacy.name} (already converted)")
                    continue
                if not store.covers_json(legacy):
                    failed += 1
                    print(f"KEPT     {legacy.name}: its folder doesn't hold the same numbers")
                    continue
                print(f"deleted  {legacy.name} (folder verified identical)")
                deleted += _size(legacy)
                legacy.unlink()
                continue
            store.import_json(legacy)
        except (OSError, ValueError, KeyError, TypeError) as e:
            failed += 1
            print(f"FAILED   {legacy.name}: {e}")
            continue
        before += _size(legacy)
        after += _size(folder)
        print(f"ok       {legacy.name}: {_size(legacy) / 1e6:.0f} MB -> {_size(folder) / 1e6:.0f} MB")
        if args.delete:
            legacy.unlink()

    print(f"\nconverted {before / 1e6:.0f} MB of JSON to {after / 1e6:.0f} MB; "
          f"freed {deleted / 1e6:.0f} MB of already-converted JSON; {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
