"""Builds resources/language_families.json (ISO 639-3 code -> language family)
from the Glottolog CLDF dataset (https://github.com/glottolog/glottolog-cldf,
CC BY 4.0), plus aliases (ISO 639-1 codes and English names) so inputs such
as "en" or "German" resolve too. The server only reads the generated JSON,
so there is no runtime dependency on the sources; rerun this to refresh it:

    python3 scripts/build_language_families.py                # downloads the data
    python3 scripts/build_language_families.py --source DIR   # DIR has languages.csv, classification.nex, iso-639-3.tab

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

ISO_TABLE_URL = "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab"
BASE_URL = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "resources" / "language_families.json"

# Glottolog's top-level Indo-European is too coarse to color by, and its
# branches sit several levels down, so for families listed here the first node
# on the path that appears in the whitelist names the family instead.
SPLIT_BRANCHES = {
    "Indo-European": {
        "Germanic": "Germanic",
        "Romance": "Romance",
        "Slavic": "Slavic",
        "Eastern Baltic": "Baltic",
        "Indo-Iranian": "Indo-Iranian",
        "Celtic": "Celtic",
        "Greek": "Greek",
        "Albanian": "Albanian",
        "Armenic": "Armenian",
    },
}

# Glottolog classifies the individual languages; FLORES-200 sometimes uses the
# ISO 639-3 macrolanguage code that covers them (e.g. zho_Hans for Mandarin).
MACROLANGUAGE_ALIASES = {
    "zho": "cmn", "ara": "arb", "msa": "zsm", "fas": "pes", "mon": "khk", "uzb": "uzn",
    "aze": "azj", "lav": "lvs", "orm": "gaz", "yid": "ydd", "kur": "kmr", "nep": "npi",
    "swa": "swh", "est": "ekk", "sqi": "als", "grn": "gug", "que": "quy", "pus": "pbt",
    "kon": "kng", "srd": "src",
}

# Glottolog files its non-language bookkeeping entries under pseudo-families.
EXCLUDED_FAMILIES = {"Bookkeeping"}

Tree = Tuple[str, list]


def parse_newick(text: str) -> Tree:
    """Parses a Newick string into (label, children); branch lengths are dropped."""
    pos = 0
    label_re = re.compile(r"([^:,()]*)(?::[^,()]*)?")

    def node() -> Tree:
        nonlocal pos
        children = []
        if text[pos] == "(":
            pos += 1
            while True:
                children.append(node())
                if text[pos] == ",":
                    pos += 1
                    continue
                pos += 1  # ")"
                break
        match = label_re.match(text, pos)
        pos = match.end()
        return match.group(1), children

    return node()


def leaf_paths(tree: Tree) -> Dict[str, List[str]]:
    """Maps every leaf label to the labels from the root down to (and including) it."""
    paths: Dict[str, List[str]] = {}

    def walk(node: Tree, prefix: List[str]) -> None:
        label, children = node
        path = prefix + [label]
        if not children:
            paths[label] = path
        for child in children:
            walk(child, path)

    walk(tree, [])
    return paths


def family_name(names_root_first: List[str]) -> str:
    root = names_root_first[0]
    branches = SPLIT_BRANCHES.get(root)
    if branches:
        for name in names_root_first:
            if name in branches:
                return branches[name]
    return root


def build_families(rows: List[dict], newick_by_family: Dict[str, str]) -> Dict[str, str]:
    by_id = {r["ID"]: r for r in rows}
    paths: Dict[str, List[str]] = {}
    for newick in newick_by_family.values():
        paths.update(leaf_paths(parse_newick(newick)))

    families: Dict[str, str] = {}
    for row in rows:
        iso = row["ISO639P3code"]
        if not iso:
            continue
        glottocode = row["Language_ID"] or row["ID"]  # a dialect takes its language's family
        path = paths.get(glottocode)
        if path:
            name = family_name([by_id[g]["Name"] for g in path if g in by_id])
            if name not in EXCLUDED_FAMILIES:
                families.setdefault(iso, name)
        elif row["Is_Isolate"].strip().lower() == "true":
            families.setdefault(iso, "Language isolate")

    for macro, individual in MACROLANGUAGE_ALIASES.items():
        if macro not in families and individual in families:
            families[macro] = families[individual]
    return families


def normalize_alias(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_aliases(iso_rows: List[dict], glottolog_rows: List[dict], families: Dict[str, str]) -> Dict[str, str]:
    """Maps ISO 639-1 codes and English language names to ISO 639-3 codes
    present in `families`. SIL names win over Glottolog names, and the first
    claimant of a name keeps it, so lookups never silently swap languages."""
    aliases: Dict[str, str] = {}

    def add(alias: str, iso: str) -> None:
        key = normalize_alias(alias)
        if key and iso in families and key not in families:
            aliases.setdefault(key, iso)

    for row in iso_rows:
        if row["Part1"]:
            add(row["Part1"], row["Id"])
    for row in iso_rows:
        if row["Scope"] in ("I", "M") and row["Language_Type"] == "L":
            add(row["Ref_Name"], row["Id"])
    for row in glottolog_rows:
        if row["ISO639P3code"]:
            add(row["Name"], row["ISO639P3code"])
    return aliases


def _read(source: str | None, name: str, url: str | None = None) -> str:
    if source:
        return (Path(source) / name).read_text(encoding="utf-8")
    with urllib.request.urlopen(url or f"{BASE_URL}/{name}") as response:
        return response.read().decode("utf-8")


def _newicks(nexus: str) -> Dict[str, str]:
    trees = {}
    for match in re.finditer(r"^\s*tree (\w+) = \[&R\] (.*);\s*$", nexus, flags=re.MULTILINE):
        trees[match.group(1)] = match.group(2)
    return trees


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", help="directory with languages.csv and classification.nex (default: download)")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    rows = list(csv.DictReader(_read(args.source, "languages.csv").splitlines()))
    families = build_families(rows, _newicks(_read(args.source, "classification.nex")))

    iso_rows = list(csv.DictReader(_read(args.source, "iso-639-3.tab", ISO_TABLE_URL).splitlines(), delimiter="\t"))
    aliases = build_aliases(iso_rows, rows, families)

    output = {
        "source": "Glottolog CLDF (https://github.com/glottolog/glottolog-cldf), CC BY 4.0",
        "generated": date.today().isoformat(),
        "families": dict(sorted(families.items())),
        "aliases": dict(sorted(aliases.items())),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(families)} languages, {len(aliases)} aliases to {args.output}")


if __name__ == "__main__":
    main()
