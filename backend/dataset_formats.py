"""Turns an uploaded file's text into the dataset dict that
server.validate_and_normalize expects ({"points": [...], ...}).

  json   the full format (also the only one that can carry tokenVectors)
  jsonl  one point object per line; leading objects without a "vector"
         are merged as metadata (model, layer, pooling, lang_families, ...)
  csv    header row; columns id, lang (required), text, pairId/pair_id,
         family (optional); the vector is either one "vector" column
         ("[1, 2]" or space/semicolon separated) or every remaining column,
         in order. Comma- or tab-delimited.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import List

_PAIR_ID_NAMES = {"pairid", "pair_id"}
_KNOWN_COLUMNS = {"id", "lang", "text", "family", "vector"} | _PAIR_ID_NAMES


def parse_dataset_text(text: str, fmt: str) -> dict:
    if fmt not in ("json", "jsonl", "csv"):
        raise ValueError(f'Unknown format "{fmt}" (expected json, jsonl or csv)')
    if not text.strip():
        raise ValueError("The file is empty")
    return {"json": _parse_json, "jsonl": _parse_jsonl, "csv": _parse_csv}[fmt](text)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def _parse_jsonl(text: str) -> dict:
    meta: dict = {}
    points: List[dict] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {number}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Line {number} must be a JSON object")
        if "vector" in obj or points:
            points.append(obj)
        else:
            meta.update(obj)
    return {**meta, "points": points}


def _parse_vector_cell(cell: str, row: int) -> List[float]:
    cell = cell.strip()
    try:
        if cell.startswith("["):
            values = json.loads(cell)
        else:
            values = [float(v) for v in re.split(r"[\s;]+", cell) if v]
        return [float(v) for v in values]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Row {row}: the vector isn't a list of numbers") from e


def _parse_csv(text: str) -> dict:
    header_line = text.lstrip("\ufeff").splitlines()[0]
    delimiter = "\t" if header_line.count("\t") > header_line.count(",") else ","
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")), delimiter=delimiter)
    header = [h.strip() for h in next(reader)]
    names = [h.lower() for h in header]

    if "id" not in names or "lang" not in names:
        raise ValueError('The CSV needs "id" and "lang" columns')
    vector_column = names.index("vector") if "vector" in names else None
    component_columns = [i for i, n in enumerate(names) if n not in _KNOWN_COLUMNS]
    if vector_column is None and not component_columns:
        raise ValueError('The CSV needs a "vector" column or one numeric column per dimension')

    points = []
    for row_number, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise ValueError(f"Row {row_number} has {len(row)} columns, expected {len(header)}")
        by_name = dict(zip(names, row))
        point = {"id": by_name["id"].strip(), "lang": by_name["lang"].strip()}
        if by_name.get("text", "").strip():
            point["text"] = by_name["text"]
        if by_name.get("family", "").strip():
            point["family"] = by_name["family"].strip()
        for name in _PAIR_ID_NAMES:
            if by_name.get(name, "").strip():
                try:
                    point["pairId"] = int(by_name[name])
                except ValueError as e:
                    raise ValueError(f"Row {row_number}: pairId must be an integer") from e
        if vector_column is not None:
            point["vector"] = _parse_vector_cell(row[vector_column], row_number)
        else:
            try:
                point["vector"] = [float(row[i]) for i in component_columns]
            except ValueError as e:
                raise ValueError(f"Row {row_number}: vector columns must be numbers") from e
        points.append(point)
    return {"points": points}
