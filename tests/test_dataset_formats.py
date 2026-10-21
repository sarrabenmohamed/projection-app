import json

import pytest

from dataset_formats import parse_dataset_text


def test_json_passthrough():
    raw = {"points": [{"id": "a", "lang": "en", "vector": [1, 2]}]}
    assert parse_dataset_text(json.dumps(raw), "json") == raw


def test_json_invalid_reports_error():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_dataset_text("{nope", "json")


def test_jsonl_one_point_per_line_skipping_blank_lines():
    text = "\n".join([
        json.dumps({"id": "a", "lang": "en", "vector": [1, 2], "text": "hi"}),
        "",
        json.dumps({"id": "b", "lang": "de", "vector": [3, 4]}),
    ])
    raw = parse_dataset_text(text, "jsonl")
    assert [p["id"] for p in raw["points"]] == ["a", "b"]
    assert raw["points"][0]["text"] == "hi"


def test_jsonl_leading_metadata_line_without_vector():
    text = "\n".join([
        json.dumps({"model": "m", "layer": "last", "lang_families": {"en": "Germanic"}}),
        json.dumps({"id": "a", "lang": "en", "vector": [1, 2]}),
    ])
    raw = parse_dataset_text(text, "jsonl")
    assert raw["model"] == "m"
    assert raw["lang_families"] == {"en": "Germanic"}
    assert len(raw["points"]) == 1


def test_jsonl_bad_line_reports_line_number():
    text = json.dumps({"id": "a", "lang": "en", "vector": [1]}) + "\n{bad"
    with pytest.raises(ValueError, match="line 2"):
        parse_dataset_text(text, "jsonl")


def test_jsonl_non_object_line_rejected():
    with pytest.raises(ValueError, match="(?i)line 1"):
        parse_dataset_text("[1, 2]", "jsonl")


def test_csv_with_numeric_columns_in_order():
    text = "id,lang,text,pairId,v0,v1\na,en,hello,0,0.5,1.5\nb,de,hallo,0,2,3\n"
    raw = parse_dataset_text(text, "csv")
    assert raw["points"][0] == {"id": "a", "lang": "en", "text": "hello", "pairId": 0, "vector": [0.5, 1.5]}
    assert raw["points"][1]["vector"] == [2.0, 3.0]


def test_csv_with_single_vector_column_json_or_space_separated():
    text = 'id,lang,vector\na,en,"[1, 2, 3]"\nb,de,4 5 6\n'
    raw = parse_dataset_text(text, "csv")
    assert [p["vector"] for p in raw["points"]] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_csv_optional_family_and_snake_case_pair_id():
    text = "id,lang,pair_id,family,v0\na,en,3,Germanic,1\n"
    p = parse_dataset_text(text, "csv")["points"][0]
    assert p["pairId"] == 3 and p["family"] == "Germanic"


def test_csv_tsv_delimiter_detected():
    text = "id\tlang\tv0\tv1\na\ten\t1\t2\n"
    assert parse_dataset_text(text, "csv")["points"][0]["vector"] == [1.0, 2.0]


def test_csv_blank_pair_id_and_text_are_omitted():
    text = "id,lang,text,pairId,v0\na,en,,,1\n"
    p = parse_dataset_text(text, "csv")["points"][0]
    assert "pairId" not in p and "text" not in p


def test_csv_missing_required_columns():
    with pytest.raises(ValueError, match="id.*lang"):
        parse_dataset_text("name,v0\na,1\n", "csv")


def test_csv_without_vector_columns():
    with pytest.raises(ValueError, match="vector"):
        parse_dataset_text("id,lang\na,en\n", "csv")


def test_csv_non_numeric_value_reports_row():
    with pytest.raises(ValueError, match="(?i)row 3"):
        parse_dataset_text("id,lang,v0\na,en,1\nb,en,x\n", "csv")


def test_csv_ragged_row_reports_row():
    with pytest.raises(ValueError, match="(?i)row 2"):
        parse_dataset_text("id,lang,v0,v1\na,en,1\n", "csv")


def test_unknown_format_rejected():
    with pytest.raises(ValueError, match="format"):
        parse_dataset_text("x", "xml")


def test_empty_file_rejected():
    for fmt in ("json", "jsonl", "csv"):
        with pytest.raises(ValueError):
            parse_dataset_text("   \n", fmt)
