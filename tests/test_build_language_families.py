from scripts import build_language_families as blf


def test_parse_newick_and_leaf_paths_report_ancestors_root_first():
    tree = blf.parse_newick("((eng:1,deu:1)germ:1,(fra:1,spa:1)roma:1)indo:1")
    paths = blf.leaf_paths(tree)
    assert paths["eng"] == ["indo", "germ", "eng"]
    assert paths["spa"] == ["indo", "roma", "spa"]


def test_family_name_is_the_root_for_ordinary_families():
    assert blf.family_name(["Turkic", "Oghuz", "Turkish"]) == "Turkic"


def test_family_name_descends_to_a_whitelisted_branch_of_indo_european():
    names = ["Indo-European", "Classical Indo-European", "Italic", "Romance", "French"]
    assert blf.family_name(names) == "Romance"
    assert blf.family_name(["Indo-European", "Classical Indo-European", "Balto-Slavic", "Eastern Baltic", "Latvian"]) == "Baltic"


def test_family_name_keeps_indo_european_when_no_branch_matches():
    assert blf.family_name(["Indo-European", "Anatolian", "Hittite"]) == "Indo-European"


ROWS = [
    {"ID": "indo", "Name": "Indo-European", "ISO639P3code": "", "Level": "family", "Language_ID": "", "Is_Isolate": ""},
    {"ID": "germ", "Name": "Germanic", "ISO639P3code": "", "Level": "family", "Language_ID": "", "Is_Isolate": ""},
    {"ID": "engl", "Name": "English", "ISO639P3code": "eng", "Level": "language", "Language_ID": "", "Is_Isolate": ""},
    {"ID": "scot", "Name": "Scots dialect", "ISO639P3code": "sco", "Level": "dialect", "Language_ID": "engl", "Is_Isolate": ""},
    {"ID": "basq", "Name": "Basque", "ISO639P3code": "eus", "Level": "language", "Language_ID": "", "Is_Isolate": "True"},
    {"ID": "unkn", "Name": "Unclassified", "ISO639P3code": "zzz", "Level": "language", "Language_ID": "", "Is_Isolate": ""},
    {"ID": "mand", "Name": "Mandarin", "ISO639P3code": "cmn", "Level": "language", "Language_ID": "", "Is_Isolate": ""},
    {"ID": "sino", "Name": "Sino-Tibetan", "ISO639P3code": "", "Level": "family", "Language_ID": "", "Is_Isolate": ""},
]
NEWICKS = {"indo": "((engl:1)germ:1)indo:1", "sino": "(mand:1)sino:1"}


def test_build_families_maps_iso_codes_and_resolves_dialects_to_their_language():
    families = blf.build_families(ROWS, NEWICKS)
    assert families["eng"] == "Germanic"
    assert families["sco"] == "Germanic"
    assert families["cmn"] == "Sino-Tibetan"


def test_build_families_labels_isolates_and_skips_unclassified_languages():
    families = blf.build_families(ROWS, NEWICKS)
    assert families["eus"] == "Language isolate"
    assert "zzz" not in families


def test_build_families_skips_bookkeeping_pseudo_family():
    rows = ROWS + [
        {"ID": "book", "Name": "Bookkeeping", "ISO639P3code": "", "Level": "family", "Language_ID": "", "Is_Isolate": ""},
        {"ID": "entr", "Name": "Some entry", "ISO639P3code": "qqq", "Level": "language", "Language_ID": "", "Is_Isolate": ""},
    ]
    assert "qqq" not in blf.build_families(rows, {**NEWICKS, "book": "(entr:1)book:1"})


def test_build_families_adds_macrolanguage_aliases_for_flores_codes():
    families = blf.build_families(ROWS, NEWICKS)
    assert families["zho"] == families["cmn"]
