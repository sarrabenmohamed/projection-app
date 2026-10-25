"""Editing a loaded FLORES-200 dataset in the browser: new languages and
sentences are embedded and placed into the existing layout, so what is
already plotted (points, selection, colours) stays put. Runs against the
real backend with a fake embedder (see fake_server.py)."""
import json
import urllib.request

import pytest

pytestmark = pytest.mark.e2e

BASE = ["eng_Latn", "fra_Latn", "deu_Latn"]
SENTENCES = 12


@pytest.fixture(scope="module")
def server_url(fake_server_url):
    return fake_server_url


def _call(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="POST" if body is not None else "GET")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


@pytest.fixture
def editor(ui, server_url):
    """The app showing a fresh BASE x SENTENCES dataset, with the FLORES-200 form open."""
    _call(f"{server_url}/__reset", {})
    _call(f"{server_url}/dataset/flores", {"lang_codes": BASE, "n_sentences": SENTENCES})
    ui.page.set_default_timeout(60_000)
    ui.page.reload()
    ui.page.wait_for_function(f"window.__app && __app.view.pointLayer.points.length === {len(BASE) * SENTENCES}")
    ui.settle()
    ui.page.locator("#flores-controls summary").click()
    ui.server_url = server_url
    return ui


def positions(ui):
    return {p["id"]: (p["x"], p["y"]) for p in ui.js("__app.points.map(p => ({id: p.id, x: p.x, y: p.y}))")}


def update_to(ui, langs, sentences=None):
    ui.page.fill("#flores-langs", ", ".join(langs))
    if sentences is not None:
        ui.page.fill("#flores-n", str(sentences))
    ui.page.click("#flores-load-button")


def embedder_calls(ui):
    return _call(f"{ui.server_url}/__calls")


def test_the_loaded_dataset_opens_in_edit_mode_with_its_settings_locked(editor):
    page = editor.page
    assert page.input_value("#flores-langs") == ", ".join(BASE)
    assert page.input_value("#flores-n") == str(SENTENCES)
    assert page.locator("#flores-mode button[data-mode=edit]").get_attribute("aria-pressed") == "true"
    assert page.is_disabled("#flores-model") and page.is_disabled("#flores-pooling")
    assert page.inner_text("#flores-load-button") == "Update dataset"
    assert page.is_disabled("#flores-load-button")  # nothing to update yet
    assert page.inner_text("#flores-diff") == "No changes yet."

    page.fill("#flores-langs", ", ".join(BASE + ["spa_Latn"]))
    assert page.inner_text("#flores-diff") == "+ spa_Latn"
    assert page.is_enabled("#flores-load-button")
    page.fill("#flores-n", "20")
    assert page.inner_text("#flores-diff") == "+ spa_Latn · sentences 12 → 20"

    page.click("#flores-mode button[data-mode=load]")
    assert page.is_enabled("#flores-model") and page.is_enabled("#flores-pooling")
    assert page.inner_text("#flores-load-button") == "Load FLORES-200"
    assert page.is_hidden("#flores-diff")


def test_adding_a_language_places_it_without_moving_anything(editor):
    before = positions(editor)
    editor.page.mouse.click(*editor.screen_of("eng_Latn-0"))
    assert editor.selection == ["eng_Latn-0"]

    update_to(editor, BASE + ["spa_Latn"])
    editor.wait_for_status("48 points")
    editor.settle()

    after = positions(editor)
    assert {i: after[i] for i in before} == before
    assert sorted(editor.js("__app.points.filter(p => p.overlaid).map(p => p.id)")) == sorted(f"spa_Latn-{i}" for i in range(SENTENCES))
    assert editor.selection == ["eng_Latn-0"]
    assert "12 overlaid" in editor.page.inner_text("#status")
    assert "12 of 48 points" in editor.page.inner_text("#overlay-note")
    # Only the new language was embedded, and the cached ones were reused.
    assert embedder_calls(editor)[-1] == {"lang_codes": ["spa_Latn"], "n_sentences": SENTENCES, "start": 0, "model": "xlm-roberta-base", "pooling": "mean"}


def test_adding_sentences_embeds_only_the_new_ones(editor):
    before = positions(editor)
    update_to(editor, BASE, SENTENCES + 4)
    editor.wait_for_status("48 points")
    editor.settle()

    after = positions(editor)
    assert {i: after[i] for i in before} == before
    assert editor.js("__app.points.filter(p => p.overlaid).length") == 12
    assert embedder_calls(editor)[-1]["start"] == SENTENCES
    assert embedder_calls(editor)[-1]["n_sentences"] == SENTENCES + 4


def test_refit_folds_the_overlaid_points_into_the_layout(editor):
    update_to(editor, BASE + ["spa_Latn"])
    editor.wait_for_status("12 overlaid")
    editor.settle()
    fit = editor.js("__app._fitId")
    before = positions(editor)

    editor.page.click("#overlay-refit")
    editor.page.wait_for_function("f => __app._fitId !== f && document.getElementById('overlay-note').hidden", arg=fit)
    editor.settle()
    assert editor.js("__app.points.some(p => p.overlaid)") is False
    assert positions(editor) != before


def test_token_overlays_survive_an_edit_but_not_a_refit(editor):
    editor.page.mouse.click(*editor.screen_of("eng_Latn-0"))
    editor.page.locator("tr.dt-row").first.locator(".actions-menu-btn").click()
    editor.page.locator(".actions-menu-dropdown.open .actions-menu-line", has_text="Tokens").click()
    tokens = "__app.view.pointLayer._tokenOverlayEntries.length"
    editor.page.wait_for_function(f"{tokens} > 0")
    shown = editor.js(tokens)

    update_to(editor, BASE + ["spa_Latn"])  # the layout is untouched, so the tokens still belong in it
    editor.wait_for_status("48 points")
    editor.settle()
    assert editor.js(tokens) == shown

    editor.page.click("#overlay-refit")
    editor.page.wait_for_function("document.getElementById('overlay-note').hidden")
    editor.settle()
    assert editor.js(tokens) == 0


def test_languages_keep_their_colours_when_a_language_is_added(editor):
    colours = "Object.fromEntries(__app.view.pointLayer.points.map(d => [d.id, __app.view.pointLayer._colorFor(d)]))"
    before = editor.js(colours)
    update_to(editor, ["arb_Arab"] + BASE)  # sorts before every language already shown
    editor.wait_for_status("48 points")
    editor.settle()
    after = editor.js(colours)
    assert {i: after[i] for i in before} == before
    assert after["arb_Arab-0"] not in set(before.values())


def test_removing_a_language_drops_its_points_and_selection(editor):
    before = positions(editor)
    for point in ("eng_Latn-0", "fra_Latn-0"):
        editor.page.mouse.click(*editor.screen_of(point))
    assert editor.selection == ["eng_Latn-0", "fra_Latn-0"]

    update_to(editor, ["eng_Latn", "deu_Latn"])
    editor.wait_for_status("24 points")
    editor.settle()

    after = positions(editor)
    assert after == {i: before[i] for i in after} and len(after) == 24
    assert editor.selection == ["eng_Latn-0"]
    assert editor.page.is_hidden("#overlay-note")


def test_a_failed_edit_explains_itself_and_keeps_the_plot(editor):
    before = positions(editor)
    update_to(editor, BASE + ["qqq_Latn"])
    editor.page.wait_for_selector("#flores-error:not([hidden])")
    assert "qqq_Latn" in editor.page.inner_text("#flores-error")
    assert "update failed" in editor.page.inner_text("#status")
    editor.settle()
    assert positions(editor) == before
    assert editor.page.is_enabled("#flores-load-button")


def test_loading_new_replaces_the_dataset_and_clears_the_selection(editor):
    editor.page.mouse.click(*editor.screen_of("eng_Latn-0"))
    editor.page.click("#flores-mode button[data-mode=load]")
    editor.page.select_option("#flores-pooling", "max")
    editor.page.click("#flores-load-button")
    editor.page.wait_for_function("document.getElementById('status').textContent.includes('max-pooled')")
    editor.settle()
    assert editor.selection == []
    assert editor.js("__app.points.length") == len(BASE) * SENTENCES
    assert editor.page.locator("#flores-mode button[data-mode=edit]").get_attribute("aria-pressed") == "true"
    assert editor.page.input_value("#flores-pooling") == "max"
    assert editor.page.is_disabled("#flores-pooling")


def test_an_uploaded_dataset_only_offers_loading(editor):
    editor.load_sample()
    page = editor.page
    assert page.is_hidden("#flores-mode")
    assert page.inner_text("#flores-load-button") == "Load FLORES-200"
    assert page.is_enabled("#flores-model")
