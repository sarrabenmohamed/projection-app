"""End-to-end checks of the flows that unit tests can't reach: they drive the
real page against the real backend with the 9-point sample dataset
(en/fr/de x 3 sentences)."""
import json

import pytest

pytestmark = pytest.mark.e2e


def test_default_colouring_is_per_language(ui):
    ui.load_sample()
    assert ui.page.input_value("#color-mode-select") == "language"
    assert sorted(ui.page.locator("#color-legend .legend-item").all_inner_texts()) == ["de", "en", "fr"]
    colours = ui.js("[...new Set(__app.view.pointLayer.points.map(d => __app.view.pointLayer._colorFor(d)))]")
    assert len(colours) == 3
    assert ui.page.locator("#lang-picker").is_hidden()


def test_colour_modes_switch_legend_and_picker(ui):
    ui.load_sample()
    ui.page.select_option("#color-mode-select", "family")
    assert ui.page.locator("#color-legend .legend-item").count() >= 2
    ui.page.select_option("#color-mode-select", "custom")
    assert ui.page.locator("#lang-picker").is_visible()
    assert ui.page.locator("#color-legend .legend-item").count() == 0


def test_clicking_a_point_pins_and_unpins_it(ui):
    ui.load_sample()
    x, y = ui.screen_of("en-0")
    ui.page.mouse.click(x, y)
    assert ui.selection == ["en-0"]
    ui.page.mouse.click(x, y)
    assert ui.selection == []


def test_clicking_a_table_row_toggles_its_pin(ui):
    ui.load_sample()
    ui.page.locator("tr.dt-row").nth(0).locator("td").nth(3).click()
    assert len(ui.selection) == 1


def test_search_filters_the_table(ui):
    ui.load_sample()
    ui.page.fill(".dt-search", "lang=fr")
    assert ui.page.locator("tr.dt-row").count() == 3


def test_right_drag_selects_the_points_in_the_box(ui):
    ui.load_sample()
    ids = ["en-0", "fr-0"]
    (x0, y0), (x1, y1) = (ui.screen_of(i) for i in ids)
    left, right = min(x0, x1) - 15, max(x0, x1) + 15
    top, bottom = min(y0, y1) - 15, max(y0, y1) + 15
    expected = ui.js(
        """([l, t, r, b]) => __app.view.pointLayer.points.filter(d => {
            const [x, y] = __app.view.pointLayer._displayedPosition(d);
            const [sx, sy] = __app.view.axes.transform.apply([x, y]);
            const box = __app.view.canvas.getBoundingClientRect(), m = __app.view.margin;
            const cx = box.left + m.left + sx, cy = box.top + m.top + sy;
            return cx >= l && cx <= r && cy >= t && cy <= b;
        }).map(d => d.id)""",
        [left, top, right, bottom],
    )
    assert set(ids) <= set(expected)
    ui.page.mouse.move(left, top)
    ui.page.mouse.down(button="right")
    ui.page.mouse.move(right, bottom, steps=5)
    ui.page.mouse.up(button="right")
    assert sorted(ui.selection) == sorted(expected)


def test_reprojecting_keeps_selection_and_colouring(ui):
    ui.load_sample()
    ui.page.select_option("#color-mode-select", "custom")
    ui.page.fill("#lang-search", "fr")
    ui.page.locator(".lang-suggestion-item").first.dispatch_event("mousedown")
    x, y = ui.screen_of("en-1")
    ui.page.mouse.click(x, y)
    ui.page.select_option("#metric-select", "cosine")
    ui.wait_for_status("9 points")
    ui.settle()
    assert ui.selection == ["en-1"]
    assert ui.js("[...__app.view.pointLayer._selectedLangs]") == ["fr"]


def test_3d_view_rotates_and_still_hit_tests(ui):
    ui.load_sample()
    ui.page.mouse.click(*ui.screen_of("de-1"))
    ui.page.locator("#view3d-toggle").check()
    ui.wait_for_status("(3D)")
    ui.settle()
    assert ui.js("__app.view.pointLayer.dims") == 3
    assert ui.selection == ["de-1"]

    box = ui.canvas_box()
    yaw = ui.js("__app.view.pointLayer._yaw")
    ui.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    ui.page.mouse.down()
    ui.page.mouse.move(box["x"] + box["width"] / 2 + 150, box["y"] + box["height"] / 2 + 40, steps=6)
    ui.page.mouse.up()
    assert ui.js("__app.view.pointLayer._yaw") != yaw
    assert ui.selection == ["de-1"]  # a rotate-drag is not a click

    ui.page.mouse.move(*ui.screen_of("en-0"))
    assert ui.js("__app.view._hoveredId") == "en-0"

    ui.page.locator("#view3d-reset").click()
    assert ui.js("__app.view.pointLayer._yaw") == pytest.approx(0.6)
    ui.page.locator("#view3d-toggle").uncheck()
    ui.wait_for_status("umap")
    ui.settle()
    assert ui.js("__app.view.pointLayer.dims") == 2


def test_3d_drag_outside_the_box_pans_instead_of_rotating(ui):
    ui.load_sample()
    ui.page.locator("#view3d-toggle").check()
    ui.wait_for_status("(3D)")
    ui.settle()
    assert ui.js("__app.view.pointLayer.boxHull().length") >= 4

    box = ui.canvas_box()
    yaw = ui.js("__app.view.pointLayer._yaw")
    pan = ui.js("__app.view.axes.transform.x")
    ui.page.mouse.move(box["x"] + 8, box["y"] + 8)
    ui.page.mouse.down()
    ui.page.mouse.move(box["x"] + 60, box["y"] + 40, steps=5)
    ui.page.mouse.up()
    assert ui.js("__app.view.pointLayer._yaw") == yaw
    assert ui.js("__app.view.axes.transform.x") != pan


def test_fisheye_hit_tests_at_displaced_positions(ui):
    ui.load_sample()
    ui.page.locator("#fisheye-toggle").check()
    box = ui.canvas_box()
    cx, cy = ui.js("[__app.view._lensCenter.x, __app.view._lensCenter.y]")
    ui.page.mouse.move(box["x"] + 5, box["y"] + 5)  # draw once with the lens on
    near = ui.js(
        """([cx, cy]) => __app.view.pointLayer.points.map(d => {
            const [x, y] = __app.view.pointLayer._displayedPosition(d);
            return [d.id, Math.hypot(x - cx, y - cy)];
        }).sort((a, b) => a[1] - b[1])[0][0]""",
        [cx, cy],
    )
    ui.page.mouse.move(*ui.screen_of(near))
    assert ui.js("__app.view._hoveredId") == near


def test_tokens_overlay_follows_the_pinned_sentence(ui):
    ui.load_sample()
    ui.page.mouse.click(*ui.screen_of("en-0"))
    row = ui.page.locator("tr.dt-row").first
    row.locator(".actions-menu-btn").click()
    ui.page.locator(".actions-menu-dropdown.open .actions-menu-line", has_text="Tokens").click()
    ui.page.wait_for_function("__app.view.pointLayer._tokenOverlayEntries.length > 0")
    ui.page.locator("#view3d-toggle").check()
    ui.wait_for_status("(3D)")
    ui.settle()
    assert ui.js("__app.view.pointLayer._tokenOverlayEntries.length") == 0  # a refit drops them


def test_select_translations_pins_the_whole_group(ui):
    ui.load_sample()
    ui.page.mouse.click(*ui.screen_of("en-0"))
    ui.page.locator("tr.dt-row").first.locator(".actions-menu-btn").click()
    ui.page.locator(".actions-menu-dropdown.open .actions-menu-line", has_text="translations").click()
    assert sorted(ui.selection) == ["de-0", "en-0", "fr-0"]
    assert ui.page.locator(".dt-show-selected").is_checked()


def test_flores_error_is_shown_in_the_sidebar(ui):
    ui.page.route("**/dataset/flores", lambda r: r.fulfill(
        status=400, content_type="application/json", body=json.dumps({"detail": "unknown language zzz"})))
    ui.page.locator("#flores-controls summary").click()
    ui.page.click("#flores-load-button")
    ui.page.wait_for_selector("#flores-error:not([hidden])")
    assert "unknown language zzz" in ui.page.inner_text("#flores-error")


def test_cancel_stops_a_running_flores_load(ui):
    pending, cancels = [], []
    ui.page.route("**/dataset/flores/cancel", lambda r: (cancels.append(1), r.fulfill(
        status=200, content_type="application/json", body='{"cancelled": true}')))
    ui.page.route("**/dataset/flores", lambda r: pending.append(r) if r.request.url.endswith("/flores") else r.fallback())
    ui.page.locator("#flores-controls summary").click()
    ui.page.click("#flores-load-button")
    ui.page.wait_for_selector("#loading-cancel:not([hidden])")
    ui.page.click("#loading-cancel")
    ui.page.wait_for_function("!document.getElementById('loading-overlay').classList.contains('visible')")
    assert cancels == [1]
    assert "cancelled" in ui.page.inner_text("#status")
    assert ui.page.locator("#flores-load-button").is_enabled()
    for route in pending:
        route.abort()


def test_canvas_follows_the_window_size(ui):
    ui.load_sample()
    ui.page.set_viewport_size({"width": 1300, "height": 1200})
    ui.page.wait_for_function(
        "() => { const r = document.querySelector('#projection').getBoundingClientRect();"
        " return document.querySelector('#projection canvas').getBoundingClientRect().height === r.height; }"
    )
    assert ui.js("document.querySelector('#projection canvas').getBoundingClientRect().bottom") == ui.js(
        "document.querySelector('#projection').getBoundingClientRect().bottom"
    )


def test_tokens_action_is_disabled_on_tsne_instead_of_switching_to_umap(ui):
    ui.load_sample()
    ui.page.select_option("#method-select", "tsne")
    ui.wait_for_status("tsne")
    ui.settle()
    ui.page.mouse.click(*ui.screen_of("de-1"))
    ui.page.locator(".actions-menu-btn").first.click()
    line = ui.page.locator(".actions-menu-line", has_text="Tokens")
    assert "actions-menu-disabled" in line.get_attribute("class")
    assert ui.js("document.getElementById('method-select').value") == "tsne"
