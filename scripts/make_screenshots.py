"""Walks through the app with a real (cached) FLORES-200 dataset and saves
screenshots of every view, panel and setting into one folder, for figures.

Every state is saved as the whole window (NN_name.png) and, where one panel
is the point, as that panel alone (NN_name__plot.png, __stats.png, ...).
Images are rendered at 2x so they stay sharp in print.

The embeddings must already be cached under data/ (load them once in the app);
nothing is embedded here.

Usage:
    python scripts/make_screenshots.py [--out screenshots] [--n 100]
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "zho_Hans", "arb_Arab"]
ADDED_LANG = "spa_Latn"
VIEWPORT = {"width": 1600, "height": 1000}

PANELS = {"plot": "#projection", "table": "#data-table-panel", "stats": "#stats-panel", "sidebar": "#sidebar"}


@contextlib.contextmanager
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/dataset/info", timeout=1)
                break
            except OSError:
                if proc.poll() is not None:
                    raise SystemExit("the server exited during startup")
                time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class Shooter:
    def __init__(self, page, out: Path):
        self.page, self.out, self.count = page, out, 0

    def settle(self, text=None):
        self.page.wait_for_function("!document.getElementById('loading-overlay').classList.contains('visible')")
        if text:
            self.page.wait_for_function("t => document.getElementById('status').textContent.includes(t)", arg=text)
        self.page.wait_for_timeout(600)  # let the canvas and the opacity transitions finish

    def shot(self, name, panels=()):
        self.count += 1
        stem = f"{self.count:02d}_{name}"
        self.page.screenshot(path=str(self.out / f"{stem}.png"))
        for panel in panels:
            self.page.locator(PANELS[panel]).screenshot(path=str(self.out / f"{stem}__{panel}.png"))
        print(f"  {stem}")

    def point_xy(self, point_id):
        return self.page.evaluate(
            """id => {
                const v = __app.view, d = v.pointLayer.points.find(p => p.id === id);
                const [x, y] = v.pointLayer._displayedPosition(d);
                const [sx, sy] = v.axes.transform.apply([x, y]);
                const r = v.canvas.getBoundingClientRect();
                return [r.left + v.margin.left + sx, r.top + v.margin.top + sy];
            }""", point_id)

    def choose(self, selector, value, status=None):
        self.page.select_option(selector, value)
        self.settle(status)

    def open_actions(self, row=0):
        self.page.locator(".actions-menu-btn").nth(row).click()


def run(url: str, out: Path, n: int):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        def expose(route):
            body = route.fetch().text().replace("new App().init();", "window.__app = new App(); window.__app.init();")
            route.fulfill(content_type="application/javascript", body=body)

        page.route("**/client-js/app.js", expose)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_default_timeout(120_000)
        page.goto(url)
        page.wait_for_function("window.__app && document.getElementById('status').textContent.length > 0")
        s = Shooter(page, out)

        s.shot("empty_start")
        page.locator("#format-help-toggle").click()
        s.shot("upload_format_help", ["sidebar"])
        page.locator("#format-help-toggle").click()

        page.locator("#flores-controls summary").click()
        page.fill("#flores-langs", ", ".join(LANGS))
        page.fill("#flores-n", str(n))
        s.shot("flores_form", ["sidebar"])
        page.click("#flores-load-button")
        s.settle(f"{len(LANGS) * n} points")
        page.locator("#flores-controls summary").click()
        s.shot("loaded_umap_colour_by_language", ["plot", "table", "sidebar"])

        sample = f"{LANGS[0]}-0"
        page.mouse.move(*s.point_xy(sample))
        page.wait_for_timeout(400)
        s.shot("hover_tooltip", ["plot"])
        page.mouse.move(5, 5)

        s.choose("#color-mode-select", "family")
        s.shot("colour_by_language_family", ["plot", "sidebar"])
        s.choose("#color-mode-select", "custom")
        page.fill("#lang-search", "fra")
        page.locator("#lang-suggestions > *").first.click()
        s.shot("colour_chosen_languages_only", ["plot", "sidebar"])
        s.choose("#color-mode-select", "language")

        for metric in ("cosine", "manhattan", "euclidean"):
            s.choose("#metric-select", metric, "umap")
            if metric != "euclidean":
                s.shot(f"metric_{metric}", ["plot", "sidebar"])

        page.locator("#projection-params summary").click()
        s.shot("umap_parameters", ["sidebar"])
        s.choose("#method-select", "tsne", "tsne")
        s.shot("tsne_parameters_and_plot", ["plot", "sidebar"])
        page.locator("#projection-params summary").click()
        s.choose("#method-select", "umap", "umap")

        page.locator("#view3d-toggle").check()
        s.settle("(3D)")
        s.shot("view_3d", ["plot"])
        box = page.locator("#projection canvas").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 140, cy + 60, steps=10)
        page.mouse.up()
        page.wait_for_timeout(500)
        s.shot("view_3d_rotated", ["plot"])
        page.locator("#view3d-toggle").uncheck()
        s.settle("umap")

        page.locator("#fisheye-toggle").check()
        page.wait_for_timeout(600)
        s.shot("fisheye_lens", ["plot", "sidebar"])
        page.locator("#fisheye-toggle").uncheck()

        # A pinned sentence, its translations and its tokens.
        page.mouse.click(*s.point_xy(sample))
        page.wait_for_timeout(500)
        s.shot("pinned_point", ["plot", "table", "stats"])
        s.open_actions()
        page.locator(".actions-menu-line", has_text="Select translations").first.click()
        page.wait_for_timeout(1200)
        s.shot("select_translations", ["plot", "table", "stats"])
        page.locator(".actions-menu-line", has_text="Tokens").first.click()
        s.settle()
        page.wait_for_timeout(600)
        s.shot("token_overlay", ["plot", "table"])
        page.locator("#data-table-panel .dt-clear-selection, #data-table-panel button", has_text="Clear selection").first.click()
        page.wait_for_timeout(400)

        # Area select: right-drag a box around the French cluster.
        xs, ys = zip(*(s.point_xy(f"{LANGS[1]}-{i}") for i in range(n)))
        left, top, right, bottom = min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15
        page.mouse.move(left, top)
        page.mouse.down(button="right")
        page.mouse.move(right, bottom, steps=8)
        s.shot("area_select_drag", ["plot"])
        page.mouse.up(button="right")
        page.wait_for_timeout(1200)
        s.shot("area_select_result", ["plot", "stats", "table"])
        page.locator("#data-table-panel button", has_text="Clear selection").first.click()

        page.fill("#data-table .dt-search, .dt-search", "lang=fra_Latn")
        page.wait_for_timeout(500)
        s.shot("data_table_search", ["table"])
        page.fill(".dt-search", "")

        # Editing the loaded dataset: add a language and overlay it.
        page.locator("#flores-controls summary").click()
        page.fill("#flores-langs", ", ".join(LANGS + [ADDED_LANG]))
        s.shot("edit_dataset_form", ["sidebar"])
        page.click("#flores-load-button")
        s.settle(f"{(len(LANGS) + 1) * n} points")
        page.locator("#flores-controls summary").click()
        s.shot("edit_dataset_overlaid_language", ["plot", "sidebar"])
        page.locator("#overlay-refit").click()
        s.settle(f"{(len(LANGS) + 1) * n} points")
        s.shot("after_refit", ["plot"])

        for name, button in (("projection", "#projection-fullscreen-btn"), ("table", "#data-table-fullscreen-btn"),
                             ("stats", "#stats-fullscreen-btn")):
            page.locator(button).click()
            page.wait_for_timeout(700)
            s.shot(f"fullscreen_{name}")
            page.locator(button).click()
            page.wait_for_timeout(400)
        page.locator("[data-collapse=sidebar]").click()
        page.wait_for_timeout(600)
        s.shot("sidebar_collapsed")

        browser.close()
        if errors:
            print("page errors:", *errors, sep="\n  ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "screenshots")
    parser.add_argument("--n", type=int, default=100, help="sentences per language")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()
    with server() as url:
        run(url, args.out, args.n)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
