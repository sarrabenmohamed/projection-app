"""Browser tests: a real uvicorn server plus headless Chromium. Skipped when
Playwright or its browser isn't installed (pip install -r requirements-dev.txt
&& playwright install chromium)."""
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "sample_embeddings.json"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/dataset/info", timeout=1)
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("uvicorn exited during startup")
                time.sleep(0.5)
        else:
            pytest.fail("uvicorn did not start in time")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binary missing
            pytest.skip(f"Chromium not available: {e}")
        yield browser
        browser.close()


class Ui:
    """The app in a page, with helpers that read its state through window.__app."""

    def __init__(self, page, url):
        self.page = page
        self.url = url
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))

    def open(self):
        # Expose the App instance so tests can read selection/points directly.
        def expose(route):
            body = route.fetch().text().replace("new App().init();", "window.__app = new App(); window.__app.init();")
            route.fulfill(content_type="application/javascript", body=body)

        self.page.route("**/client-js/app.js", expose)
        self.page.goto(self.url)
        self.page.wait_for_function("window.__app && document.getElementById('status').textContent.length > 0")
        return self

    def load_sample(self):
        self.page.set_input_files("#file-input", str(SAMPLE))
        self.wait_for_status("9 points")
        self.settle()
        self.page.wait_for_function("__app.view.pointLayer.points.length === 9")
        return self

    def wait_for_status(self, text):
        self.page.wait_for_function("t => document.getElementById('status').textContent.includes(t)", arg=text)

    def settle(self):
        self.page.wait_for_function("!document.getElementById('loading-overlay').classList.contains('visible')")

    def js(self, expression, arg=None):
        return self.page.evaluate(expression, arg)

    @property
    def selection(self):
        return self.js("__app._selection")

    @property
    def layer(self):
        return "__app.view.pointLayer"

    def screen_of(self, point_id):
        """Client (x, y) where the point is drawn right now."""
        return self.js(
            """id => {
                const v = __app.view, d = v.pointLayer.points.find(p => p.id === id);
                const [x, y] = v.pointLayer._displayedPosition(d);
                const [sx, sy] = v.axes.transform.apply([x, y]);
                const r = v.canvas.getBoundingClientRect();
                return [r.left + v.margin.left + sx, r.top + v.margin.top + sy];
            }""",
            point_id,
        )

    def canvas_box(self):
        return self.page.locator("#projection canvas").bounding_box()


@pytest.fixture
def ui(browser, server_url):
    context = browser.new_context(viewport={"width": 1300, "height": 900})
    page = context.new_page()
    helper = Ui(page, server_url).open()
    yield helper
    context.close()
    assert helper.errors == [], f"uncaught page errors: {helper.errors}"
