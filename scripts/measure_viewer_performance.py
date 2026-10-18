"""Measure load and interaction latency for the live benchmark viewer."""

import time

from playwright.sync_api import sync_playwright

URL = "https://jacobpevans-mlx-benchmarks-viewer.hf.space"
IDLE = "!document.body.innerText.includes('processing')"


with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    for index in range(3):
        page = browser.new_page(viewport={"width": 1440, "height": 2400})

        started = time.time()
        page.goto(URL, timeout=180_000)
        page.wait_for_selector("text=results across", timeout=180_000)
        page_load = time.time() - started

        started = time.time()
        page.get_by_text("Speed / diagnostic", exact=True).click()
        page.wait_for_timeout(300)
        page.wait_for_function(IDLE, timeout=60_000)
        mode_switch = time.time() - started

        started = time.time()
        page.get_by_text("Experimental", exact=True).click()
        page.wait_for_timeout(300)
        page.wait_for_function(IDLE, timeout=60_000)
        evidence_switch = time.time() - started

        print(
            f"run{index + 1}: page->data {page_load:.1f}s | "
            f"mode switch {mode_switch:.1f}s | evidence switch {evidence_switch:.1f}s"
        )
        page.close()
    browser.close()
