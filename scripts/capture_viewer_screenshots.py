"""Capture representative states of the live benchmark viewer."""

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://jacobpevans-mlx-benchmarks-viewer.hf.space"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 2400})

        started = time.time()
        page.goto(URL, timeout=120_000)
        page.wait_for_selector("text=results across", timeout=120_000)
        print(f"cold load to data: {time.time() - started:.1f}s")
        page.screenshot(path=str(output_dir / "v-default.png"), full_page=True)

        page.get_by_text("Speed / diagnostic", exact=True).click()
        page.wait_for_timeout(6_000)
        page.screenshot(path=str(output_dir / "v-diagnostic.png"), full_page=True)

        page.get_by_text("Agentic & coding", exact=True).click()
        page.wait_for_timeout(6_000)

        page.get_by_role("combobox", name="Machine").click()
        page.get_by_role("option", name="RTX PRO 6000 Max-Q 96GB").click()
        page.keyboard.press("Escape")
        page.wait_for_timeout(6_000)
        page.screenshot(path=str(output_dir / "v-throughput-knob.png"), full_page=True)

        started = time.time()
        page.get_by_role("combobox", name="Suite").click()
        page.get_by_role("option").first.wait_for(state="visible", timeout=60_000)
        options = page.get_by_role("option").all_inner_texts()
        if not options:
            raise RuntimeError("The viewer has no suite options")
        target = next(
            (item for item in options if "repo" in item.lower() or "coding" in item.lower()), options[-1]
        )
        page.get_by_role("option", name=target).click()
        page.wait_for_timeout(6_000)
        print(f"suite switch: {time.time() - started:.1f}s -> {target}")
        page.screenshot(path=str(output_dir / "v-eval.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
