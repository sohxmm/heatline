"""
Automated Browser Verification and Screenshot Capture for Streamlit App.
Uses Playwright to render all 4 tabs and capture high-resolution verification screenshots.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("results/app_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8501"


def capture_all_tabs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        print(f"Connecting to {URL}...")
        page.goto(URL, wait_until="networkidle")
        time.sleep(3)

        # Tab 1: Historical May Replay
        print("Capturing Tab 1: Historical May Replay...")
        tab1_btn = page.get_by_role("tab").filter(has_text="Historical May Replay")
        tab1_btn.click()
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "tab1_replay.png"), full_page=False)

        # Tab 2: Interactive What-If Simulator
        print("Capturing Tab 2: Interactive What-If Simulator...")
        tab2_btn = page.get_by_role("tab").filter(has_text="Interactive What-If Simulator")
        tab2_btn.click()
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "tab2_whatif.png"), full_page=False)

        # Tab 3: Model, Method & Benchmark Results
        print("Capturing Tab 3: Model, Method & Benchmark Results...")
        tab3_btn = page.get_by_role("tab").filter(has_text="Model, Method")
        tab3_btn.click()
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "tab3_explorer.png"), full_page=False)

        # Tab 4: Syllabus Alignment & Viva Guide
        print("Capturing Tab 4: Syllabus Alignment & Viva Guide...")
        tab4_btn = page.get_by_role("tab").filter(has_text="Syllabus Alignment")
        tab4_btn.click()
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "tab4_math_viva.png"), full_page=False)

        browser.close()
        print("Done! All 4 screenshots captured successfully.")


if __name__ == "__main__":
    capture_all_tabs()
