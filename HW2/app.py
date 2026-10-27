"""HW2 Streamlit entry point; original HW1 remains available in sidebar."""
from pathlib import Path
import sys
import runpy

import streamlit as st

page = st.sidebar.selectbox("選擇作業", ["HW2｜多元線性回歸", "HW1｜天氣儀表板"])
if page.startswith("HW1"):
    folder = Path(__file__).resolve().parent / "hw1"
    if not folder.exists():
        folder = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(folder))
    runpy.run_path(str(folder / "app.py"), run_name="__main__")
else:
    from regression_ui import render
    render()
