from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_both_builtin_problems_and_custom_entry():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    assert not app.exception
    app.button[0].click().run(timeout=90)
    assert not app.exception and not app.error
    assert app.success
    app.selectbox[1].select('汽車價格預測').run()
    app.button[0].click().run(timeout=90)
    assert not app.exception and not app.error
    assert app.success
    # Submission form lives in the new-sample tab, and evaluates actual inputs.
    [b for b in app.button if b.label=='預測這筆資料'][0].click().run()
    assert not app.exception and not app.error
    app.selectbox[1].select('自訂 CSV 資料集').run()
    assert not app.exception
