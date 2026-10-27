"""Spec item 10 - the UI launches, every page renders, and controls reach the backend."""
import os
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
PAGES = ["Overview", "Data-Center Simulator", "Scheduling Algorithms", "DRL Training",
         "Experimental Results", "Architecture & Methodology"]


def fresh():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def goto(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page):
    goto(fresh(), page)


def test_overview_shows_backend_state():
    at = fresh()
    labels = [m.label for m in at.metric]
    assert "Max temp (°C)" in labels and "Tasks completed" in labels


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_simulator_step_buttons_advance_the_real_simulation():
    at = goto(fresh(), "Data-Center Simulator")
    assert at.session_state.sim.step_count == 0
    _button(at, "Step ×10").click().run()
    assert at.session_state.sim.step_count == 10
    _button(at, "Run ×100").click().run()
    assert at.session_state.sim.step_count == 110
    assert len(at.session_state.sim.history) == 110
    _button(at, "Reset").click().run()
    assert at.session_state.sim.step_count == 0


def test_simulator_run_to_end_finishes_episode():
    at = goto(fresh(), "Data-Center Simulator")
    _button(at, "Run to end").click().run()
    sim = at.session_state.sim
    assert sim.done and sim.step_count == sim.cfg.episode_length
    assert _button(at, "Step ×1").disabled


def test_overview_buttons_advance_simulation():
    at = fresh()
    _button(at, "Run 100 steps").click().run()
    assert at.session_state.sim.step_count == 100


def test_invalid_configuration_is_rejected_not_crashed():
    at = goto(fresh(), "Data-Center Simulator")
    # 12 servers cannot be split into 8 racks -> backend ValueError -> friendly error
    at.selectbox[0].set_value(12)
    at.selectbox[1].set_value(8)
    next(b for b in at.button if "Apply configuration" in b.label).click().run()
    assert not at.exception
    assert any("rejected" in e.value for e in at.error)
    assert at.session_state.sim.cfg.facility.n_servers == 16     # unchanged


def test_valid_configuration_rebuilds_simulation():
    at = goto(fresh(), "Data-Center Simulator")
    at.selectbox[0].set_value(8)
    at.selectbox[1].set_value(2)
    next(b for b in at.button if "Apply configuration" in b.label).click().run()
    assert not at.exception
    assert at.session_state.sim.cfg.facility.n_servers == 8


def test_algorithm_comparison_runs_real_evaluation():
    at = goto(fresh(), "Scheduling Algorithms")
    at.multiselect[0].set_value(["round-robin", "greedy-coolest"])
    at.number_input[0].set_value(200)
    at.slider[0].set_value(2)
    _button(at, "Run comparison").click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state.last_comparison
    assert set(res["summary"]["policy"]) == {"round-robin", "greedy-coolest"}
    assert len(res["raw"]) == 4                      # 2 policies x 2 seeds


def test_training_launch_rejects_reserved_evaluation_seed():
    at = goto(fresh(), "DRL Training")
    at.number_input[1].set_value(1001)               # training seed field
    next(b for b in at.button if b.label == "Start training").click().run()
    assert any("reserved" in e.value for e in at.error)


def test_launch_training_subprocess_end_to_end(tmp_path, monkeypatch):
    import time
    from dashboard import common
    monkeypatch.setattr(common, "UI_MODELS_DIR", str(tmp_path))
    info = common.launch_training("PPO", 2048, 21, "steady", 1.0, 0.5, 0.3)
    code = info["proc"].wait(timeout=240)
    assert code == 0, common.tail(os.path.join(info["outdir"], "train.log"))
    from dcsched.training import list_runs
    recs = list_runs(info["outdir"])
    assert recs and recs[0]["status"] == "complete" and "eval" in recs[0]


def test_launch_training_validation():
    from dashboard import common
    with pytest.raises(ValueError):
        common.launch_training("PPO", 2048, 1003, "steady", 1, 0.5, 0.3)
    with pytest.raises(ValueError):
        common.launch_training("PPO", 100, 5, "steady", 1, 0.5, 0.3)
