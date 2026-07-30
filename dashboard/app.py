"""Live thesis-defense dashboard for the capacity-mechanism microgrid model.

Drives the real MicrogridModel (src/microgrid_sim/environment/model.py)
directly: every chart and readout is a read of that model's own public state
after a real model.step() call (see model_state.py's module docstring for
exactly which reads happen where). Nothing in this file reimplements broker
pricing, agent decisions, or the capacity mechanism.

Framework: Dash (already installed at 3.2.0, alongside dash-bootstrap-
components and plotly) instead of Mesa's own built-in SolaraViz. Mesa 3.5.1's
mesa.visualization package cannot even be imported in this environment: its
__init__.py unconditionally imports mesa.visualization.command_console,
which does `import solara`, and solara is not installed here (see
docs/thesis/demo_instructions.md for the exact traceback). Dash needed zero
new packages.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import figures
from model_state import (
    ABLATION_LABELS,
    ABLATIONS,
    BROKER_COUNTS,
    DEFAULT_ABLATION,
    DEFAULT_BROKER_COUNT,
    DEFAULT_K,
    HORIZON_HOURS,
    WINDOW_HOURS,
    DemoSession,
)

# ---------------------------------------------------------------------------
# speed presets (display-pacing only; never resets the model)
# ---------------------------------------------------------------------------

SPEED_PRESETS = {
    "slow": {"label": "Slow (1 h/tick)", "steps_per_tick": 1, "interval_ms": 400},
    "normal": {"label": "Normal (5 h/tick)", "steps_per_tick": 5, "interval_ms": 300},
    "fast": {"label": "Fast (20 h/tick)", "steps_per_tick": 20, "interval_ms": 200},
    "very_fast": {"label": "Very fast (60 h/tick)", "steps_per_tick": 60, "interval_ms": 150},
}
DEFAULT_SPEED_KEY = "normal"

K_MARKS = {value: str(value) for value in (0.5, 1.0, 1.5, 2.0)}

session = DemoSession()
_session_lock = threading.Lock()

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Microgrid capacity mechanism - live demo")
server = app.server


def _controls_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Controls", className="card-title"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Start", id="btn-start", color="success", n_clicks=0),
                        dbc.Button("Pause", id="btn-pause", color="warning", n_clicks=0),
                        dbc.Button("Step", id="btn-step", color="secondary", n_clicks=0),
                        dbc.Button("Reset", id="btn-reset", color="danger", n_clicks=0),
                    ],
                    className="mb-3",
                ),
                html.Label("Speed", className="fw-bold"),
                dcc.Dropdown(
                    id="speed-dropdown",
                    options=[{"label": preset["label"], "value": key} for key, preset in SPEED_PRESETS.items()],
                    value=DEFAULT_SPEED_KEY,
                    clearable=False,
                    className="mb-3",
                ),
                html.Label("Capacity threshold sensitivity, k (threshold = mean + k * std)", className="fw-bold"),
                dcc.Slider(
                    id="k-slider",
                    min=0.1,
                    max=3.0,
                    step=0.1,
                    value=DEFAULT_K,
                    marks=K_MARKS,
                    updatemode="mouseup",
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(className="mb-3"),
                html.Label("Ablation", className="fw-bold"),
                dcc.Dropdown(
                    id="ablation-dropdown",
                    options=[{"label": ABLATION_LABELS[name], "value": name} for name in ABLATIONS],
                    value=DEFAULT_ABLATION,
                    clearable=False,
                    className="mb-3",
                ),
                html.Label("Broker count", className="fw-bold"),
                dcc.Dropdown(
                    id="broker-count-dropdown",
                    options=[
                        {"label": f"{count} ({'monopoly' if count == 1 else 'competing brokers'})", "value": count}
                        for count in BROKER_COUNTS
                    ],
                    value=DEFAULT_BROKER_COUNT,
                    clearable=False,
                ),
            ]
        ),
        class_name="mb-3",
    )


def _deferred_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody(id="deferred-indicator"), class_name="mb-3")


def _metrics_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody(id="metrics-panel"), class_name="mb-3")


app.layout = dbc.Container(
    [
        html.H3("Microgrid capacity mechanism: live demo", className="mt-3"),
        html.Div(
            "Drives the real MicrogridModel used throughout the thesis (no reimplementation, no replayed data). "
            "capacity_passthrough is a signal-strength coefficient, not a price; the surcharge shown below is a "
            "real EUR/kWh quote adder.",
            className="text-muted mb-3",
        ),
        html.Div(id="status-bar", className="mb-3"),
        dbc.Row(
            [
                dbc.Col([_controls_card(), _deferred_card(), _metrics_card()], width=12, lg=3),
                dbc.Col(
                    [
                        dcc.Graph(id="feeder-graph", config={"displayModeBar": False}),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="broker-share-graph", config={"displayModeBar": False}), width=12, lg=7),
                                dbc.Col(dcc.Graph(id="surcharge-graph", config={"displayModeBar": False}), width=12, lg=5),
                            ]
                        ),
                    ],
                    width=12,
                    lg=9,
                ),
            ]
        ),
        dcc.Interval(id="tick-interval", interval=SPEED_PRESETS[DEFAULT_SPEED_KEY]["interval_ms"], disabled=True),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _status_children() -> list:
    state = "finished" if session.finished else ("running" if session.running else "paused")
    state_color = {"running": "success", "paused": "secondary", "finished": "info"}[state]
    rate = session.steps_per_second()
    rate_text = f"{rate:.1f} h/s (last batch, this process)" if rate else "n/a"
    return [
        dbc.Badge(state.upper(), color=state_color, className="me-2"),
        html.Span(f"hour {session.model.current_hour} / {session.model.horizon_hours}  |  ", className="me-1"),
        html.Span(f"k = {session.k:g}  |  ", className="me-1"),
        html.Span(f"ablation = {session.ablation}  |  ", className="me-1"),
        html.Span(f"broker_count = {session.broker_count}  |  ", className="me-1"),
        html.Span(f"window = {WINDOW_HOURS} h  |  ", className="me-1"),
        html.Span(f"speed = {rate_text}"),
    ]


def _deferred_children() -> list:
    active = session.deferred_last_step_kwh > 0.0
    return [
        html.H5("Deferred-energy channel (D7)", className="card-title"),
        dbc.Badge(
            "DEFERRING NOW" if active else "idle this hour",
            color="warning" if active else "secondary",
            className="mb-2",
        ),
        html.Div(f"this hour: {session.deferred_last_step_kwh:.3f} kWh deferred"),
        html.Div(f"cumulative over the run: {session.deferred_total_kwh:.1f} kWh"),
    ]


def _metrics_children() -> list:
    metrics = session.metrics
    names = session.broker_names
    share_rows = [
        html.Li(f"{names.get(broker_id, broker_id)}: {share:.1%}") for broker_id, share in metrics.broker_load_share.items()
    ]
    # Metric 1 is a running cumulative total, not an annual rate: it keeps
    # growing as the run advances and only equals the thesis's EUR/year figure
    # once the full 8760-hour horizon has actually been simulated. Labelling a
    # partial run "EUR/year" would put a number on screen (for example about
    # 189 at hour 2000) that an examiner could read against Chapter 6's 633.93
    # EUR/year for this same cell and conclude the demo contradicts the thesis.
    # So the unit is only claimed once it is earned. avg_cost_per_kwh_eur is a
    # ratio and is meaningful at any point, so it is shown throughout.
    run_complete = session.model.current_hour >= session.model.horizon_hours
    if run_complete:
        cost_text = f"{metrics.avg_cost_per_agent_eur:.2f} EUR/year  ({metrics.avg_cost_per_kwh_eur:.4f} EUR/kWh)"
    else:
        cost_text = (
            f"{metrics.avg_cost_per_agent_eur:.2f} EUR cumulative over "
            f"{session.model.current_hour} h of {session.model.horizon_hours} "
            f"({metrics.avg_cost_per_kwh_eur:.4f} EUR/kWh)"
        )
    return [
        html.H5("Four thesis metrics (live)", className="card-title"),
        html.Div([html.B("1. avg cost/agent: "), cost_text]),
        html.Div([html.B("2. load distribution: "), html.Ul(share_rows, className="mb-1"), f"HHI = {metrics.load_concentration_hhi:.3f}"]),
        html.Div(
            [
                html.B("3. grid stability: "),
                f"CoV = {metrics.feeder_coefficient_of_variation:.3f}, ",
                f"peak/avg = {metrics.feeder_peak_to_average_ratio:.3f}, ",
                f"mean ramp = {metrics.feeder_mean_hourly_ramp_kwh:.2f} kWh",
            ]
        ),
        html.Div([html.B("4. prosumer self-sufficiency: "), f"{metrics.prosumer_self_sufficiency:.1%}"]),
        html.Hr(),
        html.Div(
            className="text-muted small",
            children=[
                f"capacity charge so far: {metrics.total_capacity_charge_eur:.2f} EUR, ",
                f"fire rate: {metrics.capacity_fire_rate:.1%} of hours elapsed, ",
                f"total deferred so far: {metrics.total_deferred_kwh:.1f} kWh",
            ],
        ),
    ]


def _render():
    return (
        figures.feeder_figure(session),
        figures.broker_share_figure(session),
        figures.surcharge_figure(session),
        _metrics_children(),
        _deferred_children(),
        _status_children(),
    )


# ---------------------------------------------------------------------------
# the one callback that owns every state transition
# ---------------------------------------------------------------------------

_RESET_TRIGGERS = {"k-slider", "ablation-dropdown", "broker-count-dropdown", "btn-reset"}


@app.callback(
    Output("feeder-graph", "figure"),
    Output("broker-share-graph", "figure"),
    Output("surcharge-graph", "figure"),
    Output("metrics-panel", "children"),
    Output("deferred-indicator", "children"),
    Output("status-bar", "children"),
    Output("tick-interval", "disabled"),
    Output("tick-interval", "interval"),
    Input("tick-interval", "n_intervals"),
    Input("btn-start", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-step", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("k-slider", "value"),
    Input("ablation-dropdown", "value"),
    Input("broker-count-dropdown", "value"),
    Input("speed-dropdown", "value"),
)
def on_update(_n_intervals, _start, _pause, _step, _reset, k_value, ablation_value, broker_count_value, speed_key):
    triggered = dash.ctx.triggered_id
    speed = SPEED_PRESETS.get(speed_key or DEFAULT_SPEED_KEY, SPEED_PRESETS[DEFAULT_SPEED_KEY])

    with _session_lock:
        if triggered in _RESET_TRIGGERS:
            session.reset(k_value, ablation_value, broker_count_value)
        elif triggered == "btn-start":
            if not session.finished:
                session.running = True
        elif triggered == "btn-pause":
            session.running = False
        elif triggered == "btn-step":
            session.running = False
            session.step_batch(1)
        elif triggered == "tick-interval":
            if session.running and not session.finished:
                session.step_batch(speed["steps_per_tick"])
            if session.finished:
                session.running = False
        # triggered is None (initial page load) or "speed-dropdown": no
        # session mutation, just re-render with the current speed applied.

        interval_disabled = (not session.running) or session.finished
        outputs = _render()

    return (*outputs, interval_disabled, speed["interval_ms"])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _find_free_port(preferred: int = 8050, tries: int = 10) -> int:
    for offset in range(tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def main() -> None:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"[demo] MicrogridModel loaded, horizon={HORIZON_HOURS}h, window={WINDOW_HOURS}h")
    print(f"[demo] starting server at {url}")
    if port != 8050:
        print("[demo] port 8050 was busy; using the next free port instead.")

    threading.Timer(1.2, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
