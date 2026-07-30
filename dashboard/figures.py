"""Plotly figure builders for the live demo. Every function here only reads
values already sitting in a DemoSession (model_state.py); nothing here
computes broker prices, agent decisions, or capacity-mechanism arithmetic.
"""

from __future__ import annotations

import plotly.graph_objects as go

from model_state import broker_color

_PAPER_BG = "rgba(0,0,0,0)"
_GRID_COLOR = "rgba(128,128,128,0.25)"

_AXIS_STYLE = dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR)


def _empty_layout(title: str, height: int = 320) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PAPER_BG,
        xaxis=dict(title="hour", **_AXIS_STYLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


def feeder_figure(session) -> go.Figure:
    hours = list(session.hours)
    feeder = list(session.feeder)
    threshold = list(session.threshold)
    scarcity = list(session.scarcity)

    fig = go.Figure(layout=_empty_layout("Feeder net import, rolling threshold, scarcity hours"))
    fig.update_yaxes(title="feeder net import (kWh)", **_AXIS_STYLE)

    fig.add_trace(
        go.Scatter(
            x=hours,
            y=feeder,
            mode="lines",
            name="feeder net import",
            line=dict(color="#4C72B0", width=1.6),
        )
    )

    if any(value is not None for value in threshold):
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=threshold,
                mode="lines",
                name=f"threshold (mean + k*std, k={session.k:g})",
                line=dict(color="#C44E52", width=1.4, dash="dot"),
                connectgaps=False,
            )
        )

    scarcity_x = [hour for hour, flag in zip(hours, scarcity) if flag]
    scarcity_y = [value for value, flag in zip(feeder, scarcity) if flag]
    if scarcity_x:
        fig.add_trace(
            go.Scatter(
                x=scarcity_x,
                y=scarcity_y,
                mode="markers",
                name="scarcity hour (charge fired)",
                marker=dict(color="#C44E52", size=8, symbol="x", line=dict(width=1.5)),
            )
        )

    if not hours:
        fig.add_annotation(text="press Start or Step", showarrow=False, font=dict(size=13))
    return fig


def broker_share_figure(session) -> go.Figure:
    hours = list(session.hours)
    names = session.broker_names

    fig = go.Figure(layout=_empty_layout("Broker load share (this hour's positive net import, by broker)"))
    fig.update_yaxes(title="share of feeder load", range=[0, 1], **_AXIS_STYLE)

    for index, (broker_id, series) in enumerate(session.broker_share_history.items()):
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=list(series),
                mode="lines",
                name=names.get(broker_id, broker_id),
                stackgroup="one",
                line=dict(width=0.5, color=broker_color(broker_id, index)),
                fillcolor=broker_color(broker_id, index),
            )
        )

    if not hours:
        fig.add_annotation(text="press Start or Step", showarrow=False, font=dict(size=13))
    return fig


def surcharge_figure(session) -> go.Figure:
    names = session.broker_names
    broker_ids = list(session.surcharge_current.keys())
    labels = [names.get(broker_id, broker_id) for broker_id in broker_ids]
    values = [session.surcharge_current[broker_id] for broker_id in broker_ids]
    colors = [broker_color(broker_id, index) for index, broker_id in enumerate(broker_ids)]

    fig = go.Figure(layout=_empty_layout("Current per-broker surcharge (EUR/kWh, next-step quote adder)", height=300))
    fig.update_layout(xaxis=dict(title=None, **_AXIS_STYLE))
    fig.update_yaxes(title="surcharge (EUR/kWh)", **_AXIS_STYLE)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
        )
    )
    max_value = max(values) if values else 0.0
    fig.update_yaxes(range=[0, max(max_value * 1.35, 0.001)])
    return fig
