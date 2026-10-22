import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import (
    COLOR_PRIMARY_CYAN, COLOR_SECONDARY_BLUE, COLOR_ACCENT_AMBER,
    COLOR_SUCCESS_EMERALD, COLOR_DANGER_ROSE, COLOR_PURPLE,
    COLOR_BORDER, COLOR_TEXT_LIGHT
)

def apply_dark_theme(fig, height=350):
    fig.update_layout(
        paper_bgcolor="rgba(17, 24, 39, 0.7)",
        plot_bgcolor="rgba(11, 15, 25, 0.8)",
        font=dict(family="sans-serif", color=COLOR_TEXT_LIGHT, size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height
    )
    fig.update_xaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    return fig

def plot_top_divisions_bar(df: pd.DataFrame, n: int = 10):
    if df is None or df.empty:
        return None
    sub_df = df.head(n).copy()
    sub_df["short_name"] = sub_df["division"].str.replace(" Police Station", "").str.replace(" Division", "")

    fig = px.bar(
        sub_df,
        x="total_challan",
        y="short_name",
        orientation="h",
        text_auto=".2s",
        color="total_challan",
        color_continuous_scale=[[0, "#0891B2"], [1, COLOR_PRIMARY_CYAN]],
        labels={"total_challan": "Recorded Challans", "short_name": "Traffic Division"}
    )
    apply_dark_theme(fig, height=380)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside", marker_line_color=COLOR_PRIMARY_CYAN, marker_line_width=1)
    return fig

def plot_violation_categories_bar(df: pd.DataFrame, n: int = 10):
    if df is None or df.empty:
        return None
    sub_df = df.head(n).copy()

    fig = px.bar(
        sub_df,
        x="category",
        y="total_challans",
        text_auto=".2s",
        color="total_challans",
        color_continuous_scale=[[0, "#D97706"], [1, COLOR_ACCENT_AMBER]],
        labels={"total_challans": "Recorded Infractions", "category": "Violation Category"}
    )
    apply_dark_theme(fig, height=380)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_xaxes(tickangle=-25)
    fig.update_traces(textposition="outside", marker_line_color=COLOR_ACCENT_AMBER, marker_line_width=1)
    return fig

def plot_rainfall_timeline(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Daily Actual"],
        name="Daily Actual (mm)",
        marker_color=COLOR_SECONDARY_BLUE
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Daily Normal"],
        name="Daily Normal (mm)",
        mode="lines",
        line=dict(color="#94A3B8", width=1.5, dash="dash")
    ))
    apply_dark_theme(fig, height=350)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_cumulative_rainfall(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Cumulative Actual"],
        name="Cumulative Actual (mm)",
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.15)",
        line=dict(color=COLOR_PRIMARY_CYAN, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Cumulative Normal"],
        name="Cumulative Normal (mm)",
        mode="lines",
        line=dict(color="#94A3B8", width=1.5, dash="dot")
    ))
    apply_dark_theme(fig, height=300)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_rainfall_departure(df: pd.DataFrame):
    if df is None or df.empty or "Daily Departure Per_num" not in df.columns:
        return None

    dep_df = df.dropna(subset=["Daily Departure Per_num"]).copy()
    colors = [COLOR_SUCCESS_EMERALD if v >= 0 else COLOR_ACCENT_AMBER for v in dep_df["Daily Departure Per_num"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dep_df["Date"],
        y=dep_df["Daily Departure Per_num"],
        name="Departure %",
        marker_color=colors
    ))
    apply_dark_theme(fig, height=300)
    fig.update_yaxes(ticksuffix="%")
    return fig

def plot_traffic_speed_by_area(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    fig = px.bar(
        df,
        x="Area",
        y="Avg_Speed_kmph",
        text_auto=".1f",
        color="Avg_Speed_kmph",
        color_continuous_scale=[[0, COLOR_DANGER_ROSE], [0.5, COLOR_ACCENT_AMBER], [1, COLOR_SUCCESS_EMERALD]],
        labels={"Avg_Speed_kmph": "Avg Speed (km/h)", "Area": "Mumbai Area"}
    )
    apply_dark_theme(fig, height=320)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(textposition="outside")
    return fig

def plot_congestion_donut(df: pd.DataFrame):
    if df is None or df.empty or "Congestion_Level" not in df.columns:
        return None

    counts = df["Congestion_Level"].value_counts().reset_index()
    counts.columns = ["Congestion_Level", "Count"]

    color_map = {
        "Low": COLOR_SUCCESS_EMERALD,
        "Medium": COLOR_ACCENT_AMBER,
        "High": COLOR_DANGER_ROSE
    }

    fig = px.pie(
        counts,
        names="Congestion_Level",
        values="Count",
        hole=0.55,
        color="Congestion_Level",
        color_discrete_map=color_map
    )
    apply_dark_theme(fig, height=320)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def plot_vehicle_enforcement_bar(df: pd.DataFrame, n: int = 10):
    if df is None or df.empty:
        return None
    sub_df = df.head(n).copy()

    fig = px.bar(
        sub_df,
        x="vehicle_type",
        y="total_challan",
        text_auto=".2s",
        color="total_challan",
        color_continuous_scale=[[0, "#7E22CE"], [1, COLOR_PURPLE]],
        labels={"total_challan": "Recorded Challans", "vehicle_type": "Vehicle Type"}
    )
    apply_dark_theme(fig, height=360)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_xaxes(tickangle=-25)
    fig.update_traces(textposition="outside", marker_line_color=COLOR_PURPLE, marker_line_width=1)
    return fig
