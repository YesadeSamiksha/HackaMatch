"""
mapping.py - Interactive Geospatial Visualization Engine for CityPulse
Renders verified Los Angeles County highway sensor networks on Plotly Mapbox.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_MAP = {
    "Low": "#10B981",       # Emerald Green
    "Moderate": "#F59E0B",  # Amber Yellow
    "High": "#EF4444"       # Crimson Red
}

def create_sensor_map(
    network_df: pd.DataFrame, 
    selected_node_id: int = None,
    color_by: str = "pred_condition",
    map_style: str = "carto-positron",
    height: int = 580
) -> go.Figure:
    """
    Renders an interactive Mapbox map centered on Los Angeles County,
    displaying all 207 verified loop detector sensor stations.
    """
    df = network_df.copy()
    
    # Ensure latitude and longitude exist
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Network dataframe must contain 'latitude' and 'longitude' coordinates.")
        
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    
    # Custom hover text
    hover_texts = []
    for _, row in df.iterrows():
        node_id = row.get("node_id", "")
        sensor_id = row.get("sensor_id", node_id)
        corridor = row.get("corridor", "Corridor")
        curr_s = row.get("current_speed", row.get("x_t+0_d0", 0.0))
        pred_s = row.get("pred_speed_15m", curr_s)
        cond = row.get("pred_condition", "Low")
        conf = row.get("confidence_score", 0.0)
        p_tier = row.get("priority_tier", "Normal")
        
        txt = (
            f"<b>Sensor Station #{sensor_id}</b> (Node {node_id})<br>"
            f"<b>Corridor:</b> {corridor}<br>"
            f"----------------------------------------<br>"
            f"<b>Current Speed:</b> {curr_s:.1f} mph<br>"
            f"<b>Forecast (t+15m):</b> {pred_s:.1f} mph<br>"
            f"<b>Predicted State:</b> {cond} ({conf:.0f}% conf)<br>"
            f"<b>Operational Priority:</b> {p_tier}"
        )
        hover_texts.append(txt)
    df["hover_info"] = hover_texts
    
    fig = go.Figure()
    
    # Group by condition to get clean legends
    conditions = ["High", "Moderate", "Low"]
    labels = {
        "High": "High Congestion (< 35 mph)",
        "Moderate": "Moderate Congestion (35-55 mph)",
        "Low": "Free-Flow (>= 55 mph)"
    }
    
    for cond in conditions:
        subset = df[df["pred_condition"] == cond]
        if not subset.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=subset["latitude"],
                    lon=subset["longitude"],
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=subset.get("priority_score", pd.Series(12, index=subset.index)).clip(10, 24),
                        color=COLOR_MAP.get(cond, "#3B82F6"),
                        opacity=0.88
                    ),
                    name=labels.get(cond, cond),
                    hovertext=subset["hover_info"],
                    hoverinfo="text",
                    customdata=subset["node_id"]
                )
            )
            
    # Highlight selected node if provided
    if selected_node_id is not None:
        sel_row = df[df["node_id"] == int(selected_node_id)]
        if not sel_row.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=sel_row["latitude"],
                    lon=sel_row["longitude"],
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=28,
                        color="#8B5CF6", # Bright Purple halo
                        opacity=0.95,
                        symbol="circle"
                    ),
                    name=f"Selected Node #{selected_node_id}",
                    hovertext=sel_row["hover_info"],
                    hoverinfo="text",
                    showlegend=True
                )
            )
            
    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.2
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig
