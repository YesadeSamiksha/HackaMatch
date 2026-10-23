"""
mapping.py - Interactive Geospatial Visualization Engine for CityPulse
Renders verified Los Angeles County highway sensor networks on Plotly Mapbox.
Supports Mapbox tokens when provided, with high-fidelity OpenStreetMap fallback.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

COLOR_MAP = {
    "Low": "#10B981",       # Emerald Green: Free-Flow (>= 55 mph)
    "Moderate": "#F59E0B",  # Amber Yellow: Moderate Congestion (35-55 mph)
    "High": "#EF4444"       # Crimson Red: High Congestion (< 35 mph)
}

def get_mapbox_token() -> str | None:
    """
    Safely resolves Mapbox token from environment or Streamlit secrets without exposing keys.
    """
    token = (
        os.getenv("MAPBOX_API_TOKEN") or 
        os.getenv("MAPBOX_ACCESS_TOKEN") or 
        os.getenv("MAPBOX_TOKEN")
    )
    if not token:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                token = st.secrets.get("MAPBOX_API_TOKEN") or st.secrets.get("MAPBOX_ACCESS_TOKEN")
        except Exception:
            token = None
    return token

def create_sensor_map(
    network_df: pd.DataFrame, 
    selected_node_id: int = None,
    color_by: str = "pred_condition",
    map_style: str = "open-street-map",
    height: int = 540
) -> go.Figure:
    """
    Renders an interactive Mapbox map centered on Los Angeles County,
    displaying verified loop detector sensor stations with accessible contrast,
    real coordinates, severity coloring, and robust no-token basemap fallback.
    """
    try:
        df = network_df.copy()
        
        # Ensure latitude and longitude exist
        if "latitude" not in df.columns or "longitude" not in df.columns:
            raise ValueError("Network dataframe must contain 'latitude' and 'longitude' coordinates.")
            
        # Calculate geometric midpoint bounds to center the map precisely
        min_lat, max_lat = float(df["latitude"].min()), float(df["latitude"].max())
        min_lon, max_lon = float(df["longitude"].min()), float(df["longitude"].max())
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        
        # Determine appropriate zoom level based on coordinate span
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        if lon_span > 0.25 or lat_span > 0.15:
            # Full 207-node Los Angeles highway network framing
            initial_zoom = 9.85
        elif lon_span > 0.08 or lat_span > 0.05:
            initial_zoom = 10.6
        else:
            initial_zoom = 11.5
            
        # Check map token and normalize style
        mapbox_token = get_mapbox_token()
        
        # If Carto styles are requested without an explicit Carto key, avoid watermarks
        carto_key = os.getenv("CARTO_API_KEY")
        if ("carto" in map_style.lower()) and not carto_key:
            resolved_style = "open-street-map"
        elif mapbox_token:
            resolved_style = map_style
        else:
            # Safe, reliable, attributed no-token basemap with real roads and labels
            resolved_style = "open-street-map" if map_style in ["carto-positron", "carto-darkmatter", "open-street-map"] else "open-street-map"

        # Build accessible, high-contrast hover text adhering to WCAG AA guidelines
        hover_texts = []
        for _, row in df.iterrows():
            node_id = row.get("node_id", "")
            sensor_id = row.get("sensor_id", node_id)
            corridor = row.get("corridor", "Corridor Identifier Unavailable")
            
            curr_s = row.get("current_speed", row.get("x_t+0_d0", None))
            curr_s_str = f"{float(curr_s):.1f} mph" if pd.notnull(curr_s) else "Unavailable"
            
            pred_s = row.get("pred_speed_15m", None)
            pred_s_str = f"{float(pred_s):.1f} mph" if pd.notnull(pred_s) else "Unavailable"
            
            cond = row.get("pred_condition", "Unavailable")
            
            conf = row.get("confidence_score", None)
            conf_str = f"{float(conf):.1f}%" if pd.notnull(conf) and float(conf) > 0 else "Unavailable"
            
            # Operational priority derived from severity_tier / priority_score
            p_tier = row.get("severity_tier", row.get("priority_tier", "Unavailable"))
            p_score = row.get("priority_score", None)
            p_str = f"{p_tier} ({float(p_score):.1f}/100)" if pd.notnull(p_score) else f"{p_tier}"
            
            txt = (
                f"<b>Loop Station #{sensor_id}</b> (Node {node_id})<br>"
                f"<b>Corridor:</b> {corridor}<br>"
                f"----------------------------------------<br>"
                f"<b>Current Speed:</b> {curr_s_str}<br>"
                f"<b>Forecast Speed (t+15m):</b> {pred_s_str}<br>"
                f"<b>Predicted Category:</b> {cond}<br>"
                f"<b>Model Confidence:</b> {conf_str}<br>"
                f"<b>Operational Priority:</b> {p_str}"
            )
            hover_texts.append(txt)
        df["hover_info"] = hover_texts
        
        fig = go.Figure()
        
        # Order traces by condition so all categories render consistently
        conditions = ["High", "Moderate", "Low"]
        labels = {
            "High": "High Congestion (< 35 mph)",
            "Moderate": "Moderate Congestion (35-55 mph)",
            "Low": "Free-Flow (>= 55 mph)"
        }
        
        for cond in conditions:
            subset = df[df["pred_condition"] == cond]
            if not subset.empty:
                # Dynamic marker sizes based on priority score, clipped comfortably
                marker_sizes = (
                    subset["priority_score"].clip(10, 22) 
                    if "priority_score" in subset.columns 
                    else pd.Series(13, index=subset.index)
                )
                fig.add_trace(
                    go.Scattermapbox(
                        lat=subset["latitude"],
                        lon=subset["longitude"],
                        mode="markers",
                        marker=go.scattermapbox.Marker(
                            size=marker_sizes,
                            color=COLOR_MAP.get(cond, "#38BDF8"),
                            opacity=0.90
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
                sensor_label = sel_row.iloc[0].get("sensor_id", selected_node_id)
                fig.add_trace(
                    go.Scattermapbox(
                        lat=sel_row["latitude"],
                        lon=sel_row["longitude"],
                        mode="markers",
                        marker=go.scattermapbox.Marker(
                            size=26,
                            color="#8B5CF6", # Bright Purple halo
                            opacity=0.95
                        ),
                        name=f"Selected Station #{sensor_label}",
                        hovertext=sel_row["hover_info"],
                        hoverinfo="text",
                        showlegend=True
                    )
                )
                
        # Configure mapbox layout with dark theme & accessible contrast
        mapbox_cfg = dict(
            style=resolved_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=initial_zoom
        )
        if mapbox_token:
            mapbox_cfg["accesstoken"] = mapbox_token
            
        fig.update_layout(
            mapbox=mapbox_cfg,
            autosize=True,
            margin=dict(l=0, r=0, t=10, b=0),
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(15, 23, 42, 0.90)",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(color="#F1F5F9", size=11, family="Inter, system-ui, sans-serif")
            ),
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 42, 0.96)",
                bordercolor="#38BDF8",
                font=dict(
                    color="#FFFFFF",
                    size=12,
                    family="Inter, system-ui, -apple-system, sans-serif"
                ),
                align="left"
            )
        )
        return fig

    except Exception as e:
        # Actionable Fallback figure if map fails to load
        fallback_fig = go.Figure()
        if "latitude" in network_df.columns and "longitude" in network_df.columns:
            for cond in ["High", "Moderate", "Low"]:
                sub = network_df[network_df.get("pred_condition", "Low") == cond]
                if not sub.empty:
                    fallback_fig.add_trace(
                        go.Scatter(
                            x=sub["longitude"],
                            y=sub["latitude"],
                            mode="markers",
                            name=f"{cond} Congestion",
                            marker=dict(size=10, color=COLOR_MAP.get(cond, "#38BDF8"))
                        )
                    )
        fallback_fig.update_layout(
            title=dict(
                text=f"Geographic Basemap Fallback (Coordinate Plot Mode) — {str(e)}",
                font=dict(color="#F59E0B", size=13)
            ),
            xaxis=dict(title="Longitude", gridcolor="#1E293B"),
            yaxis=dict(title="Latitude", gridcolor="#1E293B"),
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#111827",
            height=height
        )
        return fallback_fig
