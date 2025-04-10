import streamlit as st
import folium
import matplotlib.pyplot as plt
import numpy as np
import polyline
import pydeck as pdk
import pandas as pd
from streamlit_folium import st_folium
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from math import sin, cos, sqrt, atan2, radians
import plotly.graph_objects as go
import plotly.express as px

# Constants
DEFAULT_MAP_STYLE = "mapbox://styles/mapbox/light-v10"
ELEVATION_SCALE = 5  # Vertical exaggeration for 3D

@dataclass
class MapConfig:
    """Configuration for map visualizations."""
    width: int = 800
    height: int = 600
    zoom: int = 12
    map_style: str = DEFAULT_MAP_STYLE
    terrain: bool = True
    show_traffic: bool = True

# ----------------- 2D Route Visualization -----------------

def plot_route(route_data: Dict, 
               start_coords: Tuple[float, float], 
               end_coords: Tuple[float, float], 
               config: Optional[MapConfig] = None) -> folium.Map:
    """
    Generate an interactive 2D route map using Folium with traffic overlay and elevation profile.

    Args:
        route_data: GeoJSON route data containing coordinates and properties.
        start_coords: Starting coordinate as (latitude, longitude).
        end_coords: Destination coordinate as (latitude, longitude).
        config: Optional MapConfig object for visualization settings (defaults to MapConfig()).

    Returns:
        folium.Map object with the route, markers, traffic overlay, and elevation profile.
    """
    config = config or MapConfig()
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lng = (start_coords[1] + end_coords[1]) / 2

    m = folium.Map(
        location=[mid_lat, mid_lng],
        zoom_start=config.zoom,
        width=config.width,
        height=config.height,
        tiles="cartodbpositron",
        control_scale=True
    )

    if route_data and "features" in route_data and route_data["features"]:
        try:
            feature = route_data["features"][0]
            coordinates = feature["geometry"]["coordinates"]
            # Convert from [lng, lat] to (lat, lon) for Folium
            decoded_points = [(lat, lon) for lon, lat in coordinates]

            # Determine route color based on traffic data if available
            if "properties" in feature and "traffic_data" in feature["properties"]:
                traffic_delay = feature["properties"]["traffic_data"].get("trafficDelay", 0)
                if traffic_delay > 300:  # 5 minutes
                    color = "red"
                    st.write("Traffic: Heavy")
                elif traffic_delay > 60:
                    color = "yellow"
                    st.write("Traffic: Moderate")
                else:
                    color = "green"
                    st.write("Traffic: Light")
            else:
                color = "#4285F4"  # Google Maps blue

            # Add main route polyline
            folium.PolyLine(
                decoded_points,
                color=color,
                weight=6,
                opacity=0.8,
                tooltip="Route",
                popup=f"Distance: {feature['properties']['summary']['distance']} m" if "summary" in feature["properties"] else "Route"
            ).add_to(m)

            # Simulate traffic by coloring a middle segment red
            if len(decoded_points) > 10:
                traffic_start = int(len(decoded_points) * 0.4)
                traffic_end = int(len(decoded_points) * 0.6)
                traffic_segment = decoded_points[traffic_start:traffic_end]
                folium.PolyLine(
                    traffic_segment,
                    color="red",
                    weight=6,
                    opacity=0.8,
                    tooltip="Heavy Traffic"
                ).add_to(m)

            # Add elevation profile if available
            if "elevation" in feature["properties"]:
                _add_elevation_profile(m, feature["properties"]["elevation"], decoded_points)

        except Exception as e:
            st.error(f"Route visualization error: {str(e)}")

    # Add start and destination markers with custom icons
    folium.Marker(
        start_coords,
        popup="Start Point",
        icon=folium.Icon(color="green", icon="rocket", prefix="fa", icon_color="white")
    ).add_to(m)
    folium.Marker(
        end_coords,
        popup="Destination",
        icon=folium.Icon(color="red", icon="map-pin", prefix="fa", icon_color="white")
    ).add_to(m)

    # Add live location marker if available in session state
    if "current_location" in st.session_state and st.session_state["current_location"]:
        loc = st.session_state["current_location"]
        folium.Marker(
            [loc[0], loc[1]],  # Use tuple indices instead of dictionary keys
            popup="📍 You Are Here",
            icon=folium.Icon(color="blue", icon="user", prefix="fa")
        ).add_to(m)

    return m

# ----------------- 3D Route Visualization -----------------

def plot_3d_route(route_data: Dict, 
                  start_coords: Tuple[float, float], 
                  end_coords: Tuple[float, float], 
                  config: Optional[MapConfig] = None) -> Optional[pdk.Deck]:
    """
    Generate a 3D terrain-aware route visualization using PyDeck.

    Args:
        route_data: GeoJSON route data.
        start_coords: Starting coordinate as (latitude, longitude).
        end_coords: Destination coordinate as (latitude, longitude).
        config: Optional MapConfig object for visualization settings.

    Returns:
        pdk.Deck object for 3D visualization or None if failed.
    """
    config = config or MapConfig()
    if not route_data or "features" not in route_data or not route_data["features"]:
        return None

    try:
        feature = route_data["features"][0]
        coordinates = feature["geometry"]["coordinates"]
        # Convert coordinates to DataFrame (PyDeck uses [lng, lat])
        df = pd.DataFrame(coordinates, columns=["lng", "lat"])
        elevation = feature["properties"].get("elevation", [0] * len(df))
        df["elevation"] = elevation

        # Path layer for the route
        path_layer = pdk.Layer(
            "PathLayer",
            data=df,
            get_path=["lng", "lat"],
            get_color=[255, 0, 0],  # Red
            width_scale=20,
            width_min_pixels=2,
            get_width=5,
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            get_elevation="elevation"
        )

        # Scatter layer for route points
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lng", "lat"],
            get_elevation="elevation",
            get_radius=100,
            get_fill_color=[255, 140, 0],  # Orange
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            extruded=True
        )

        # Set view state
        view_state = pdk.ViewState(
            longitude=df["lng"].mean(),
            latitude=df["lat"].mean(),
            zoom=config.zoom,
            pitch=45,
            bearing=0
        )

        return pdk.Deck(
            layers=[path_layer, scatter_layer],
            initial_view_state=view_state,
            map_style=config.map_style,
            tooltip={"html": "<b>Elevation:</b> {elevation} m", "style": {"color": "white"}}
        )
    except Exception as e:
        st.error(f"3D visualization failed: {str(e)}")
        return None

# ----------------- Additional Visualization Functions -----------------

def plot_3d_route_comparison(main_route: Dict, alt_route: Dict, 
                             start_coords: Tuple[float, float], 
                             end_coords: Tuple[float, float],
                             config: Optional[MapConfig] = None) -> Optional[pdk.Deck]:
    """
    Generate a 3D visualization comparing two routes (e.g., fastest vs shortest).

    Args:
        main_route: GeoJSON data for the main route.
        alt_route: GeoJSON data for the alternate route.
        start_coords: Starting coordinate as (latitude, longitude).
        end_coords: Destination coordinate as (latitude, longitude).
        config: Optional MapConfig for visualization settings.

    Returns:
        pdk.Deck object with both routes overlaid or None if failed.
    """
    config = config or MapConfig()
    try:
        # Main route
        main_feature = main_route["features"][0]
        main_coords = main_feature["geometry"]["coordinates"]
        df_main = pd.DataFrame(main_coords, columns=["lng", "lat"])
        df_main["elevation"] = main_feature["properties"].get("elevation", [0] * len(df_main))

        # Alternate route
        alt_feature = alt_route["features"][0]
        alt_coords = alt_feature["geometry"]["coordinates"]
        df_alt = pd.DataFrame(alt_coords, columns=["lng", "lat"])
        df_alt["elevation"] = alt_feature["properties"].get("elevation", [0] * len(df_alt))

        # Layers for main route (red)
        main_layer = pdk.Layer(
            "PathLayer",
            data=df_main,
            get_path=["lng", "lat"],
            get_color=[255, 0, 0],
            width_scale=20,
            width_min_pixels=2,
            get_width=5,
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            get_elevation="elevation"
        )
        main_scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_main,
            get_position=["lng", "lat"],
            get_elevation="elevation",
            get_radius=100,
            get_fill_color=[255, 0, 0],
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            extruded=True
        )

        # Layers for alternate route (blue)
        alt_layer = pdk.Layer(
            "PathLayer",
            data=df_alt,
            get_path=["lng", "lat"],
            get_color=[0, 0, 255],
            width_scale=20,
            width_min_pixels=2,
            get_width=5,
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            get_elevation="elevation"
        )
        alt_scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_alt,
            get_position=["lng", "lat"],
            get_elevation="elevation",
            get_radius=100,
            get_fill_color=[0, 0, 255],
            pickable=True,
            elevation_scale=ELEVATION_SCALE,
            extruded=True
        )

        # Combined view state
        combined_df = pd.concat([df_main, df_alt])
        view_state = pdk.ViewState(
            longitude=combined_df["lng"].mean(),
            latitude=combined_df["lat"].mean(),
            zoom=config.zoom,
            pitch=45,
            bearing=0
        )

        return pdk.Deck(
            layers=[main_layer, alt_layer, main_scatter, alt_scatter],
            initial_view_state=view_state,
            map_style=config.map_style,
            tooltip={"html": "<b>Elevation:</b> {elevation} m", "style": {"color": "white"}}
        )
    except Exception as e:
        st.error(f"3D route comparison visualization failed: {str(e)}")
        return None

def plot_combined_metrics(results: Dict[str, Dict[str, float]]) -> None:
    """
    Generate a combined bar chart for algorithm performance metrics.

    Args:
        results: Dictionary with metrics (e.g., Time, Efficiency, Accuracy) for each algorithm.
    """
    if not results:
        st.warning("No results to visualize")
        return

    fig = go.Figure()
    metrics = ["Time", "Efficiency", "Accuracy"]
    colors = px.colors.qualitative.Plotly

    for i, (algo, data) in enumerate(results.items()):
        fig.add_trace(go.Bar(
            x=metrics,
            y=[data[m] for m in metrics],
            name=algo,
            marker_color=colors[i % len(colors)],
            text=[f"{data[m]:.2f}" for m in metrics],
            textposition="auto"
        ))

    fig.update_layout(
        title="Algorithm Performance Comparison",
        barmode="group",
        xaxis_title="Metrics",
        yaxis_title="Value",
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Utility Functions -------------------

def _haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate the great-circle distance between two coordinates in kilometers."""
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))

def _fig_to_html(fig: plt.Figure) -> str:
    """Convert a Matplotlib figure to an HTML image tag."""
    from io import BytesIO
    import base64
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}">'

def _add_elevation_profile(m: folium.Map, elevation: List[float], points: List[Tuple[float, float]]) -> None:
    """Add an elevation profile below the map using Matplotlib."""
    distances = [0]
    for i in range(1, len(points)):
        distances.append(distances[-1] + _haversine(points[i-1], points[i]))

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(distances, elevation, color="#4285F4")
    ax.fill_between(distances, elevation, alpha=0.3)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    plt.tight_layout()

    html = _fig_to_html(fig)
    plt.close(fig)
    folium.Marker(
        location=[points[0][0], points[0][1]],  # Placed at start point
        icon=folium.DivIcon(html=html)
    ).add_to(m)

def _add_traffic_layer(m: folium.Map, traffic_data: Dict) -> None:
    """Add a traffic layer to the map (placeholder for actual implementation)."""
    # Placeholder; implement based on actual traffic data format if available
    pass