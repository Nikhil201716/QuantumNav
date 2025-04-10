import requests
import streamlit as st
import openrouteservice as ors
import polyline
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
from geopy.distance import great_circle

from config import (
    ORS_API_KEY,
    TOMTOM_API_KEY,
    GOOGLE_MAPS_API_KEY,
    POI_CATEGORIES,
    DEFAULT_POI_RADIUS,
    CACHE_EXPIRY,
    TRAFFIC_CACHE_EXPIRY
)

# ---- Enums and DataClasses ----

class RoutingProfile(Enum):
    """Enumeration of supported routing profiles."""
    WALK = "foot-walking"
    CYCLE = "cycling-regular"
    BIKE = "cycling-electric"
    CAR = "driving-car"
    TRUCK = "driving-hgv"
    DRIVING_CAR = "driving-car"  # Included for compatibility

class TrafficModel(Enum):
    """Enumeration of traffic prediction models."""
    BEST_GUESS = auto()
    PESSIMISTIC = auto()
    OPTIMISTIC = auto()

@dataclass
class RouteSummary:
    """Summary of a route including traffic-aware data."""
    distance_km: float
    base_duration_hours: float
    traffic_duration_hours: float
    avg_speed_kmh: float
    congestion_percentage: float
    waypoints: int
    avoid_features: List[str]
    traffic_enabled: bool
    traffic_model: Optional[TrafficModel]

@dataclass
class TrafficInfo:
    """Detailed traffic information for a route segment."""
    speed: float
    congestion_level: int
    delay_seconds: int
    free_flow_speed: float
    historical_speed: float

@dataclass
class RouteInstruction:
    """Individual instruction for a route segment."""
    distance_km: float
    base_duration_min: float
    traffic_duration_min: float
    instruction: str
    step_type: str
    traffic: Optional[TrafficInfo]

# ---- Client Setup ----

ors_client = ors.Client(key=ORS_API_KEY)

# ---- Main Routing Logic ----

@st.cache_data(show_spinner="Calculating optimal route...", ttl=CACHE_EXPIRY)
def get_route(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    mode: str,
    api_key: str,
    traffic: bool = False,
    preference: str = "recommended"
) -> Optional[Dict]:
    """
    Fetch a route using ORS, TomTom, or Google Maps based on traffic preference.

    - If `traffic` is True: Uses TomTom with traffic data, falls back to Google Maps.
    - If `traffic` is False: Uses ORS with specified preference, falls back to TomTom (no traffic), then Google Maps.

    Args:
        start_coords (Tuple[float, float]): Starting coordinates (latitude, longitude).
        end_coords (Tuple[float, float]): Ending coordinates (latitude, longitude).
        mode (str): Routing profile (e.g., "driving-car").
        api_key (str): API key for ORS.
        traffic (bool): Enable traffic-aware routing. Defaults to False.
        preference (str): Route preference for ORS ("recommended" or "shortest"). Defaults to "recommended".

    Returns:
        Optional[Dict]: Route data in GeoJSON format, or None if all APIs fail.
    """
    if traffic:
        try:
            return _get_tomtom_route(start_coords, end_coords, mode, traffic=True)
        except Exception as e:
            st.warning(f"TomTom API with traffic failed: {e}, trying Google Maps...")
            try:
                return _get_google_route(start_coords, end_coords, mode)
            except Exception as e:
                st.error(f"Google Maps routing failed: {e}")
                return None
    else:
        try:
            valid_modes = [p.value for p in RoutingProfile]
            if mode not in valid_modes:
                raise ValueError(f"Unsupported mode: {mode}")

            # ORS expects coordinates in [longitude, latitude] order
            start = list(reversed(start_coords))
            end = list(reversed(end_coords))

            url = f"https://api.openrouteservice.org/v2/directions/{mode}"
            headers = {"Authorization": api_key, "Content-Type": "application/json"}
            body = {
                "coordinates": [start, end],
                "instructions": True,      # Ensure instructions are requested
                "preference": preference
            }
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"ORS API failed: {e}, trying TomTom without traffic...")
            try:
                return _get_tomtom_route(start_coords, end_coords, mode, traffic=False)
            except Exception as e:
                st.warning(f"TomTom API without traffic failed: {e}, trying Google Maps...")
                try:
                    return _get_google_route(start_coords, end_coords, mode)
                except Exception as e:
                    st.error(f"Google Maps routing failed: {e}")
                    return None

# ---- Alternate Routes Helper ----

def get_alternate_routes(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    mode: str,
    api_key: str
) -> Dict[str, Dict]:
    """
    Fetch both the fastest and shortest routes.

    Args:
        start_coords (Tuple[float, float]): Starting coordinates (latitude, longitude).
        end_coords (Tuple[float, float]): Ending coordinates (latitude, longitude).
        mode (str): Routing profile (e.g., "driving-car").
        api_key (str): API key for ORS.

    Returns:
        Dict[str, Dict]: Dictionary with "fastest" and "shortest" route data.
    """
    fastest_route = get_route(start_coords, end_coords, mode, api_key, traffic=True, preference="recommended")
    shortest_route = get_route(start_coords, end_coords, mode, api_key, traffic=False, preference="shortest")
    return {"fastest": fastest_route, "shortest": shortest_route}

# ---- TomTom Routing ----

def _get_tomtom_route(start: Tuple[float, float], end: Tuple[float, float], mode: str, traffic: bool) -> Dict:
    """
    Fetch a route from TomTom API with optional traffic data.

    Args:
        start (Tuple[float, float]): Starting coordinates (latitude, longitude).
        end (Tuple[float, float]): Ending coordinates (latitude, longitude).
        mode (str): Routing profile (e.g., "driving-car").
        traffic (bool): Include traffic data in the route.

    Returns:
        Dict: Route data converted to ORS-like GeoJSON format.
    """
    travel_mode = "truck" if mode == RoutingProfile.TRUCK.value else "car"
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start[0]},{start[1]}:{end[0]},{end[1]}/json"
    params = {
        "key": TOMTOM_API_KEY,
        "travelMode": travel_mode,
        "routeType": "fastest",
        "traffic": "true" if traffic else "false"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return _convert_tomtom_to_ors_format(r.json())

def _convert_tomtom_to_ors_format(data: Dict) -> Dict:
    """
    Convert TomTom route data to ORS-like GeoJSON format.

    Args:
        data (Dict): Raw route data from TomTom API.

    Returns:
        Dict: Route data in ORS-like GeoJSON format.
    """
    route = data["routes"][0]
    # Extract coordinates from each point in each leg
    coords = [[pt["longitude"], pt["latitude"]] for leg in route["legs"] for pt in leg["points"]]
    # For TomTom, steps might not be provided – leaving empty for now.
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "segments": [{
                    "distance": route["summary"]["lengthInMeters"],
                    "duration": route["summary"]["travelTimeInSeconds"],
                    "steps": []  # No instructions provided by default
                }],
                "summary": {
                    "distance": route["summary"]["lengthInMeters"],
                    "duration": route["summary"]["travelTimeInSeconds"]
                },
                "traffic_data": {
                    "trafficLength": route["summary"].get("trafficLengthInMeters", 0),
                    "trafficDelay": route["summary"].get("trafficDelayInSeconds", 0)
                } if "trafficDelayInSeconds" in route["summary"] else {}
            }
        }]
    }

# ---- Google Maps Fallback ----

def _get_google_route(start: Tuple[float, float], end: Tuple[float, float], mode: str) -> Dict:
    """
    Fetch a route from Google Maps API.

    Args:
        start (Tuple[float, float]): Starting coordinates (latitude, longitude).
        end (Tuple[float, float]): Ending coordinates (latitude, longitude).
        mode (str): Routing profile (e.g., "driving-car").

    Returns:
        Dict: Route data converted to ORS-like GeoJSON format.
    """
    gm_mode = {
        RoutingProfile.WALK.value: "walking",
        RoutingProfile.CYCLE.value: "bicycling",
        RoutingProfile.BIKE.value: "bicycling",
        RoutingProfile.CAR.value: "driving",
        RoutingProfile.TRUCK.value: "driving"
    }.get(mode, "driving")

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start[0]},{start[1]}",
        "destination": f"{end[0]},{end[1]}",
        "mode": gm_mode,
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return _convert_google_to_ors_format(r.json())
    except Exception as e:
        st.error(f"Google Maps routing failed: {e}")
        return {}

def _convert_google_to_ors_format(data: Dict) -> Dict:
    """
    Convert Google Maps route data to ORS-like GeoJSON format.

    This function extracts the polyline, decodes it to get coordinates, computes the
    total distance and duration, and then extracts individual steps (instructions) from
    each leg provided by Google Maps.

    Args:
        data (Dict): Raw route data from Google Maps API.

    Returns:
        Dict: Route data in ORS-like GeoJSON format with segments and steps.
    """
    if "routes" not in data or not data["routes"]:
        return {}
    
    route = data["routes"][0]
    # Decode the overview polyline for route geometry
    decoded = polyline.decode(route["overview_polyline"]["points"])
    # Google returns lat, lon pairs; convert to [lon, lat] for consistency
    coordinates = [[lon, lat] for lat, lon in decoded]
    
    # Calculate total distance and duration from all legs
    total_distance = sum(leg["distance"]["value"] for leg in route["legs"])
    total_duration = sum(leg["duration"]["value"] for leg in route["legs"])
    
    # Extract detailed steps for each leg
    steps = []
    for leg in route["legs"]:
        for step in leg.get("steps", []):
            steps.append({
                "instruction": step.get("html_instructions", "Proceed"),
                "distance": step["distance"]["value"],
                "duration": step["duration"]["value"]
            })
    
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "segments": [{
                    "steps": steps,
                    "distance": total_distance,
                    "duration": total_duration
                }],
                "summary": {
                    "distance": total_distance,
                    "duration": total_duration
                }
            }
        }]
    }

# ---- Traffic-Aware Travel Times ----

@st.cache_data(ttl=TRAFFIC_CACHE_EXPIRY, show_spinner="Fetching traffic info...")
def get_travel_times(
    coords: List[Tuple[float, float]],
    mode: Union[str, RoutingProfile] = RoutingProfile.DRIVING_CAR,
    depart_at: Optional[datetime] = None,
    model: TrafficModel = TrafficModel.BEST_GUESS
) -> Dict:
    """
    Fetch traffic-aware travel times for a route.

    Args:
        coords (List[Tuple[float, float]]): List of coordinates (latitude, longitude).
        mode (Union[str, RoutingProfile]): Routing profile. Defaults to DRIVING_CAR.
        depart_at (Optional[datetime]): Departure time. Defaults to None (current time).
        model (TrafficModel): Traffic prediction model. Defaults to BEST_GUESS.

    Returns:
        Dict: Route data with traffic information.
    """
    if len(coords) < 2:
        return {}
    try:
        if isinstance(mode, RoutingProfile):
            mode = mode.value
        route = get_route(coords[0], coords[-1], mode=mode, api_key=ORS_API_KEY, traffic=True)
        congestion_level = get_congestion_level(coords[0][0], coords[0][1]) or 0.0
        if "features" in route:
            route["metadata"] = {
                "traffic_data": {"congestion_percentage": round(congestion_level * 100, 2)}
            }
        return route
    except Exception as e:
        st.error(f"Traffic info error: {e}")
        return {}

def get_congestion_level(lat: float, lon: float) -> Optional[float]:
    """
    Calculate congestion level at a specific location using TomTom's traffic API.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        Optional[float]: Congestion level (0 to 1), or None if the request fails.
    """
    try:
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {"point": f"{lat},{lon}", "key": TOMTOM_API_KEY, "unit": "KMPH"}
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        speed = data["flowSegmentData"]["currentSpeed"]
        free_speed = data["flowSegmentData"]["freeFlowSpeed"]
        return 1 - (speed / free_speed) if free_speed > 0 else None
    except Exception as e:
        st.warning(f"Failed to get congestion level: {e}")
        return None

# ---- Route Instructions ----

def get_route_instructions(route_data: Dict) -> List[RouteInstruction]:
    """
    Extract detailed instructions from route data.

    Args:
        route_data (Dict): Route data in GeoJSON format.

    Returns:
        List[RouteInstruction]: List of route instructions.
    """
    if not route_data or "features" not in route_data:
        return []
    steps = []
    for segment in route_data["features"][0]["properties"]["segments"]:
        for step in segment.get("steps", []):
            steps.append(RouteInstruction(
                distance_km=step["distance"] / 1000,
                base_duration_min=step["duration"] / 60,
                traffic_duration_min=step["duration"] / 60,
                instruction=step.get("instruction", "Move"),
                step_type=step.get("type", "turn"),
                traffic=None
            ))
    return steps

# ---- Utility Functions ----

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """
    Decode a polyline string into a list of coordinates.

    Args:
        polyline_str (str): Encoded polyline string from Google Maps.

    Returns:
        List[Tuple[float, float]]: List of (latitude, longitude) coordinates.
    """
    try:
        return polyline.decode(polyline_str)
    except Exception as e:
        st.error(f"Polyline decoding failed: {e}")
        return []

def validate_coordinates(coords: Tuple[float, float]) -> bool:
    """
    Validate if coordinates are within valid geographical ranges.

    Args:
        coords (Tuple[float, float]): (latitude, longitude) coordinates.

    Returns:
        bool: True if valid, False otherwise.
    """
    lat, lon = coords
    return -90 <= lat <= 90 and -180 <= lon <= 180

def compare_algorithms(start_coords: Tuple[float, float], end_coords: Tuple[float, float], mode: str) -> Dict:
    """
    Compare classical and quantum routing algorithms (placeholder implementation).

    Args:
        start_coords (Tuple[float, float]): Starting coordinates (latitude, longitude).
        end_coords (Tuple[float, float]): Ending coordinates (latitude, longitude).
        mode (str): Mode of transportation.

    Returns:
        Dict: Comparison results with placeholder values.
    """
    classical_result = {"Time": 120, "Cost": 20, "Efficiency": 0.9}
    quantum_result = {"Time": 110, "Cost": 15, "Efficiency": 0.95}
    return {"Classical": classical_result, "Quantum": quantum_result}
