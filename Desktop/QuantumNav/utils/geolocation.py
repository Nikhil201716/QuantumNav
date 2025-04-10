import requests
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import time
from openrouteservice import Client
from config import ORS_API_KEY, GOOGLE_MAPS_API_KEY  # Assuming GOOGLE_MAPS_API_KEY is in config.py

# Constants
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
GOOGLE_MAPS_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
ORS_API_URL = "https://api.openrouteservice.org"  # Base URL for OpenRouteService
USER_AGENT = "QuantumNav/1.0 (https://github.com/your-repo; contact@yourdomain.com)"
REQUEST_TIMEOUT = 15  # seconds
CACHE_EXPIRY = 3600  # 1 hour in seconds

class GeoService(Enum):
    """Enumeration of available geolocation services"""
    NOMINATIM = "nominatim"
    GOOGLE_MAPS = "google_maps"
    OPENROUTE = "openroute"  # Added OpenRouteService

@dataclass
class GeoResult:
    """Structured geolocation result"""
    latitude: float
    longitude: float
    address: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    service: Optional[GeoService] = None
    response_time: Optional[float] = None

@dataclass
class ReverseGeoResult:
    """Structured reverse geocoding result"""
    address: str
    components: Dict[str, Any]
    raw_data: Dict[str, Any]
    service: GeoService
    response_time: float

def _make_nominatim_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generic Nominatim request handler"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        response_time = time.time() - start_time
        
        response.raise_for_status()
        return {
            "data": response.json(),
            "response_time": response_time
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Geolocation API error: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_EXPIRY, show_spinner=False)
def get_coordinates(
    location_query: str,
    service: GeoService = GeoService.NOMINATIM,
    country_bias: Optional[str] = None,
    viewbox: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> Optional[GeoResult]:
    """
    Convert location query to geographic coordinates with enhanced features
    
    Args:
        location_query: Address or place name to geocode
        service: Which geocoding service to use
        country_bias: Prioritize results in this country (ISO code)
        viewbox: Bounding box to prioritize results [(min_lon, min_lat), (max_lon, max_lat)]
    
    Returns:
        GeoResult object with structured data or None if failed
    """
    if not location_query or not isinstance(location_query, str):
        st.error("Invalid location query")
        return None

    try:
        if service == GeoService.NOMINATIM:
            params = {
                "q": location_query,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1
            }
            
            if country_bias:
                params["countrycodes"] = country_bias.lower()
            
            if viewbox:
                params["viewbox"] = f"{viewbox[0][0]},{viewbox[0][1]},{viewbox[1][0]},{viewbox[1][1]}"
                params["bounded"] = 1
            
            result = _make_nominatim_request(f"{NOMINATIM_URL}/search", params)
            if not result or not result["data"]:
                st.error(f"Could not geocode address with Nominatim: {location_query}")
                return None
                
            data = result["data"][0]
            return GeoResult(
                latitude=float(data["lat"]),
                longitude=float(data["lon"]),
                address=data.get("display_name"),
                raw_data=data,
                service=service,
                response_time=result["response_time"]
            )
            
        elif service == GeoService.GOOGLE_MAPS:
            params = {
                "address": location_query,
                "key": GOOGLE_MAPS_API_KEY
            }
            
            if country_bias:
                params["components"] = f"country:{country_bias}"
            
            result = _make_nominatim_request(GOOGLE_MAPS_GEOCODE_URL, params)
            if not result or result["data"]["status"] != "OK":
                st.error(f"Could not geocode address with Google Maps: {location_query}")
                return None
                
            data = result["data"]["results"][0]
            location = data["geometry"]["location"]
            return GeoResult(
                latitude=location["lat"],
                longitude=location["lng"],
                address=data["formatted_address"],
                raw_data=data,
                service=service,
                response_time=result["response_time"]
            )
            
        elif service == GeoService.OPENROUTE:
            client = Client(key=ORS_API_KEY)
            try:
                start_time = time.time()
                result = client.pelias_search(text=location_query)
                response_time = time.time() - start_time
                
                if result and 'features' in result and len(result['features']) > 0:
                    data = result['features'][0]
                    geometry = data['geometry']
                    return GeoResult(
                        latitude=geometry['coordinates'][1],
                        longitude=geometry['coordinates'][0],
                        address=data.get('properties', {}).get('label'),
                        raw_data=data,
                        service=service,
                        response_time=response_time
                    )
                else:
                    st.error(f"Could not geocode address with OpenRouteService: {location_query}")
                    return None
            except Exception as e:
                st.error(f"OpenRouteService API error: {str(e)}")
                return None
            
    except Exception as e:
        st.error(f"Geocoding failed: {str(e)}")
        return None

def geocode_address(address: str, service: GeoService = GeoService.OPENROUTE) -> Optional[Tuple[float, float]]:
    """
    Convert a textual address to latitude/longitude coordinates.
    
    Args:
        address: The address or place name to geocode.
        service: The geocoding service to use (default: OpenRouteService).
    
    Returns:
        A tuple of (latitude, longitude) if successful, otherwise None.
    """
    result = get_coordinates(location_query=address, service=service)
    if result:
        return (result.latitude, result.longitude)
    return None

@st.cache_data(ttl=CACHE_EXPIRY, show_spinner=False)
def reverse_geocode(
    latitude: float,
    longitude: float,
    service: GeoService = GeoService.NOMINATIM,
    zoom: Optional[int] = None,
    language: str = "en"
) -> Optional[ReverseGeoResult]:
    """
    Convert coordinates to human-readable address with enhanced features
    
    Args:
        latitude: Geographic latitude
        longitude: Geographic longitude
        service: Which reverse geocoding service to use
        zoom: Level of detail (Nominatim only, 0-18)
        language: Preferred language for results
    
    Returns:
        ReverseGeoResult object with structured data or None if failed
    """
    if not validate_coordinates((latitude, longitude)):
        st.error("Invalid coordinates provided")
        return None

    try:
        if service == GeoService.NOMINATIM:
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "addressdetails": 1,
                "accept-language": language
            }
            
            if zoom is not None:
                params["zoom"] = min(max(zoom, 0), 18)
            
            result = _make_nominatim_request(f"{NOMINATIM_URL}/reverse", params)
            if not result:
                return None
                
            data = result["data"]
            return ReverseGeoResult(
                address=data.get("display_name", "Unknown location"),
                components=data.get("address", {}),
                raw_data=data,
                service=service,
                response_time=result["response_time"]
            )
            
        elif service == GeoService.GOOGLE_MAPS:
            params = {
                "latlng": f"{latitude},{longitude}",
                "key": GOOGLE_MAPS_API_KEY,
                "language": language
            }
            
            result = _make_nominatim_request(GOOGLE_MAPS_GEOCODE_URL, params)
            if not result or result["data"]["status"] != "OK":
                return None
                
            data = result["data"]["results"][0]
            return ReverseGeoResult(
                address=data["formatted_address"],
                components=_parse_google_components(data["address_components"]),
                raw_data=data,
                service=service,
                response_time=result["response_time"]
            )
            
    except Exception as e:
        st.error(f"Reverse geocoding failed: {str(e)}")
        return None

def _parse_google_components(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse Google Maps address components into structured format"""
    parsed = {}
    for component in components:
        for type in component["types"]:
            parsed[type] = component["long_name"]
    return parsed

def validate_coordinates(coords: Tuple[float, float]) -> bool:
    """
    Validate geographic coordinates
    
    Args:
        coords: Tuple of (latitude, longitude)
    
    Returns:
        True if coordinates are valid, False otherwise
    """
    try:
        lat, lon = coords
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except (TypeError, ValueError):
        return False

def batch_geocode(
    queries: List[str],
    service: GeoService = GeoService.NOMINATIM,
    rate_limit: float = 0.1
) -> List[Optional[GeoResult]]:
    """
    Batch geocode multiple location queries with rate limiting
    
    Args:
        queries: List of location strings to geocode
        service: Which geocoding service to use
        rate_limit: Minimum delay between requests in seconds
    
    Returns:
        List of GeoResult objects (or None for failed queries)
    """
    results = []
    for query in queries:
        if query:
            result = get_coordinates(query, service)
            results.append(result)
            time.sleep(rate_limit)
        else:
            results.append(None)
    return results

def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    method: str = "haversine"
) -> Optional[float]:
    """
    Calculate distance between two points in kilometers
    
    Args:
        point1: (lat, lon) of first point
        point2: (lat, lon) of second point
        method: Calculation method ('haversine' or 'vincenty')
    
    Returns:
        Distance in kilometers or None if calculation failed
    """
    if not all(validate_coordinates(p) for p in [point1, point2]):
        return None
    
    try:
        from geopy.distance import geodesic
        return geodesic(point1, point2).km
    except Exception as e:
        st.error(f"Distance calculation failed: {str(e)}")
        return None