# QuantumNav Configuration File
# Last Updated: 2024-06-20

# ========================
# PERFORMANCE SETTINGS
# ========================
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
CONNECTION_POOL_SIZE = 10
THREAD_POOL_SIZE = 4

# ========================
# API KEYS CONFIGURATION
# ========================
"""API keys for external services used in routing and geocoding."""
ORS_API_KEY = "5b3ce3597851110001cf6248d95e9ba45205245a0661afd4207d9c23f08e1ff73487a7c3bdaa2a29"
GOOGLE_MAPS_API_KEY = "AIzaSyD4KIseK9v6PbICkPvQ7P-lPIqXPNV3Xxs"
MAPBOX_API_KEY = "pk.eyJ1IjoibmlraGlsMjM0NzEzNSIsImEiOiJjbTdqM3ZpdTcwMzg2MmpzZWY3amZrbnFoIn0.zOYYRKM_A31dAIA_Iqd1hg"
TOMTOM_API_KEY = "JgnpGSzMajgRFaPZAZ5Fpl0EbFL8t71b"
NOMINATIM_USER_AGENT = "quantum_nav"

# ========================
# API ENDPOINTS
# ========================
"""Base URLs for API services."""
ORS_BASE_URL = "https://api.openrouteservice.org/v2"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/"
TOMTOM_BASE_URL = "https://api.tomtom.com/routing/1/"

# ========================
# API CLIENT CONFIGURATION
# ========================
"""Configuration for the ORS API client."""
ORS_CLIENT_CONFIG = {
    'base_url': ORS_BASE_URL,
    'api_key': ORS_API_KEY,
    'timeout': REQUEST_TIMEOUT
}

# ========================
# CACHING CONFIGURATION
# ========================
"""Settings for caching to improve performance."""
CACHE_EXPIRY = 3600  # 1 hour in seconds
CACHE_MAX_SIZE = 100  # Max cached items
CACHE_DIR = "cache"
CACHE_ENABLED = True
TRAFFIC_CACHE_EXPIRY = 300  # 5 minutes for traffic data
ROUTE_CACHE_EXPIRY = 1800  # 30 minutes for route data
POI_CACHE_EXPIRY = 7200  # 2 hours for POI data

# ========================
# POINTS OF INTEREST (POI)
# ========================
"""Categories and settings for Points of Interest (POI)."""
POI_CATEGORIES = {
    'Petrol Pump': 'fuel',
    'Hospital': 'healthcare.hospital',
    'Restaurant': 'catering.restaurant',
    'Pharmacy': 'healthcare.pharmacy',
    'Hotel': 'accommodation.hotel',
    'Mall': 'commercial.shopping_mall',
    'Park': 'leisure.park'
}
POI_SEARCH_RADIUS = 5000  # meters (search radius for POIs)
MAX_POI_RESULTS = 50      # Maximum number of POI results to return
DEFAULT_POI_RADIUS = 5000 # meters (default radius for POI searches)

# ========================
# ROUTING CONFIGURATION
# ========================
"""Settings for route calculation and optimization."""
ROUTING_PROFILES = {
    'Car': 'driving-car',
    'Bike': 'cycling-regular',
    'Truck': 'driving-hgv',
    'Walking': 'foot-walking'
}
TRAFFIC_ENABLED = True
AVOID_TOLLS = False
AVOID_FERRIES = False
ALTERNATIVE_ROUTES = 2
MAX_ROUTE_ALTERNATIVES = 3
ROUTE_OPTIMIZATION = True

# ------------------------
# NEW FEATURE: Detailed Routing Instructions
# ------------------------
# When enabled, API calls for routing will request and return detailed turn-by-turn instructions.
ROUTE_INSTRUCTIONS_ENABLED = True

# ========================
# QUANTUM CONFIGURATION
# ========================
"""Settings for quantum routing optimization."""
QUANTUM_ITERATIONS = {
    'HIGH_PERFORMANCE': 5,
    'BALANCED': 3,
    'POWER_SAVER': 1
}
QUANTUM_SIMULATOR = 'aer_simulator'
MAX_QUBITS = 16
QUANTUM_OPTIMIZATION_ENABLED = True

# ========================
# SYSTEM DEFAULTS
# ========================
"""Default system settings."""
DEFAULT_VEHICLE_PROFILE = "driving-car"
LOGGING_ENABLED = True
LOG_LEVEL = "INFO"
LOG_FILE = "quantum_nav.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========================
# POWER MANAGEMENT
# ========================
"""Power mode configurations affecting update frequency and enabled features."""
POWER_MODES = {
    'HIGH_PERFORMANCE': {
        'update_interval': 2,
        'features_enabled': ['traffic', 'alternative_routes', 'quantum_optimization']
    },
    'BALANCED': {
        'update_interval': 5,
        'features_enabled': ['traffic', 'alternative_routes']
    },
    'POWER_SAVER': {
        'update_interval': 10,
        'features_enabled': ['basic_routing']
    }
}

# ========================
# ERROR HANDLING
# ========================
"""Error messages and retry settings."""
ERROR_MESSAGES = {
    'API_ERROR': 'Failed to connect to routing service',
    'GEOCODING_ERROR': 'Unable to find location',
    'ROUTING_ERROR': 'Could not calculate route',
    'CACHE_ERROR': 'Cache operation failed',
    'POI_ERROR': 'Failed to fetch points of interest',
    'QUANTUM_ERROR': 'Quantum optimization failed',
    'TRAFFIC_ERROR': 'Unable to fetch traffic data'
}
MAX_ERROR_RETRIES = 3
ERROR_RETRY_DELAY = 2  # seconds

# ========================
# DEBUG CONFIGURATION
# ========================
"""Settings for debugging and performance profiling."""
DEBUG = True
DEBUG_LOG_FILE = "debug.log"
VERBOSE_LOGGING = True
PROFILE_PERFORMANCE = True

# ========================
# SECURITY SETTINGS
# ========================
"""Security-related configurations."""
API_RATE_LIMIT = 100  # requests per minute
ENABLE_SSL_VERIFY = True
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# ========================
# HTTP CLIENT SETTINGS
# ========================
"""HTTP client configuration for API requests."""
HTTP_HEADERS = {
    'User-Agent': 'QuantumNav/1.0',
    'Accept': 'application/json'
}
PROXY_SETTINGS = {
    'enabled': False,
    'http': None,
    'https': None
}