# utils/location_tracker.py

import streamlit as st
import streamlit.components.v1 as components

def inject_location_tracker(interval: int = 5):
    """
    Injects JavaScript to get the user's location and update it every few seconds.
    
    This function embeds a JavaScript snippet into the Streamlit app that uses the
    Geolocation API to fetch the user's latitude and longitude. The location data is
    sent to the parent window via postMessage events, which can be processed elsewhere
    (e.g., via query parameters or a custom component).

    Args:
        interval (int): How often to fetch the location (in seconds). Default is 5 seconds.
    """
    js_code = f"""
    <script>
    const getLocation = () => {{
        if (navigator.geolocation) {{
            navigator.geolocation.getCurrentPosition(
                (position) => {{
                    const coords = {{
                        lat: position.coords.latitude,
                        lon: position.coords.longitude
                    }};
                    window.parent.postMessage({{type: "LOCATION", coords}}, "*");
                    // Log to console for debugging purposes
                    console.log("Location updated:", coordsن);
                }},
                (error) => {{
                    console.warn("Geolocation error:", error.message);
                }},
                {{
                    enableHighAccuracy: true,
                    timeout: 5000,
                    maximumAge: 0
                }}
            );
        }} else {{
            console.warn("Geolocation not supported by this browser");
        }}
    }};

    // Fetch location immediately and then at regular intervals
    setInterval(getLocation, {interval * 1000});
    getLocation(); // Initial call to get location right away
    </script>
    """
    # Inject the JavaScript into the Streamlit app with zero height to keep it invisible
    components.html(js_code, height=0)

def receive_location_from_js():
    """
    Placeholder function to handle location updates from JavaScript postMessage events.
    
    Due to Streamlit's iframe sandbox limitations, direct event handling via postMessage
    is not possible in the default setup. This function serves as a conceptual placeholder,
    outlining the intended behavior and suggesting workarounds for real-time updates.

    Intended behavior:
        - Listen for postMessage events from the injected JavaScript.
        - Update st.session_state.current_location with the received coordinates (e.g., {'lat': 37.7749, 'lon': -122.4194}).

    Limitations:
        - Streamlit's sandboxed iframes prevent direct JavaScript event listeners.
    
    Workarounds:
        1. Use a custom Streamlit component to handle real-time JavaScript events.
        2. Poll st.session_state or use query parameters to simulate updates (as done in app.py).
    """
    # Conceptual implementation (not functional in default Streamlit due to sandboxing)
    def message_handler(event):
        if event.data.get("type") == "LOCATION":
            st.session_state.current_location = event.data["coords"]

    # Note: Streamlit does not natively support direct event listeners for postMessage.
    # As a workaround, consider implementing one of the suggested solutions above.
    pass  # Placeholder for now; replace with a custom solution as needed