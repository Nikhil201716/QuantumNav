import streamlit as st
from typing import Optional, Callable
import time
from enum import Enum, auto
import platform

class PowerMode(Enum):
    """Device power modes"""
    HIGH_PERFORMANCE = auto()
    BALANCED = auto()
    POWER_SAVER = auto()

class BatteryManager:
    """
    Mobile power management system that dynamically adjusts location polling
    and computation based on battery status
    """
    def __init__(self):
        self._update_interval = 5  # seconds (default)
        self._current_mode = PowerMode.BALANCED
        self._last_check = 0
        self._callbacks = []
        
    def check_battery_status(self) -> Optional[float]:
        """
        Check battery level (simulated/actual based on platform)
        Returns:
            Battery percentage (0-100) or None if unavailable
        """
        try:
            # Mobile detection (simplified - in production use proper detection)
            if platform.system() == 'Android' or platform.system() == 'iOS':
                # This would use platform-specific APIs in production
                return self._simulate_battery_level()
            else:
                # Desktop/laptop - use psutil for actual battery info
                import psutil
                battery = psutil.sensors_battery()
                return battery.percent if battery else 100
        except Exception:
            return None
    
    def optimize_power_usage(self) -> PowerMode:
        """
        Adjust operations based on battery level
        Returns:
            Current power mode
        """
        battery_level = self.check_battery_status()
        
        if battery_level is None:
            return PowerMode.HIGH_PERFORMANCE
            
        if battery_level < 20:
            self._current_mode = PowerMode.POWER_SAVER
            self._update_interval = 30
        elif battery_level < 50:
            self._current_mode = PowerMode.BALANCED
            self._update_interval = 15
        else:
            self._current_mode = PowerMode.HIGH_PERFORMANCE
            self._update_interval = 5
            
        # Notify all registered callbacks
        for callback in self._callbacks:
            callback(self._current_mode, self._update_interval)
            
        return self._current_mode
    
    def register_callback(self, callback: Callable[[PowerMode, int], None]):
        """Register for power mode changes"""
        self._callbacks.append(callback)
    
    def get_current_mode(self) -> PowerMode:
        return self._current_mode
        
    def get_update_interval(self) -> int:
        return self._update_interval
        
    def _simulate_battery_level(self) -> float:
        """Simulate battery drain for demo purposes"""
        if not hasattr(self, '_simulated_battery'):
            self._simulated_battery = 80.0
            
        # Simulate 1% drain per minute
        elapsed = time.time() - self._last_check
        self._simulated_battery = max(0, self._simulated_battery - (elapsed / 60))
        self._last_check = time.time()
        return self._simulated_battery

# Global battery manager instance
battery_manager = BatteryManager()