import sqlite3
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import os
import streamlit as st

class OfflineCache:
    """
    Local caching system for route data, map tiles, and POIs using SQLite
    with automatic expiration and size management
    """
    def __init__(self, db_path: str = "quantumnav_cache.db", max_size_mb: int = 100):
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routes (
                    id TEXT PRIMARY KEY,
                    start_coords TEXT NOT NULL,
                    end_coords TEXT NOT NULL,
                    route_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    size_kb INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS map_tiles (
                    id TEXT PRIMARY KEY,
                    tile_key TEXT NOT NULL,
                    tile_data BLOB NOT NULL,
                    zoom INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    size_kb INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pois (
                    id TEXT PRIMARY KEY,
                    coords TEXT NOT NULL,
                    category TEXT NOT NULL,
                    poi_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    size_kb INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def _get_connection(self):
        """Get thread-safe database connection"""
        return sqlite3.connect(self.db_path, timeout=10)
    
    def _generate_key(self, *args) -> str:
        """Generate consistent cache key from input parameters"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _check_storage(self):
        """Ensure we don't exceed max storage size"""
        with self._get_connection() as conn:
            total_size = conn.execute(
                "SELECT SUM(size_kb) FROM (SELECT size_kb FROM routes UNION ALL SELECT size_kb FROM map_tiles UNION ALL SELECT size_kb FROM pois)"
            ).fetchone()[0] or 0
            
            if total_size > self.max_size_mb * 1024:
                self._cleanup_old_items()
    
    def _cleanup_old_items(self):
        """Remove least recently used items"""
        with self._get_connection() as conn:
            # Get all items sorted by last access
            items = []
            
            # Routes
            routes = conn.execute(
                "SELECT id, 'route' as type, size_kb, last_accessed FROM routes"
            ).fetchall()
            items.extend(routes)
            
            # Tiles
            tiles = conn.execute(
                "SELECT id, 'tile' as type, size_kb, last_accessed FROM map_tiles"
            ).fetchall()
            items.extend(tiles)
            
            # POIs
            pois = conn.execute(
                "SELECT id, 'poi' as type, size_kb, last_accessed FROM pois"
            ).fetchall()
            items.extend(pois)
            
            # Sort by last accessed (oldest first)
            items.sort(key=lambda x: x[3])
            
            # Delete until we're under 90% of max size
            target_size = self.max_size_mb * 1024 * 0.9
            current_size = sum(item[2] for item in items)
            
            for item in items:
                if current_size <= target_size:
                    break
                
                if item[1] == 'route':
                    conn.execute("DELETE FROM routes WHERE id = ?", (item[0],))
                elif item[1] == 'tile':
                    conn.execute("DELETE FROM map_tiles WHERE id = ?", (item[0],))
                elif item[1] == 'poi':
                    conn.execute("DELETE FROM pois WHERE id = ?", (item[0],))
                
                current_size -= item[2]
            
            conn.commit()
    
    def cache_route(self, start_coords: Tuple[float, float], 
                   end_coords: Tuple[float, float], 
                   route_data: Dict[str, Any],
                   ttl_hours: int = 24) -> bool:
        """
        Cache a route with expiration time
        Args:
            start_coords: (lat, lng) tuple
            end_coords: (lat, lng) tuple
            route_data: Route data to cache
            ttl_hours: Time-to-live in hours
        Returns:
            True if successful
        """
        try:
            cache_id = self._generate_key(start_coords, end_coords)
            serialized = pickle.dumps(route_data)
            size_kb = len(serialized) // 1024
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO routes 
                    (id, start_coords, end_coords, route_data, expires_at, size_kb)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_id,
                        json.dumps(start_coords),
                        json.dumps(end_coords),
                        serialized,
                        expires_at.isoformat(),
                        size_kb
                    )
                )
                conn.commit()
            
            self._check_storage()
            return True
        except Exception as e:
            st.error(f"Route caching failed: {str(e)}")
            return False
    
    def get_cached_route(self, start_coords: Tuple[float, float],
                        end_coords: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached route if available and not expired
        Args:
            start_coords: (lat, lng) tuple
            end_coords: (lat, lng) tuple
        Returns:
            Cached route data or None if not available/expired
        """
        try:
            cache_id = self._generate_key(start_coords, end_coords)
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT route_data, expires_at FROM routes 
                    WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (cache_id, datetime.now().isoformat())
                )
                result = cursor.fetchone()
                
                if result:
                    # Update last accessed time
                    conn.execute(
                        "UPDATE routes SET last_accessed = ? WHERE id = ?",
                        (datetime.now().isoformat(), cache_id)
                    )
                    conn.commit()
                    return pickle.loads(result[0])
            return None
        except Exception as e:
            st.error(f"Route retrieval failed: {str(e)}")
            return None
    
    def cache_pois(self, coords: Tuple[float, float], 
                  category: str, 
                  poi_data: Dict[str, Any],
                  ttl_hours: int = 12) -> bool:
        """
        Cache points of interest data
        Args:
            coords: (lat, lng) center point
            category: POI category
            poi_data: POI data to cache
            ttl_hours: Time-to-live in hours
        """
        try:
            cache_id = self._generate_key(coords, category)
            serialized = pickle.dumps(poi_data)
            size_kb = len(serialized) // 1024
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pois 
                    (id, coords, category, poi_data, expires_at, size_kb)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_id,
                        json.dumps(coords),
                        category,
                        serialized,
                        expires_at.isoformat(),
                        size_kb
                    )
                )
                conn.commit()
            
            self._check_storage()
            return True
        except Exception as e:
            st.error(f"POI caching failed: {str(e)}")
            return False
    
    def get_cached_pois(self, coords: Tuple[float, float],
                       category: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached POIs if available
        Args:
            coords: (lat, lng) center point
            category: POI category
        Returns:
            Cached POI data or None if not available/expired
        """
        try:
            cache_id = self._generate_key(coords, category)
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT poi_data FROM pois 
                    WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (cache_id, datetime.now().isoformat())
                )
                result = cursor.fetchone()
                return pickle.loads(result[0]) if result else None
        except Exception as e:
            st.error(f"POI retrieval failed: {str(e)}")
            return None
    
    def cache_map_tile(self, tile_key: str, 
                      tile_data: bytes, 
                      zoom: int, x: int, y: int) -> bool:
        """
        Cache map tile data
        Args:
            tile_key: Unique tile identifier
            tile_data: Binary tile data
            zoom: Zoom level
            x: Tile X coordinate
            y: Tile Y coordinate
        """
        try:
            cache_id = self._generate_key(tile_key)
            size_kb = len(tile_data) // 1024
            
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO map_tiles 
                    (id, tile_key, tile_data, zoom, x, y, size_kb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cache_id, tile_key, tile_data, zoom, x, y, size_kb)
                )
                conn.commit()
            
            self._check_storage()
            return True
        except Exception as e:
            st.error(f"Tile caching failed: {str(e)}")
            return False
    
    def get_cached_tile(self, tile_key: str) -> Optional[bytes]:
        """
        Retrieve cached map tile if available
        Args:
            tile_key: Unique tile identifier
        Returns:
            Cached tile data or None if not available
        """
        try:
            cache_id = self._generate_key(tile_key)
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT tile_data FROM map_tiles WHERE id = ?",
                    (cache_id,)
                )
                result = cursor.fetchone()
                if result:
                    # Update last accessed time
                    conn.execute(
                        "UPDATE map_tiles SET last_accessed = ? WHERE id = ?",
                        (datetime.now().isoformat(), cache_id)
                    )
                    conn.commit()
                    return result[0]
            return None
        except Exception as e:
            st.error(f"Tile retrieval failed: {str(e)}")
            return None
    
    def clear_cache(self) -> bool:
        """Clear all cached data"""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM routes")
                conn.execute("DELETE FROM map_tiles")
                conn.execute("DELETE FROM pois")
                conn.commit()
            return True
        except Exception as e:
            st.error(f"Cache clearing failed: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        try:
            with self._get_connection() as conn:
                # Routes
                routes = conn.execute(
                    "SELECT COUNT(*), SUM(size_kb) FROM routes"
                ).fetchone()
                stats["routes"] = {
                    "count": routes[0] or 0,
                    "size_kb": routes[1] or 0
                }
                
                # Tiles
                tiles = conn.execute(
                    "SELECT COUNT(*), SUM(size_kb) FROM map_tiles"
                ).fetchone()
                stats["tiles"] = {
                    "count": tiles[0] or 0,
                    "size_kb": tiles[1] or 0
                }
                
                # POIs
                pois = conn.execute(
                    "SELECT COUNT(*), SUM(size_kb) FROM pois"
                ).fetchone()
                stats["pois"] = {
                    "count": pois[0] or 0,
                    "size_kb": pois[1] or 0
                }
                
                # Total
                stats["total"] = {
                    "count": stats["routes"]["count"] + stats["tiles"]["count"] + stats["pois"]["count"],
                    "size_mb": (stats["routes"]["size_kb"] + stats["tiles"]["size_kb"] + stats["pois"]["size_kb"]) / 1024
                }
        except Exception as e:
            st.error(f"Cache stats failed: {str(e)}")
        
        return stats

# Global cache instance
cache = OfflineCache()