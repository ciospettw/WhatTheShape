import argparse
import csv
import os
import sys
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import queue
import pickle
import re
import math

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pyproj import Geod, Transformer
from shapely.geometry import LineString
import tqdm

# Use pyosmium (osmium) to parse the PBF and build graphs
import osmium


# -----------------------------
# Utilities
# -----------------------------

globe = Geod(ellps="WGS84")
# Project lon/lat -> meters for shape simplification, then back
_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def geodesic_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return geodesic distance in meters between two WGS84 coords."""
    # pyproj.Geod expects lon, lat
    _, _, dist = globe.inv(lon1, lat1, lon2, lat2)
    return float(dist)


@dataclass
class GraphIndex:
    graph: nx.DiGraph
    node_ids: np.ndarray  # ordered node ids aligned with coords
    coords: np.ndarray  # Nx2 array [lat, lon]
    kdtree: cKDTree


def graph_to_index(G: nx.DiGraph) -> GraphIndex:
    """Create GraphIndex from a DiGraph, rebuilding KDTree."""
    node_ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
    if node_ids.size == 0:
        coords = np.zeros((0, 2), dtype=float)
        kdt = cKDTree(np.zeros((0, 2), dtype=float))
        return GraphIndex(G, node_ids, coords, kdt)
    lats = np.array([G.nodes[n]["lat"] for n in node_ids], dtype=float)
    lons = np.array([G.nodes[n]["lon"] for n in node_ids], dtype=float)
    coords = np.column_stack([lats, lons])
    kdt = cKDTree(coords)
    return GraphIndex(G, node_ids, coords, kdt)


class _RailStopCollector(osmium.SimpleHandler):
    """Collect rail-related stop nodes from OSM."""

    RAIL_STOP_TAGS = {"station", "halt", "stop", "tram_stop", "stop_position"}

    def __init__(self, bbox: Optional[Tuple[float, float, float, float]] = None):
        super().__init__()
        self.bbox = bbox
        self.points: List[Tuple[float, float]] = []
        self._node_counter = 0
        self._last_log = 0

    def _in_bbox(self, lat: float, lon: float) -> bool:
        if self.bbox is None:
            return True
        minlon, minlat, maxlon, maxlat = self.bbox
        return (minlon <= lon <= maxlon) and (minlat <= lat <= maxlat)

    def node(self, n):
        if not n.location.valid():
            return
        lat = n.location.lat
        lon = n.location.lon
        if not self._in_bbox(lat, lon):
            return
        self._node_counter += 1
        if self._node_counter - self._last_log >= 200000:
            print(f"[OSM] scanned {self._node_counter:,} nodes for rail stops...", flush=True)
            self._last_log = self._node_counter
        tags = {t.k: t.v for t in n.tags}
        railway_tag = tags.get("railway")
        pt_tag = tags.get("public_transport")
        if (railway_tag and railway_tag in self.RAIL_STOP_TAGS) or pt_tag == "stop_position":
            self.points.append((lat, lon))


class _WayGraphBuilder(osmium.SimpleHandler):
    """Build a road/rail graph from OSM ways.

    mode = 'road' or 'rail'
    """

    ROAD_HIGHWAYS = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "service",
        "living_street",
        "busway",
        "road",
    }

    # Restrict to heavy/mainline rail to avoid tram/metro routing for Trenitalia
    RAIL_TYPES = {
        "rail",
    }

    FERRY_ROUTE_TAG = {"ferry"}  # route=ferry on a way

    def __init__(self, mode: str = "road", bbox: Optional[Tuple[float, float, float, float]] = None, event_queue: Optional[queue.Queue] = None, region: str = ""):
        super().__init__()
        self.mode = mode
        self.G: nx.DiGraph = nx.DiGraph()
        # Cache of default oneway-by-type
        self._oneway_default = {"motorway": True, "motorway_link": True}
        # Optional bounding box filter (minlon, minlat, maxlon, maxlat)
        self.bbox = bbox
        # Optional event queue for real-time visualization
        self.event_queue = event_queue
        self.region = region
        self.edge_counter = 0
        self.batch_size = 50  # Send edges in batches

    def _in_bbox(self, lat: float, lon: float) -> bool:
        if self.bbox is None:
            return True
        minlon, minlat, maxlon, maxlat = self.bbox
        return (minlon <= lon <= maxlon) and (minlat <= lat <= maxlat)

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}

        if self.mode == "road":
            hw = tags.get("highway")
            if hw is None or hw not in self.ROAD_HIGHWAYS:
                return
            oneway_tag = tags.get("oneway", "no").lower()
            oneway = oneway_tag in {"yes", "true", "1"} or self._oneway_default.get(hw, False) or tags.get("junction") == "roundabout"
        else:  # rail
            rw = tags.get("railway")
            route_tag = tags.get("route")

            is_rail = rw is not None and rw in self.RAIL_TYPES
            is_ferry = route_tag in self.FERRY_ROUTE_TAG

            if not (is_rail or is_ferry):
                return

            # Assume rails are bidirectional unless tagged otherwise; ferry too
            oneway = False

        maxspeed = _parse_maxspeed(tags.get("maxspeed"))
        if maxspeed is None and route_tag in self.FERRY_ROUTE_TAG:
            maxspeed = 25.0  # default ferry cruising speed km/h when missing

        # Build edges from node refs and optionally filter by bbox
        node_seq = []
        any_in_bbox = False
        for n in w.nodes:
            if not n.location.valid():
                continue
            lat = n.location.lat
            lon = n.location.lon
            if self._in_bbox(lat, lon):
                any_in_bbox = True
            node_seq.append((n.ref, lat, lon))
        if self.bbox is not None and not any_in_bbox:
            return
        if len(node_seq) < 2:
            return

        # Add nodes (with coords) and edges
        for (u, lat1, lon1), (v, lat2, lon2) in zip(node_seq[:-1], node_seq[1:]):
            length = geodesic_m(lat1, lon1, lat2, lon2)

            # Add nodes with attributes
            if not self.G.has_node(u):
                self.G.add_node(u, lat=lat1, lon=lon1)
            if not self.G.has_node(v):
                self.G.add_node(v, lat=lat2, lon=lon2)

            # Add directed edge(s)
            self.G.add_edge(u, v, length=length, maxspeed=maxspeed)
            if not oneway:
                self.G.add_edge(v, u, length=length, maxspeed=maxspeed)
            
            # Emit edge for visualization
            if self.event_queue is not None:
                self.edge_counter += 1
                if self.edge_counter % self.batch_size == 0:
                    try:
                        self.event_queue.put_nowait({
                            'type': 'edge',
                            'region': self.region,
                            'mode': self.mode,
                            'lat1': lat1,
                            'lon1': lon1,
                            'lat2': lat2,
                            'lon2': lon2,
                            'count': self.edge_counter
                        })
                    except queue.Full:
                        pass  # Skip if queue is full


# -----------------------------
# GTFS loading
# -----------------------------


def load_gtfs(gtfs_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"), dtype=str)
    stop_times = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"), dtype=str)
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"), dtype=str)
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"), dtype=str)

    # Ensure numeric lat/lon
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    # Ensure sequence integer for sorting
    if "stop_sequence" in stop_times.columns:
        stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    return stops, stop_times, trips, routes


def detect_route_mode(row, prefer_rail_only: bool = False) -> str:
    """Heuristic classification of a GTFS route into road/rail."""
    text = " ".join([
        str(row.get("route_id", "")),
        str(row.get("route_short_name", "")),
        str(row.get("route_long_name", "")),
    ]).lower()

    bus_keywords = ["bus", "autobus", "pullman", "freccialink", "fl"]
    rail_keywords = [
        "rail", "train", "metro", "subway", "tram",
        "ferrovia", "ferrovi", "metropolitana",
        "freccia", "ic", "icn", "rv", "regionale", "reg", "fr", "fb", "fa", "en", "ec", "eurocity",
    ]

    if any(k in text for k in bus_keywords):
        return "road"
    if any(k in text for k in rail_keywords):
        return "rail"

    try:
        rtype = int(row.get("route_type", "3"))
    except Exception:
        rtype = 3
    if rtype == 3:
        return "road"
    if rtype in (0, 1, 2):
        return "rail"
    return "rail" if prefer_rail_only else "road"


def detect_speed_pref(row) -> str:
    """Classify rail routes by desired track speed bucket.

    Returns: "high", "low", "flex" (FA/FB), or "any" for non-rail.
    """
    # Use short_name primarily for precise matching to avoid substring collisions
    short_name = str(row.get("route_short_name", "")).strip().lower()
    long_name = str(row.get("route_long_name", "")).strip().lower()
    route_id = str(row.get("route_id", "")).lower()

    # Check short_name first with exact matches for precision
    if short_name in ["fr"]:
        return "high"
    if short_name in ["fa", "fb", "ic", "icn", "ec", "en"]:
        return "flex"
    if short_name in ["reg", "rv", "sfm", "met", "exp", "bus", "fl"]:
        return "low"

    # Fallback to substring matching in long_name and route_id
    text = f"{route_id} {long_name}"
    
    high_kw = ["frecciarossa", "alta velocita", "av", "tav", "italo", "hs"]
    flex_kw = ["frecciargento", "frecciabianca", "intercity", "eurocity", "euronight"]
    low_kw = [
        "regionale", "espresso", "pullman", "freccialink"
    ]

    if any(k in text for k in high_kw):
        return "high"
    if any(k in text for k in flex_kw):
        return "flex"
    if any(k in text for k in low_kw):
        return "low"
    return "any"


def normalize_rail_stops_with_osm(
    stops: pd.DataFrame,
    rail_stop_ids: List[str],
    pbf_paths: List[str],
    bbox: Optional[Tuple[float, float, float, float]],
    max_distance_m: float = 600.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Snap rail GTFS stops to the nearest OSM rail stop if close enough."""

    if not rail_stop_ids:
        return stops, {"updated": 0, "skipped": 0, "too_far": 0, "candidates": 0}

    collector = _RailStopCollector(bbox=bbox)
    for i, pbf in enumerate(pbf_paths, start=1):
        print(f"[STEP] Parsing OSM for rail stops ({i}/{len(pbf_paths)}): {pbf}", flush=True)
        collector.apply_file(pbf, locations=True)
        print(f"[STEP] Done {pbf}: collected {len(collector.points)} candidate stops so far", flush=True)

    if not collector.points:
        return stops, {"updated": 0, "skipped": len(set(rail_stop_ids)), "too_far": 0, "candidates": 0}

    coords = np.array(collector.points, dtype=float)
    kdt = cKDTree(coords)

    stops_norm = stops.copy()
    updated = 0
    too_far = 0
    visited = set()

    for sid in tqdm.tqdm(rail_stop_ids, desc="Normalize rail stops", unit="stop"):
        if sid in visited:
            continue
        visited.add(sid)
        row = stops_norm.loc[stops_norm["stop_id"] == sid]
        if row.empty:
            continue
        lat = float(row.iloc[0]["stop_lat"])
        lon = float(row.iloc[0]["stop_lon"])
        dist, pos = kdt.query([lat, lon], k=1)
        osm_lat, osm_lon = coords[pos]
        dist_m = geodesic_m(lat, lon, osm_lat, osm_lon)
        if dist_m <= max_distance_m:
            stops_norm.loc[stops_norm["stop_id"] == sid, ["stop_lat", "stop_lon"]] = [osm_lat, osm_lon]
            updated += 1
        else:
            too_far += 1

    summary = {
        "updated": updated,
        "skipped": len(visited) - updated - too_far,
        "too_far": too_far,
        "candidates": len(coords),
    }
    return stops_norm, summary


def normalize_rail_stops_with_graph(
    stops: pd.DataFrame,
    rail_stop_ids: List[str],
    rail_idx: GraphIndex,
    max_distance_m: float = 600.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Snap rail GTFS stops to nearest rail graph node to avoid re-reading PBFs."""
    if rail_idx.coords.shape[0] == 0 or not rail_stop_ids:
        return stops, {"updated": 0, "skipped": len(set(rail_stop_ids)), "too_far": 0}

    kdt = rail_idx.kdtree
    coords = rail_idx.coords

    stops_norm = stops.copy()
    updated = 0
    too_far = 0
    visited = set()

    for sid in tqdm.tqdm(rail_stop_ids, desc="Normalize rail stops (graph)", unit="stop"):
        if sid in visited:
            continue
        visited.add(sid)
        row = stops_norm.loc[stops_norm["stop_id"] == sid]
        if row.empty:
            continue
        lat = float(row.iloc[0]["stop_lat"])
        lon = float(row.iloc[0]["stop_lon"])
        dist, pos = kdt.query([lat, lon], k=1)
        osm_lat, osm_lon = coords[pos]
        dist_m = geodesic_m(lat, lon, osm_lat, osm_lon)
        if dist_m <= max_distance_m:
            stops_norm.loc[stops_norm["stop_id"] == sid, ["stop_lat", "stop_lon"]] = [osm_lat, osm_lon]
            updated += 1
        else:
            too_far += 1

    summary = {
        "updated": updated,
        "skipped": len(visited) - updated - too_far,
        "too_far": too_far,
    }
    return stops_norm, summary


def load_stop_overrides(path: str) -> List[Dict[str, Union[str, float]]]:
    """Load manual stop overrides from CSV with columns stop_name, stop_lat, stop_lon."""
    overrides: List[Dict[str, Union[str, float]]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_raw = (row.get("stop_name") or "").strip()
            if not name_raw:
                continue
            try:
                lat = float(row.get("stop_lat", ""))
                lon = float(row.get("stop_lon", ""))
            except ValueError:
                continue
            overrides.append({
                "stop_name": name_raw,
                "stop_name_norm": name_raw.casefold(),
                "stop_lat": lat,
                "stop_lon": lon,
            })
    return overrides


def apply_stop_overrides(stops: pd.DataFrame, overrides_path: str) -> Tuple[pd.DataFrame, Dict[str, Union[int, List[str]]]]:
    """Apply manual stop coordinate overrides by matching stop_name case-insensitively."""
    overrides = load_stop_overrides(overrides_path)
    if not overrides:
        return stops, {"updated": 0, "missing": [], "total_overrides": 0}

    stops_out = stops.copy()
    name_norm = stops_out["stop_name"].astype(str).str.strip().str.casefold()

    updated = 0
    missing: List[str] = []

    for ov in overrides:
        mask = name_norm == ov["stop_name_norm"]
        count = int(mask.sum())
        if count:
            stops_out.loc[mask, ["stop_lat", "stop_lon"]] = [ov["stop_lat"], ov["stop_lon"]]
            updated += count
        else:
            missing.append(str(ov["stop_name"]))

    summary = {
        "updated": updated,
        "missing": missing,
        "total_overrides": len(overrides),
    }
    return stops_out, summary


# -----------------------------
# OSM graph construction
# -----------------------------


def _ensure_pbf_list(pbf_input: Union[str, List[str]]) -> List[str]:
    return [pbf_input] if isinstance(pbf_input, str) else list(pbf_input)


def _count_ways_in_pbf(pbf_path: str) -> int:
    print(f"[COUNT] Counting ways in {pbf_path}...")
    class WayCounter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.count = 0
        def way(self, w):
            self.count += 1
    counter = WayCounter()
    counter.apply_file(pbf_path)
    print(f"[COUNT] {pbf_path}: {counter.count} ways")
    return counter.count


def build_graphs_from_pbf(pbf_paths: Union[str, List[str]], modes: List[str], bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[Optional[GraphIndex], Optional[GraphIndex]]:
    pbf_list = _ensure_pbf_list(pbf_paths)

    print("Counting OSM ways for progress tracking...")
    total_ways = 0
    for p in pbf_list:
        total_ways += _count_ways_in_pbf(p)
    print(f"Total ways across PBFs: {total_ways}")
    
    road_idx = None
    rail_idx = None

    # Build road graph with progress
    if "road" in modes:
        print("Building road graph...")
        road_builder = _WayGraphBuilder(mode="road", bbox=bbox)
        with tqdm.tqdm(total=total_ways, desc="Road graph", unit="way") as pbar:
            class ProgressWrapper(osmium.SimpleHandler):
                def __init__(self, handler, pbar):
                    super().__init__()
                    self.handler = handler
                    self.pbar = pbar
                def way(self, w):
                    self.handler.way(w)
                    self.pbar.update(1)
            wrapper = ProgressWrapper(road_builder, pbar)
            for pbf in pbf_list:
                wrapper.apply_file(pbf, locations=True)
        road_idx = graph_to_index(road_builder.G)

    # Build rail graph with progress
    if "rail" in modes:
        print("Building rail graph...")
        rail_builder = _WayGraphBuilder(mode="rail", bbox=bbox)
        with tqdm.tqdm(total=total_ways, desc="Rail graph", unit="way") as pbar:
            class ProgressWrapper2(osmium.SimpleHandler):
                def __init__(self, handler, pbar):
                    super().__init__()
                    self.handler = handler
                    self.pbar = pbar
                def way(self, w):
                    self.handler.way(w)
                    self.pbar.update(1)
            wrapper = ProgressWrapper2(rail_builder, pbar)
            for pbf in pbf_list:
                wrapper.apply_file(pbf, locations=True)
        rail_idx = graph_to_index(rail_builder.G)

    return road_idx, rail_idx


def build_rail_graph_only(pbf_paths: Union[str, List[str]], bbox: Optional[Tuple[float, float, float, float]] = None) -> GraphIndex:
    """Build only the rail graph from PBFs (for Italy-wide OSM where we don't need roads)."""
    pbf_list = _ensure_pbf_list(pbf_paths)
    print("Counting OSM ways for progress tracking...")
    total_ways = sum(_count_ways_in_pbf(p) for p in pbf_list)
    print(f"Total ways across PBFs: {total_ways}")
    
    # Build rail graph with progress
    print("Building rail graph...")
    rail_builder = _WayGraphBuilder(mode="rail", bbox=bbox)
    with tqdm.tqdm(total=total_ways, desc="Rail graph", unit="way") as pbar:
        class ProgressWrapper(osmium.SimpleHandler):
            def __init__(self, handler, pbar):
                super().__init__()
                self.handler = handler
                self.pbar = pbar
            def way(self, w):
                self.handler.way(w)
                self.pbar.update(1)
        wrapper = ProgressWrapper(rail_builder, pbar)
        for pbf in pbf_list:
            wrapper.apply_file(pbf, locations=True)

    return graph_to_index(rail_builder.G)


def save_graph_cache(cache_path: str, road_idx: Optional[GraphIndex], rail_idx: Optional[GraphIndex]) -> None:
    """Persist graphs to disk for reuse."""
    payload = {
        "road": road_idx.graph if road_idx else None,
        "rail": rail_idx.graph if rail_idx else None,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_cache(cache_path: str, modes: List[str]) -> Tuple[Optional[GraphIndex], Optional[GraphIndex]]:
    """Load cached graphs and rebuild indices."""
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    road_graph = payload.get("road") if isinstance(payload, dict) else None
    rail_graph = payload.get("rail") if isinstance(payload, dict) else None

    road_idx = graph_to_index(road_graph) if road_graph is not None and "road" in modes else None
    rail_idx = graph_to_index(rail_graph) if rail_graph is not None and "rail" in modes else None
    return road_idx, rail_idx


# -----------------------------
# Routing and shapes creation
# -----------------------------


def _parse_maxspeed(raw: Optional[Union[str, int, float]]) -> Optional[float]:
    """Parse an OSM maxspeed tag into km/h (best-effort)."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"signals", "unposted", "variable"}:
        return None
    # Handle composite values like "130;110" by taking the first token
    s = re.split(r"[;|]", s)[0]
    mph = "mph" in s
    s = s.replace("km/h", "").replace("kph", "").replace("mph", "")
    match = re.search(r"\d+(?:\.\d+)?", s)
    if not match:
        return None
    val = float(match.group())
    if mph:
        val *= 1.60934
    return val


def _edge_weight(data: Dict[str, float], speed_pref: str) -> float:
    """Edge weight that biases rail routing using OSM maxspeed.

    speed_pref: "high" (prefer >=200 km/h), "low" (prefer <=200),
    "flex" (FA/FB/IC/EC tolerate both, slight preference to >160), or "any".
    """
    length = float(data.get("length", 1.0))
    maxspeed = data.get("maxspeed")

    # Treat missing maxspeed as low speed (e.g. 80 km/h) to avoid assuming untagged lines are fast
    ms_val = maxspeed if maxspeed is not None else 80.0

    if speed_pref == "high":
        # Strong preference for AV lines (>200 km/h)
        # We give a significant "bonus" (multiplier < 1) to high speed lines
        # so the router prefers them even if the path is longer.
        if ms_val >= 210:
            return length * 0.1  # Very cheap to travel on AV
        elif ms_val >= 160:
            return length * 1.5  # Moderate cost for fast regional/direct lines
        else:
            return length * 5.0  # High cost for slow lines

    elif speed_pref == "low":
        # Avoid AV lines
        if ms_val >= 200:
            return length * 10.0 # Strongly avoid AV
        return length

    elif speed_pref == "flex":
        # Prefer fast lines, but allow normal lines
        if ms_val >= 200:
             return length * 0.8
        elif ms_val >= 140:
             return length * 1.0
        else:
             return length * 1.5

    # Any / default
    return length


def _min_turn_angle_deg(coords: List[Tuple[float, float]]) -> float:
    """Return the smallest turn angle (in degrees) along a polyline of coords.

    Coords are (lat, lon) tuples. Returns 180 if fewer than 3 points.
    """
    if len(coords) < 3:
        return 180.0
    min_ang = 180.0
    for i in range(1, len(coords) - 1):
        ax, ay = coords[i - 1][1], coords[i - 1][0]  # lon, lat
        bx, by = coords[i][1], coords[i][0]
        cx, cy = coords[i + 1][1], coords[i + 1][0]
        v1x, v1y = ax - bx, ay - by
        v2x, v2y = cx - bx, cy - by
        n1 = (v1x ** 2 + v1y ** 2) ** 0.5
        n2 = (v2x ** 2 + v2y ** 2) ** 0.5
        if n1 == 0 or n2 == 0:
            continue
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))
        if ang < min_ang:
            min_ang = ang
    return min_ang


def _max_deviation_m(seg_pts: List[Tuple[float, float]], a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Max lateral deviation from straight line A->B using equirectangular approximation."""
    if len(seg_pts) < 2:
        return 0.0
    to_rad = math.radians
    R = 6371000.0
    lat1 = to_rad(a_lat)
    lon1 = to_rad(a_lon)
    lat2 = to_rad(b_lat)
    lon2 = to_rad(b_lon)
    x2 = (lon2 - lon1) * math.cos((lat1 + lat2) * 0.5)
    y2 = (lat2 - lat1)
    norm = math.hypot(x2, y2)
    if norm == 0:
        return 0.0
    max_dev = 0.0
    for lat, lon in seg_pts:
        x = (to_rad(lon) - lon1) * math.cos((lat1 + lat2) * 0.5)
        y = to_rad(lat) - lat1
        cross = abs(x * y2 - y * x2) / norm
        d = cross * R
        if d > max_dev:
            max_dev = d
    return max_dev


def nearest_node(idx: GraphIndex, lat: float, lon: float) -> Optional[int]:
    if idx.coords.shape[0] == 0:
        return None
    dist, pos = idx.kdtree.query([lat, lon], k=1)
    return int(idx.node_ids[pos])


def candidate_nodes(
    idx: GraphIndex,
    lat: float,
    lon: float,
    k: int = 5,
    max_distance_m: float = 600.0,
) -> List[int]:
    """Return up to k nearest graph nodes within a distance threshold."""
    if idx.coords.shape[0] == 0:
        return []

    k = min(k, idx.coords.shape[0])
    dists, poses = idx.kdtree.query([lat, lon], k=k)

    # Normalize outputs when k==1
    if k == 1:
        dists = [dists]
        poses = [poses]

    out: List[int] = []
    for d, p in zip(dists, poses):
        if np.isfinite(d):
            # kdtree is in degrees; convert approx to meters using geodesic
            # (More precise than degree scaling when near poles)
            node_id = int(idx.node_ids[p])
            plat = idx.coords[p, 0]
            plon = idx.coords[p, 1]
            d_m = geodesic_m(lat, lon, plat, plon)
            if d_m <= max_distance_m:
                out.append(node_id)
    return out


def build_trip_shape_points(
    trip_id: str,
    stop_ids: List[str],
    stop_lookup: Dict[str, Tuple[float, float]],
    graph_idx: GraphIndex,
    seg_cache: Dict[Tuple[str, str, str], List[Tuple[float, float]]],
    event_queue: Optional[queue.Queue] = None,
    mode: str = "road",
    speed_pref: str = "any",
) -> List[Tuple[float, float]]:
    """Return a list of (lat, lon) points composing the whole trip shape."""
    path_points: List[Tuple[float, float]] = []

    # Precompute candidate nearest nodes for each stop (multi-choice to avoid wrong track)
    nn_for_stop: Dict[str, List[int]] = {}
    for idx_stop, sid in enumerate(stop_ids):
        lat, lon = stop_lookup[sid]
        # Broader search at first/last stops to allow choosing the correct track within big stations
        if idx_stop == 0 or idx_stop == len(stop_ids) - 1:
            nn_for_stop[sid] = candidate_nodes(graph_idx, lat, lon, k=12, max_distance_m=900.0)
        else:
            nn_for_stop[sid] = candidate_nodes(graph_idx, lat, lon, k=5, max_distance_m=600.0)

    for i in range(len(stop_ids) - 1):
        a, b = stop_ids[i], stop_ids[i + 1]
        # Skip if same stop repeats
        if a == b:
            continue

        cache_key = (a, b, speed_pref)
        if cache_key in seg_cache:
            seg_pts = seg_cache[cache_key]
        else:
            a_lat, a_lon = stop_lookup[a]
            b_lat, b_lon = stop_lookup[b]
            cand_a = nn_for_stop.get(a, [])
            cand_b = nn_for_stop.get(b, [])
            weight_fn = None if mode == "road" else (lambda u, v, data: _edge_weight(data, speed_pref))
            direct = geodesic_m(a_lat, a_lon, b_lat, b_lon)

            best_len = float("inf")
            best_pair: Optional[Tuple[int, int]] = None

            def try_pairs(ca: List[int], cb: List[int], max_detour_factor: float = 3.0) -> None:
                nonlocal best_len, best_pair
                for na in ca:
                    for nb in cb:
                        try:
                            node_path = nx.shortest_path(graph_idx.graph, na, nb, weight=weight_fn or "length")
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        # Compute weighted length manually (needed when weight_fn is callable)
                        path_len = 0.0
                        coords_tmp: List[Tuple[float, float]] = []
                        for u, v in zip(node_path[:-1], node_path[1:]):
                            data = graph_idx.graph[u][v]
                            w = weight_fn(u, v, data) if weight_fn else float(data.get("length", 1.0))
                            path_len += w
                        for n in node_path:
                            coords_tmp.append((graph_idx.graph.nodes[n]["lat"], graph_idx.graph.nodes[n]["lon"]))
                        min_turn = _min_turn_angle_deg(coords_tmp)
                        # Skip paths with unrealistically sharp turns (<25 deg)
                        if min_turn < 25.0:
                            continue
                        ratio = path_len / max(direct, 1.0)
                        if path_len < best_len and ratio <= max_detour_factor:
                            best_len = path_len
                            best_pair = (na, nb)

            # Try combinations of nearest nodes to avoid wrong-track snapping
            if cand_a and cand_b:
                try_pairs(cand_a, cand_b, max_detour_factor=2.5)
                # If nothing feasible, relax search for terminal segments to allow track change
                if best_pair is None and (i == 0 or i == len(stop_ids) - 2):
                    extra_a = cand_a or candidate_nodes(graph_idx, a_lat, a_lon, k=15, max_distance_m=1500.0)
                    extra_b = cand_b or candidate_nodes(graph_idx, b_lat, b_lon, k=15, max_distance_m=1500.0)
                    try_pairs(extra_a, extra_b, max_detour_factor=3.0)

            seg_pts: List[Tuple[float, float]]
            detour_ok = best_len < float("inf") and (best_len <= direct * 3.0 + 2000)
            if best_pair and detour_ok:
                try:
                    node_path = nx.shortest_path(graph_idx.graph, best_pair[0], best_pair[1], weight=weight_fn or "length")
                    seg_pts = [(a_lat, a_lon)]
                    for n in node_path:
                        nlat = graph_idx.graph.nodes[n]["lat"]
                        nlon = graph_idx.graph.nodes[n]["lon"]
                        if not seg_pts or (seg_pts[-1][0] != nlat or seg_pts[-1][1] != nlon):
                            seg_pts.append((nlat, nlon))
                    if seg_pts[-1] != (b_lat, b_lon):
                        seg_pts.append((b_lat, b_lon))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    seg_pts = _straight_segment(a_lat, a_lon, b_lat, b_lon)
            else:
                seg_pts = _straight_segment(a_lat, a_lon, b_lat, b_lon)

            # Post-check: if segment bends too far from straight line, fallback to straight
            if seg_pts:
                max_dev = _max_deviation_m(seg_pts, a_lat, a_lon, b_lat, b_lon)
                seg_len = 0.0
                for (p1_lat, p1_lon), (p2_lat, p2_lon) in zip(seg_pts[:-1], seg_pts[1:]):
                    seg_len += geodesic_m(p1_lat, p1_lon, p2_lat, p2_lon)
                if max_dev > 350.0 and seg_len > direct * 1.25:
                    seg_pts = _straight_segment(a_lat, a_lon, b_lat, b_lon)

            # Deduplicate consecutive identical points
            dedup = []
            for p in seg_pts:
                if not dedup or dedup[-1] != p:
                    dedup.append(p)
            seg_pts = dedup
            seg_cache[cache_key] = seg_pts

        # Append to global path; avoid duplicating joint point
        if not path_points:
            path_points.extend(seg_pts)
        else:
            if path_points[-1] == seg_pts[0]:
                path_points.extend(seg_pts[1:])
            else:
                path_points.extend(seg_pts)
        
        # Emit segment for real-time visualization
        if event_queue is not None and len(seg_pts) >= 2:
            try:
                event_queue.put_nowait({
                    'type': 'trip_segment',
                    'trip_id': trip_id,
                    'mode': mode,
                    'segment_index': i,
                    'total_segments': len(stop_ids) - 1,
                    'segment_coords': seg_pts,
                    'cumulative_points': len(path_points)
                })
            except queue.Full:
                pass

    # Emit completed trip shape for visualization
    if event_queue is not None and len(path_points) >= 2:
        try:
            event_queue.put_nowait({
                'type': 'trip_shape',
                'trip_id': trip_id,
                'mode': mode,
                'coords': path_points
            })
            # Also print to terminal for visibility
            print(f"[TRIP] {trip_id} ({mode}) - {len(path_points)} points", flush=True)
        except queue.Full:
            pass

    return path_points


def _straight_segment(a_lat: float, a_lon: float, b_lat: float, b_lon: float, n: int = 10) -> List[Tuple[float, float]]:
    """Interpolate a straight segment between two points with n steps (including ends)."""
    lats = np.linspace(a_lat, b_lat, n)
    lons = np.linspace(a_lon, b_lon, n)
    return list(zip(lats.tolist(), lons.tolist()))

def _simplify_polyline(pts: List[Tuple[float, float]], tolerance_m: float) -> List[Tuple[float, float]]:
    """Simplify polyline using Douglas-Peucker in meters via Web Mercator projection."""
    if len(pts) < 3 or tolerance_m <= 0:
        return pts
    # Transform to meters (x=lon,y=lat ordering for transformer)
    xs, ys = [], []
    for lat, lon in pts:
        x, y = _to_merc.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    line = LineString(zip(xs, ys))
    simplified = line.simplify(tolerance_m, preserve_topology=False)
    out: List[Tuple[float, float]] = []
    for x, y in simplified.coords:
        lon, lat = _to_wgs.transform(x, y)
        out.append((lat, lon))
    # Ensure we keep original endpoints exactly
    if out and (out[0] != pts[0]):
        out[0] = pts[0]
    if out and (out[-1] != pts[-1]):
        out[-1] = pts[-1]
    return out

def _dedup_close_pts(pts: List[Tuple[float, float]], min_gap_m: float = 0.5) -> List[Tuple[float, float]]:
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for lat, lon in pts[1:]:
        plat, plon = out[-1]
        if geodesic_m(plat, plon, lat, lon) >= min_gap_m:
            out.append((lat, lon))
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out

def _shape_signature(pts: List[Tuple[float, float]], round_decimals: int = 6) -> str:
    """Return a stable signature string for a polyline by rounding coords and hashing.
    We avoid importing hashlib to keep things lightweight; use a simple rolling hash.
    """
    if not pts:
        return ""
    # Build a compact string like 'lat,lon;lat,lon;...'
    s_parts = []
    for lat, lon in pts:
        s_parts.append(f"{round(lat, round_decimals):.{round_decimals}f},{round(lon, round_decimals):.{round_decimals}f}")
    s = ";".join(s_parts)
    # Simple FNV-1a 64-bit
    fnv = 1469598103934665603
    for ch in s.encode("utf-8"):
        fnv ^= ch
        fnv = (fnv * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"S{fnv:016x}"


def shape_points_to_rows(shape_id: str, pts: List[Tuple[float, float]]) -> List[Tuple[str, float, float, int, float]]:
    rows: List[Tuple[str, float, float, int, float]] = []
    cum = 0.0
    seq = 1
    prev = None
    for lat, lon in pts:
        if prev is not None:
            cum += geodesic_m(prev[0], prev[1], lat, lon)
        rows.append((shape_id, lat, lon, seq, cum))
        prev = (lat, lon)
        seq += 1
    return rows


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Rebuild GTFS shapes from OSM and trips")
    parser.add_argument("--gtfs", default="trgtfs", help="Path to GTFS directory (unpacked)")
    parser.add_argument("--osm", nargs="+", default=["lazio.osm.pbf"], help="Paths to one or more OSM PBF files")
    parser.add_argument("--modes", default="both", choices=["road", "rail", "both"], help="Which graphs to build (road, rail, or both)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just report")
    parser.add_argument("--max-trips", type=int, default=None, help="Limit number of trips processed (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--tolerance-road", type=float, default=5.0, help="Simplification tolerance in meters for road routes")
    parser.add_argument("--tolerance-rail", type=float, default=3.0, help="Simplification tolerance in meters for rail routes")
    parser.add_argument("--round-decimals", type=int, default=6, help="Decimals for rounding when computing shape signature")
    parser.add_argument("--with-viewer", action="store_true", help="Start web viewer in browser (opens http://127.0.0.1:1890)")
    parser.add_argument("--load-graphs", help="Path to a previously saved graph cache to skip rebuilding")
    parser.add_argument("--save-graphs", help="Path to save the built graphs for reuse")
    parser.add_argument("--normalize-rail-stops", action="store_true", help="Snap rail stops to OSM before building shapes")
    parser.add_argument("--normalize-rail-threshold", type=float, default=600.0, help="Max meters to move a rail stop when normalizing")
    parser.add_argument("--apply-stop-overrides", action="store_true", help="Apply manual stop coordinate overrides from CSV")
    parser.add_argument(
        "--stop-overrides-path",
        default=os.path.join(os.path.dirname(__file__) or ".", "stop_overrides.csv"),
        help="CSV with columns stop_name, stop_lat, stop_lon for manual overrides",
    )
    parser.add_argument("--zip-output", nargs="?", const="", help="Zip the GTFS directory after processing; optional custom zip path")
    args = parser.parse_args()

    # Start viewer if requested
    if args.with_viewer:
        import subprocess
        import webbrowser
        import time
        
        # Start viewer.py in background
        viewer_proc = subprocess.Popen(
            [sys.executable, "viewer.py"],
            cwd=os.path.dirname(__file__) or ".",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Give it time to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open("http://127.0.0.1:1890")
        print("Viewer started at http://127.0.0.1:1890")
        print("Click 'Build Graphs (Live View)' to watch real-time processing!")
        print("")

    gtfs_dir = args.gtfs
    osm_pbf_paths = args.osm if isinstance(args.osm, list) else [args.osm]

    if not os.path.isdir(gtfs_dir):
        print(f"GTFS directory not found: {gtfs_dir}", file=sys.stderr)
        sys.exit(1)
    missing = [p for p in osm_pbf_paths if p and not os.path.exists(p)]
    # Only require OSM files when we are going to build graphs from scratch.
    if not args.load_graphs:
        if missing:
            print(f"OSM PBF not found: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
    else:
        # If a cache is provided, we skip the hard check; warn only when absent.
        if missing:
            print(f"[WARN] OSM PBF missing but load-graphs is provided; will rely on cache: {', '.join(missing)}", file=sys.stderr)

    print("[STEP] Loading GTFS...")
    stops, stop_times, trips, routes = load_gtfs(gtfs_dir)
    print("[STEP] Loaded GTFS")

    # Determine mode and speed preference for each route_id
    prefer_rail_only = args.modes == "rail"
    routes["mode"] = routes.apply(lambda r: detect_route_mode(r, prefer_rail_only), axis=1)
    routes["speed_pref"] = routes.apply(detect_speed_pref, axis=1)
    route_mode_map = dict(zip(routes["route_id"], routes["mode"]))
    route_speed_pref_map = dict(zip(routes["route_id"], routes["speed_pref"]))

    # Build trip -> ordered stop list
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])  # type: ignore[arg-type]
    trip_to_stops: Dict[str, List[str]] = defaultdict(list)
    for tid, group in stop_times_sorted.groupby("trip_id"):
        trip_to_stops[tid] = group["stop_id"].tolist()

    trip_route_map = dict(zip(trips["trip_id"], trips.get("route_id", pd.Series(dtype=str))))
    rail_stop_ids: set[str] = set()
    for tid, sids in trip_to_stops.items():
        route_id = trip_route_map.get(tid)
        if route_mode_map.get(route_id, "road") == "rail":
            rail_stop_ids.update(sids)

    # Compute tight bbox from all stops (pad ~0.25 deg)
    def bbox_from_stops(stops_df: pd.DataFrame) -> Optional[Tuple[float, float, float, float]]:
        if stops_df.empty:
            return None
        lats = stops_df["stop_lat"].values
        lons = stops_df["stop_lon"].values
        pad = 0.25
        return (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)

    initial_bbox = bbox_from_stops(stops)
    print(f"Computed bbox from stops: {initial_bbox}")

    # Parse modes
    modes = ["road", "rail"] if args.modes == "both" else [args.modes]

    # Build or load OSM graphs
    road_idx: Optional[GraphIndex]
    rail_idx: Optional[GraphIndex]
    # Default bbox before any normalization adjustments
    bbox = initial_bbox

    if args.load_graphs:
        try:
            road_idx, rail_idx = load_graph_cache(args.load_graphs, modes)
            print(f"Loaded graph cache from {args.load_graphs}")
        except Exception as e:
            print(f"Failed to load graph cache from {args.load_graphs}: {e}", file=sys.stderr)
            print("Falling back to building graphs from PBF...", flush=True)
            road_idx, rail_idx = build_graphs_from_pbf(osm_pbf_paths, modes=modes, bbox=bbox)
    else:
        print(f"[STEP] Building OSM graphs ({', '.join(modes)}) with bbox filter", flush=True)
        road_idx, rail_idx = build_graphs_from_pbf(osm_pbf_paths, modes=modes, bbox=bbox)
    
    if road_idx:
        print(f"Road graph: nodes={road_idx.graph.number_of_nodes()} edges={road_idx.graph.number_of_edges()}")
    if rail_idx:
        print(f"Rail graph: nodes={rail_idx.graph.number_of_nodes()} edges={rail_idx.graph.number_of_edges()}")

    if args.save_graphs:
        try:
            save_graph_cache(args.save_graphs, road_idx, rail_idx)
            print(f"Saved graph cache to {args.save_graphs}")
        except Exception as e:
            print(f"Warning: failed to save graph cache to {args.save_graphs}: {e}", file=sys.stderr)

    # Optionally normalize rail stops now that we have the rail graph (avoids re-reading PBF)
    norm_summary_final = None

    if args.normalize_rail_stops and rail_idx is not None:
        print("[STEP] Normalizing rail stops using rail graph nodes")
        stops, norm_summary = normalize_rail_stops_with_graph(
            stops,
            list(rail_stop_ids),
            rail_idx,
            max_distance_m=args.normalize_rail_threshold,
        )
        norm_summary_final = norm_summary
        print(
            "Rail stop normalization -> "
            f"updated {norm_summary['updated']}, "
            f"too_far {norm_summary['too_far']}, "
            f"skipped {norm_summary['skipped']}"
        )

        # Persist updated stop coordinates
        stops_path = os.path.join(gtfs_dir, "stops.txt")
        stops_backup = stops_path + ".bak"
        if os.path.exists(stops_path):
            os.replace(stops_path, stops_backup)
            print(f"Backed up stops.txt to {stops_backup}")
        stops.to_csv(stops_path, index=False)
        print(f"Wrote normalized stops to {stops_path}")

        # Recompute bbox after normalization for downstream steps
        bbox = bbox_from_stops(stops)
        print(f"Recomputed bbox from normalized stops: {bbox}")
    else:
        bbox = initial_bbox

    # Apply manual stop overrides if requested (takes precedence over normalization)
    manual_override_summary = None
    if args.apply_stop_overrides:
        overrides_path = args.stop_overrides_path
        if not os.path.exists(overrides_path):
            print(f"[WARN] Stop overrides requested but file not found: {overrides_path}", file=sys.stderr)
        else:
            stops, manual_override_summary = apply_stop_overrides(stops, overrides_path)
            print(
                "[STEP] Applied manual stop overrides "
                f"(updated {manual_override_summary['updated']}/{manual_override_summary['total_overrides']} names; "
                f"missing {len(manual_override_summary['missing'])})"
            )
            if manual_override_summary["missing"]:
                print("  Missing stop names:", ", ".join(manual_override_summary["missing"]))

            stops_path = os.path.join(gtfs_dir, "stops.txt")
            pre_override_backup = stops_path + ".pre_overrides.bak"
            if os.path.exists(stops_path) and not os.path.exists(pre_override_backup):
                os.replace(stops_path, pre_override_backup)
                print(f"Backed up stops.txt to {pre_override_backup} before applying overrides")
            stops.to_csv(stops_path, index=False)
            print(f"Wrote stops.txt with manual overrides to {stops_path}")

            bbox = bbox_from_stops(stops)
            print(f"Recomputed bbox after manual overrides: {bbox}")

    # Build lookup after potential normalization
    stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops.iterrows()}

    # Prepare caches per network

    # Caches
    seg_cache_road: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    seg_cache_rail: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}

    # Iterate trips and build shapes
    out_rows: List[Tuple[str, float, float, int, float]] = []
    processed = 0
    failed_trips = 0
    unique_shapes = 0

    # Caches to avoid recomputing identical stop sequences / shapes
    stopseq_to_shape: Dict[Tuple[str, ...], str] = {}
    shape_rows_cache: Dict[str, List[Tuple[str, float, float, int, float]]] = {}
    shapes_emitted: set[str] = set()

    # To update trips.txt -> use shape_id = trip_id
    trips_updated = trips.copy()
    if "shape_id" not in trips_updated.columns:
        trips_updated["shape_id"] = ""

    # Map signature -> shape_id (dedupe)
    sig_to_shape: Dict[str, str] = {}
    # Optional mapping for summary
    trip_to_shape: List[Tuple[str, str, str, str]] = []  # (trip_id, route_id, mode, shape_id)

    iterator = trips.iterrows()
    if not args.verbose:
        iterator = tqdm.tqdm(list(trips.iterrows()), desc="Trips", unit="trip")

    for _, trip_row in iterator:
        trip_id = trip_row["trip_id"]
        route_id = trip_row.get("route_id")
        mode = route_mode_map.get(route_id, "road")
        speed_pref = route_speed_pref_map.get(route_id, "any") if mode == "rail" else "any"

        stops_list = trip_to_stops.get(trip_id)
        if not stops_list or len(stops_list) < 2:
            failed_trips += 1
            continue

        stop_seq = tuple(stops_list)

        # Fast-path reuse: identical stop sequence already mapped to a shape
        if stop_seq in stopseq_to_shape:
            shape_id = stopseq_to_shape[stop_seq]
            trips_updated.loc[trips_updated["trip_id"] == trip_id, "shape_id"] = shape_id
            trip_to_shape.append((trip_id, route_id or "", mode, shape_id))
            if shape_id in shape_rows_cache and shape_id not in shapes_emitted:
                out_rows.extend(shape_rows_cache[shape_id])
                shapes_emitted.add(shape_id)
            processed += 1
            continue

        if mode == "road":
            idx = road_idx
        else:
            idx = rail_idx
        
        if idx is None:
            # Graph for this mode was not built
            continue

        seg_cache = seg_cache_road if mode == "road" else seg_cache_rail
        
        # Print current trip being processed
        print(f"[PROCESSING] Trip {processed+1}/{len(trips)}: {trip_id} ({mode}, {len(stops_list)} stops)", flush=True)

        try:
            pts = build_trip_shape_points(trip_id, stops_list, stop_lookup, idx, seg_cache, speed_pref=speed_pref)
            if len(pts) < 2:
                failed_trips += 1
                continue
            # Simplify and clean
            tol = args.tolerance_road if mode == "road" else args.tolerance_rail
            pts = _simplify_polyline(pts, tolerance_m=tol)
            pts = _dedup_close_pts(pts, min_gap_m=0.5)
            # Compute signature and assign/reuse shape_id
            sig = _shape_signature(pts, round_decimals=args.round_decimals)
            if sig not in sig_to_shape:
                sig_to_shape[sig] = sig  # use signature as shape_id
                unique_shapes += 1
            shape_id = sig_to_shape[sig]

            if shape_id not in shape_rows_cache:
                rows = shape_points_to_rows(shape_id, pts)
                shape_rows_cache[shape_id] = rows
            stopseq_to_shape[stop_seq] = shape_id

            if shape_id not in shapes_emitted:
                out_rows.extend(shape_rows_cache[shape_id])
                shapes_emitted.add(shape_id)

            trips_updated.loc[trips_updated["trip_id"] == trip_id, "shape_id"] = shape_id
            trip_to_shape.append((trip_id, route_id or "", mode, shape_id))
            processed += 1
        except Exception as e:
            if args.verbose:
                print(f"Trip {trip_id} failed: {e}")
            failed_trips += 1

        if args.max_trips is not None and processed >= args.max_trips:
            break

    if args.dry_run:
        print(
            f"Dry-run: trips={processed}, unique_shapes={unique_shapes}, total_shape_points={len(out_rows)}, failed_trips={failed_trips}"
        )
        return

    # Write shapes.txt
    shapes_path = os.path.join(gtfs_dir, "shapes.txt")
    backup_path = shapes_path + ".bak"
    if os.path.exists(shapes_path):
        os.replace(shapes_path, backup_path)
        print(f"Backed up existing shapes.txt to {backup_path}")

    with open(shapes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"])
        for row in out_rows:
            writer.writerow(row)
    print(f"Wrote {shapes_path} with {len(out_rows)} points across {processed} trips and {unique_shapes} unique shapes")

    # Update trips.txt with shape_id = trip_id
    trips_path = os.path.join(gtfs_dir, "trips.txt")
    trips_backup = trips_path + ".bak"
    os.replace(trips_path, trips_backup)
    trips_updated.to_csv(trips_path, index=False)
    print(f"Updated trips.txt (backup at {trips_backup}); assigned shared shape_id for identical shapes")

    # Write mapping for reference
    map_path = os.path.join(gtfs_dir, "shape_id_map.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trip_id", "route_id", "mode", "shape_id"])
        w.writerows(trip_to_shape)
    print(f"Wrote {map_path} mapping trips to shape_id")

    if args.normalize_rail_stops and norm_summary_final is not None:
        print(
            "[SUMMARY] Rail stop normalization: "
            f"updated {norm_summary_final['updated']}, "
            f"too_far {norm_summary_final['too_far']}, "
            f"skipped {norm_summary_final['skipped']}, "
            f"threshold {args.normalize_rail_threshold} m"
        )

    # Optional zipping of output GTFS directory
    if args.zip_output is not None:
        zip_target = args.zip_output
        if zip_target == "":
            base_name = os.path.abspath(gtfs_dir)
        else:
            base_name = os.path.abspath(os.path.splitext(zip_target)[0])
        zip_path = shutil.make_archive(base_name=base_name, format="zip", root_dir=gtfs_dir)
        print(f"[SUMMARY] Zipped GTFS to {zip_path}")


def build_graphs_for_viewer(osm_pbf: str, event_queue: queue.Queue, modes: List[str] = ["road", "rail"]):
    """Build OSM graphs with real-time event emission for visualization.
    
    This function is called by the viewer to build graphs and stream edges
    to the frontend as they're parsed.
    """
    import os
    
    if not os.path.exists(osm_pbf):
        event_queue.put({'type': 'error', 'message': f'OSM PBF not found: {osm_pbf}'})
        return
    
    # Load GTFS to compute bbox
    gtfs_dir = "trgtfs"
    if not os.path.isdir(gtfs_dir):
        event_queue.put({'type': 'error', 'message': f'GTFS directory not found: {gtfs_dir}'})
        return
    
    try:
        stops, stop_times, trips, routes = load_gtfs(gtfs_dir)
    except Exception as e:
        event_queue.put({'type': 'error', 'message': f'Failed to load GTFS: {e}'})
        return
    
    # Compute bbox from all stops (with padding)
    lats = stops["stop_lat"].values
    lons = stops["stop_lon"].values
    pad = 0.25
    bbox = (lons.min() - pad, lats.min() - pad, lons.max() + pad, lats.max() + pad)
    
    event_queue.put({
        'type': 'info',
        'message': f'Building graphs ({", ".join(modes)}) (bbox: {bbox})',
        'bbox': bbox
    })
    
    # Build graphs with event emission
    try:
        # Count ways first
        class WayCounter(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.count = 0
            def way(self, w):
                self.count += 1
        
        counter = WayCounter()
        counter.apply_file(osm_pbf)
        total_ways = counter.count
        
        road_builder = None
        rail_builder = None

        if "road" in modes:
            event_queue.put({
                'type': 'progress',
                'region': 'all',
                'phase': 'road',
                'message': f'Building road graph ({total_ways} ways)',
                'total': total_ways
            })
            
            # Build road graph
            road_builder = _WayGraphBuilder(mode="road", bbox=bbox, event_queue=event_queue, region="all")
            road_builder.apply_file(osm_pbf, locations=True)
            
            event_queue.put({
                'type': 'graph_complete',
                'region': 'all',
                'mode': 'road',
                'nodes': road_builder.G.number_of_nodes(),
                'edges': road_builder.G.number_of_edges()
            })
        
        if "rail" in modes:
            event_queue.put({
                'type': 'progress',
                'region': 'all',
                'phase': 'rail',
                'message': f'Building rail graph',
                'total': total_ways
            })
            
            # Build rail graph
            rail_builder = _WayGraphBuilder(mode="rail", bbox=bbox, event_queue=event_queue, region="all")
            rail_builder.apply_file(osm_pbf, locations=True)
            
            event_queue.put({
                'type': 'graph_complete',
                'region': 'all',
                'mode': 'rail',
                'nodes': rail_builder.G.number_of_nodes(),
                'edges': rail_builder.G.number_of_edges()
            })
        
    except Exception as e:
        event_queue.put({'type': 'error', 'message': f'Graph building failed: {e}'})
        return
    
    event_queue.put({
        'type': 'phase_complete',
        'phase': 'graphs',
        'message': 'Graph building complete, starting trip shape generation...'
    })
    
    # Now build trip shapes with visualization
    build_trip_shapes_for_viewer(gtfs_dir, stops, stop_times, trips, routes, 
                                  road_builder.G if road_builder else nx.DiGraph(), 
                                  rail_builder.G if rail_builder else nx.DiGraph(),
                                  bbox, modes, event_queue)
    
    event_queue.put({
        'type': 'complete',
        'message': 'All processing complete!'
    })


def build_trip_shapes_for_viewer(gtfs_dir: str, stops: pd.DataFrame, stop_times: pd.DataFrame, 
                                 trips: pd.DataFrame, routes: pd.DataFrame,
                                 road_graph: nx.DiGraph, rail_graph: nx.DiGraph,
                                 bbox: Tuple[float, float, float, float],
                                 modes: List[str],
                                 event_queue: queue.Queue):
    """Build trip shapes and emit them for real-time visualization."""
    
    road_idx = graph_to_index(road_graph)
    rail_idx = graph_to_index(rail_graph)
    
    # Determine mode for each route
    prefer_rail_only = ("rail" in modes) and ("road" not in modes)
    routes["mode"] = routes.apply(lambda r: detect_route_mode(r, prefer_rail_only), axis=1)
    routes["speed_pref"] = routes.apply(detect_speed_pref, axis=1)
    route_mode_map = dict(zip(routes["route_id"], routes["mode"]))
    route_speed_pref_map = dict(zip(routes["route_id"], routes["speed_pref"]))
    
    # Build trip -> stops
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])  # type: ignore[arg-type]
    trip_to_stops: Dict[str, List[str]] = defaultdict(list)
    for tid, group in stop_times_sorted.groupby("trip_id"):
        trip_to_stops[tid] = group["stop_id"].tolist()
    
    stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops.iterrows()}
    
    event_queue.put({
        'type': 'info',
        'message': f'Processing {len(trips)} trips'
    })
    
    # Process trips
    seg_cache_road: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    seg_cache_rail: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    
    processed = 0
    for _, trip_row in trips.iterrows():
        trip_id = trip_row["trip_id"]
        route_id = trip_row.get("route_id")
        mode = route_mode_map.get(route_id, "road")
        speed_pref = route_speed_pref_map.get(route_id, "any") if mode == "rail" else "any"
        
        stops_list = trip_to_stops.get(trip_id)
        if not stops_list or len(stops_list) < 2:
            continue
        
        if mode == "road":
            idx = road_idx
        else:
            idx = rail_idx
        seg_cache = seg_cache_road if mode == "road" else seg_cache_rail
        
        try:
            print(f"\n[PROCESSING] Trip {processed+1}/{len(trips)}: {trip_id} ({mode})", flush=True)
            pts = build_trip_shape_points(trip_id, stops_list, stop_lookup, idx, seg_cache, event_queue, mode, speed_pref)
            if len(pts) >= 2:
                processed += 1
                if processed % 10 == 0:
                    event_queue.put({
                        'type': 'progress',
                        'phase': 'trips',
                        'message': f'Processed {processed}/{len(trips)} trips',
                        'count': processed
                    })
                    print(f"[PROGRESS] {processed}/{len(trips)} trips completed", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to process {trip_id}: {e}", flush=True)
            pass
    
    event_queue.put({
        'type': 'phase_complete',
        'phase': 'trips',
        'message': f'Trip processing complete: {processed} trips'
    })


if __name__ == "__main__":
    main()