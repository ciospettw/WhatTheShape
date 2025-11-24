import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import queue

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

    RAIL_TYPES = {
        "rail",
        "light_rail",
        "subway",
        "tram",
        "monorail",
    }

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
            if rw is None or rw not in self.RAIL_TYPES:
                return
            # Assume rails are bidirectional unless tagged otherwise
            oneway = False

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
            self.G.add_edge(u, v, length=length)
            if not oneway:
                self.G.add_edge(v, u, length=length)
            
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


# -----------------------------
# OSM graph construction
# -----------------------------


def build_graphs_from_pbf(pbf_path: str, modes: List[str], bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[Optional[GraphIndex], Optional[GraphIndex]]:
    # First pass: count ways for progress bar
    class WayCounter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.count = 0
        def way(self, w):
            self.count += 1
    
    print("Counting OSM ways for progress tracking...")
    counter = WayCounter()
    counter.apply_file(pbf_path)
    total_ways = counter.count
    print(f"Total ways in PBF: {total_ways}")
    
    road_idx = None
    rail_idx = None

    def to_index(G: nx.DiGraph) -> GraphIndex:
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
            wrapper.apply_file(pbf_path, locations=True)
        road_idx = to_index(road_builder.G)

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
            wrapper.apply_file(pbf_path, locations=True)
        rail_idx = to_index(rail_builder.G)

    return road_idx, rail_idx
    return road_idx, rail_idx


def build_rail_graph_only(pbf_path: str, bbox: Optional[Tuple[float, float, float, float]] = None) -> GraphIndex:
    """Build only the rail graph from PBF (for Italy-wide OSM where we don't need roads)."""
    # First pass: count ways for progress bar
    class WayCounter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.count = 0
        def way(self, w):
            self.count += 1
    
    print("Counting OSM ways for progress tracking...")
    counter = WayCounter()
    counter.apply_file(pbf_path)
    total_ways = counter.count
    print(f"Total ways in PBF: {total_ways}")
    
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
        wrapper.apply_file(pbf_path, locations=True)

    def to_index(G: nx.DiGraph) -> GraphIndex:
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

    return to_index(rail_builder.G)


# -----------------------------
# Routing and shapes creation
# -----------------------------


def nearest_node(idx: GraphIndex, lat: float, lon: float) -> Optional[int]:
    if idx.coords.shape[0] == 0:
        return None
    dist, pos = idx.kdtree.query([lat, lon], k=1)
    return int(idx.node_ids[pos])


def build_trip_shape_points(
    trip_id: str,
    stop_ids: List[str],
    stop_lookup: Dict[str, Tuple[float, float]],
    graph_idx: GraphIndex,
    seg_cache: Dict[Tuple[str, str], List[Tuple[float, float]]],
    event_queue: Optional[queue.Queue] = None,
    mode: str = "road",
) -> List[Tuple[float, float]]:
    """Return a list of (lat, lon) points composing the whole trip shape."""
    path_points: List[Tuple[float, float]] = []

    # Precompute nearest nodes for stops for this mode
    nn_for_stop: Dict[str, Optional[int]] = {}
    for sid in stop_ids:
        lat, lon = stop_lookup[sid]
        nn_for_stop[sid] = nearest_node(graph_idx, lat, lon)

    for i in range(len(stop_ids) - 1):
        a, b = stop_ids[i], stop_ids[i + 1]
        # Skip if same stop repeats
        if a == b:
            continue

        cache_key = (a, b)
        if cache_key in seg_cache:
            seg_pts = seg_cache[cache_key]
        else:
            a_node = nn_for_stop[a]
            b_node = nn_for_stop[b]
            a_lat, a_lon = stop_lookup[a]
            b_lat, b_lon = stop_lookup[b]

            seg_pts: List[Tuple[float, float]] = []
            if a_node is not None and b_node is not None and a_node in graph_idx.graph and b_node in graph_idx.graph:
                try:
                    node_path = nx.shortest_path(graph_idx.graph, a_node, b_node, weight="length")
                    # Build polyline from node coordinates
                    # Ensure we include the exact stop coordinate at ends
                    seg_pts.append((a_lat, a_lon))
                    for n in node_path:
                        nlat = graph_idx.graph.nodes[n]["lat"]
                        nlon = graph_idx.graph.nodes[n]["lon"]
                        if not seg_pts or (seg_pts[-1][0] != nlat or seg_pts[-1][1] != nlon):
                            seg_pts.append((nlat, nlon))
                    if seg_pts[-1] != (b_lat, b_lon):
                        seg_pts.append((b_lat, b_lon))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Fallback: straight line interpolation
                    seg_pts = _straight_segment(a_lat, a_lon, b_lat, b_lon)
            else:
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
    parser.add_argument("--osm", default="lazio.osm.pbf", help="Path to OSM PBF file")
    parser.add_argument("--modes", default="both", choices=["road", "rail", "both"], help="Which graphs to build (road, rail, or both)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just report")
    parser.add_argument("--max-trips", type=int, default=None, help="Limit number of trips processed (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--tolerance-road", type=float, default=5.0, help="Simplification tolerance in meters for road routes")
    parser.add_argument("--tolerance-rail", type=float, default=3.0, help="Simplification tolerance in meters for rail routes")
    parser.add_argument("--round-decimals", type=int, default=6, help="Decimals for rounding when computing shape signature")
    parser.add_argument("--with-viewer", action="store_true", help="Start web viewer in browser (opens http://127.0.0.1:1890)")
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
    osm_pbf = args.osm

    if not os.path.isdir(gtfs_dir):
        print(f"GTFS directory not found: {gtfs_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(osm_pbf):
        print(f"OSM PBF not found: {osm_pbf}", file=sys.stderr)
        sys.exit(1)

    stops, stop_times, trips, routes = load_gtfs(gtfs_dir)

    # Determine mode for each route_id
    def route_mode(row) -> str:
        # Heuristics by ID/name first (override wrong route_type)
        text = " ".join([
            str(row.get("route_id", "")),
            str(row.get("route_short_name", "")),
            str(row.get("route_long_name", "")),
        ]).lower()

        bus_keywords = ["bus", "autobus", "pullman"]
        rail_keywords = [
            "rail", "train", "metro", "subway", "tram",
            "ferrovia", "ferrovi", "metropolitana"
        ]

        if any(k in text for k in bus_keywords):
            return "road"
        if any(k in text for k in rail_keywords):
            return "rail"

        # Fall back to route_type only if keywords are inconclusive
        try:
            rtype = int(row.get("route_type", "3"))
        except Exception:
            rtype = 3
        # GTFS: 0=tram, 1=subway, 2=rail, 3=bus
        if rtype == 3:
            return "road"
        if rtype in (0, 1, 2):
            return "rail"
        # Default to road if unknown
        return "road"

    routes["mode"] = routes.apply(route_mode, axis=1)
    route_mode_map = dict(zip(routes["route_id"], routes["mode"]))

    # Build trip -> ordered stop list
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])  # type: ignore[arg-type]
    trip_to_stops: Dict[str, List[str]] = defaultdict(list)
    for tid, group in stop_times_sorted.groupby("trip_id"):
        trip_to_stops[tid] = group["stop_id"].tolist()

    stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops.iterrows()}

    # Compute tight bbox from all stops (pad ~0.25 deg)
    def bbox_from_stops(stops_df: pd.DataFrame) -> Optional[Tuple[float, float, float, float]]:
        if stops_df.empty:
            return None
        lats = stops_df["stop_lat"].values
        lons = stops_df["stop_lon"].values
        pad = 0.25
        return (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)

    bbox = bbox_from_stops(stops)
    print(f"Computed bbox from stops: {bbox}")

    # Parse modes
    modes = ["road", "rail"] if args.modes == "both" else [args.modes]

    # Build OSM graphs
    print(f"Building OSM graphs ({', '.join(modes)}) with bbox filter", flush=True)
    road_idx, rail_idx = build_graphs_from_pbf(osm_pbf, modes=modes, bbox=bbox)
    
    if road_idx:
        print(f"Road graph: nodes={road_idx.graph.number_of_nodes()} edges={road_idx.graph.number_of_edges()}")
    if rail_idx:
        print(f"Rail graph: nodes={rail_idx.graph.number_of_nodes()} edges={rail_idx.graph.number_of_edges()}")

    # Prepare caches per network

    # Caches
    seg_cache_road: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    seg_cache_rail: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

    # Iterate trips and build shapes
    out_rows: List[Tuple[str, float, float, int, float]] = []
    processed = 0
    failed_trips = 0
    unique_shapes = 0

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

        stops_list = trip_to_stops.get(trip_id)
        if not stops_list or len(stops_list) < 2:
            # Not enough stops to build a shape
            failed_trips += 1
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
            pts = build_trip_shape_points(trip_id, stops_list, stop_lookup, idx, seg_cache)
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
            rows = shape_points_to_rows(shape_id, pts)
            out_rows.extend(rows)
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
                                  bbox, event_queue)
    
    event_queue.put({
        'type': 'complete',
        'message': 'All processing complete!'
    })


def build_trip_shapes_for_viewer(gtfs_dir: str, stops: pd.DataFrame, stop_times: pd.DataFrame, 
                                 trips: pd.DataFrame, routes: pd.DataFrame,
                                 road_graph: nx.DiGraph, rail_graph: nx.DiGraph,
                                 bbox: Tuple[float, float, float, float],
                                 event_queue: queue.Queue):
    """Build trip shapes and emit them for real-time visualization."""
    
    # Create graph indices
    def to_index(G: nx.DiGraph) -> GraphIndex:
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
    
    road_idx = to_index(road_graph)
    rail_idx = to_index(rail_graph)
    
    # Determine mode for each route
    def route_mode(row) -> str:
        text = " ".join([
            str(row.get("route_id", "")),
            str(row.get("route_short_name", "")),
            str(row.get("route_long_name", "")),
        ]).lower()
        
        bus_keywords = ["bus", "autobus", "pullman"]
        rail_keywords = ["rail", "train", "metro", "subway", "tram", "ferrovia", "ferrovi", "metropolitana"]
        
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
        return "road"
    
    routes["mode"] = routes.apply(route_mode, axis=1)
    route_mode_map = dict(zip(routes["route_id"], routes["mode"]))
    
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
    seg_cache_road: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    seg_cache_rail: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    
    processed = 0
    for _, trip_row in trips.iterrows():
        trip_id = trip_row["trip_id"]
        route_id = trip_row.get("route_id")
        mode = route_mode_map.get(route_id, "road")
        
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
            pts = build_trip_shape_points(trip_id, stops_list, stop_lookup, idx, seg_cache, event_queue, mode)
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