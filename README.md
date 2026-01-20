# GTFS Shapes Fixer

This tool rebuilds `shapes.txt` for a GTFS feed by map-matching trip stops to OpenStreetMap (OSM) data. It supports both road (bus) and rail (mainline train) networks.

## Features

-   **Automatic Mode Detection**: Infers whether a route is "road" or "rail" based on `route_type` and keywords in route names.
-   **Hybrid Graph Building**: Builds separate routing graphs for road and rail networks from one or more OSM PBF files (rail now restricted to `rail` + ferry links; no tram/metro/light_rail/monorail).
-   **Rail Stop Normalizer**: Optionally snaps rail GTFS stops to nearby OSM rail stops before shape generation to fix bad coordinates.
-   **Rail speed bias**: Uses OSM `maxspeed` to prefer tracks coerent with train class (AV, IC/EC, REG, etc.).
-   **Anti-detour & multi-candidate snapping**: For each stop pair, tries multiple nearby rail nodes, avoids huge detours, rejects unrealistically sharp turns.
-   **Ferry bridging**: Includes `route=ferry` segments in the rail graph to span sea gaps (es. Stretto di Messina).
-   **Shape Simplification**: Simplifies the resulting polylines to reduce file size while maintaining accuracy.
-   **Deduplication & reuse**: Assigns shared `shape_id`s to trips with identical geometries and skips recomputation when the stop sequence repeats.
-   **Visualizer**: Includes a web-based viewer to watch the graph building and shape generation process in real-time.

## Prerequisites

-   Python 3.9+
-   A valid GTFS feed (unpacked directory).
-   An OpenStreetMap PBF (OSM.PBF) file covering the region of the GTFS feed.

## Installation

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `osmium` dependencies separately if the pip install fails (e.g., `libosmium` on Linux).*

## Usage

### Basic Usage

To rebuild shapes for a GTFS feed using an OSM PBF file:

```bash
python main.py --gtfs /path/to/gtfs_dir --osm /path/to/region.osm.pbf
```

You can also pass multiple PBFs (they are merged while building the graphs):

```bash
python main.py --gtfs /path/to/gtfs_dir --osm /path/to/region.osm.pbf /path/to/extra.osm.pbf
```

This will:
1.  Load the GTFS data.
2.  Compute the bounding box of all stops.
3.  Build road and/or rail graphs from the PBF file within that bounding box.
4.  Process every trip in `trips.txt`, generating a shape.
5.  Write a new `shapes.txt` to the GTFS directory.
6.  Update `trips.txt` with the new `shape_id`s (backing up the original as `trips.txt.bak`).
7.  Generate a `shape_id_map.csv` in the GTFS directory, mapping each trip to its assigned shape ID.

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--gtfs` | Path to the unpacked GTFS directory. | `trgtfs` |
| `--osm` | Path(s) to one or more OSM PBF files. | `lazio.osm.pbf` |
| `--modes` | Which graphs to build: `road`, `rail`, or `both`. | `both` |
| `--dry-run` | Run without writing changes to disk. | `False` |
| `--max-trips` | Limit the number of trips to process (for testing). | `None` |
| `--tolerance-road` | Simplification tolerance (meters) for road shapes. | `5.0` |
| `--tolerance-rail` | Simplification tolerance (meters) for rail shapes. | `3.0` |
| `--with-viewer` | Launch the web visualizer. | `False` |
| `--load-graphs` | Load a previously saved road/rail graph cache instead of rebuilding. | `None` |
| `--save-graphs` | Save the built road/rail graphs for future runs. | `None` |
| `--normalize-rail-stops` | Snap rail GTFS stops to nearby OSM rail stops before building shapes. | `False` |
| `--normalize-rail-threshold` | Max distance in meters to move a rail stop when normalizing. | `600.0` |

### Selective Graph Building

If you only want to process rail trips (e.g., for a train-only feed or to save time):

```bash
python main.py --gtfs ./my_gtfs --osm ./italy.osm.pbf --modes rail
```

Any trips identified as "road" (bus) will be skipped.

### Visualizer

To watch the process in real-time:

```bash
python main.py --with-viewer
```

### Reusing built graphs

If you already built the road/rail graphs once, you can reuse them to skip parsing the PBF:

```bash
python main.py --gtfs ./my_gtfs --osm ./italy.osm.pbf --modes rail --save-graphs ./rail_graphs.gpickle
# Later
python main.py --gtfs ./my_gtfs --osm ./italy.osm.pbf --modes rail --load-graphs ./rail_graphs.gpickle
```

The cache stores road and rail graphs together; loading automatically pulls only the modes you request.

This will open a web browser at `http://127.0.0.1:1890`. Click "Build Graphs (Live View)" to start.

**Note**: The visualizer is a basic tool for debugging and watching progress. It is not highly optimized and may struggle with very large datasets. It is "not one of the bests", but it gets the job done for monitoring.

## Important Notes & Limitations

1.  **OSM Coverage**: You **MUST** provide an OSM PBF that covers the **entire** area of your GTFS feed. If the PBF is too small, stops outside the area will not be matched, and trips may fail or result in straight lines.
    -   *Tip*: Download a larger region (e.g., the whole country or municipality) from [Geofabrik](https://download.geofabrik.de/). The script automatically filters the graph to the bounding box of your stops, so using a large PBF is efficient.

2.  **Graph Connectivity**: The script assumes the OSM network is connected. If stops are far from any road/rail (e.g., bad stop coordinates or missing OSM data), the map matching may fail or produce straight lines between those stops.

3.  **Rail scope**: The rail graph now keeps only `rail` (mainline) and `route=ferry` links; tram/metro/light_rail/monorail are excluded to avoid wrong routings.

4.  **Rebuilding cache**: If you use a saved graph cache, rebuild it after updates that add attributes (e.g., `maxspeed`, ferry inclusion) to benefit from new routing bias.

5.  **Performance**:
    -   Building graphs from large PBFs can take time and memory (RAM), and may as well cause CPU strain.
    -   Processing thousands of trips can take a while, especially if they are road ones. Use `--max-trips` to test on a subset first.

6.  **Route Mode Inference**: The script uses a "smart" heuristic to determine if a route is **Road** (Bus) or **Rail** (Train/Tram/Subway).
    *   **Priority 1: Keywords**: It checks `route_id`, `route_short_name`, and `route_long_name` for keywords.
        *   *Road keywords*: "bus", "autobus", "pullman"
        *   *Rail keywords*: "rail", "train", "metro", "subway", "tram", "ferrovia", "metropolitana"
    *   **Priority 2: GTFS `route_type`**: If no keywords are found, it falls back to the standard GTFS `route_type` field.
        *   `3` = **Road** (Bus)
        *   `0` (Tram), `1` (Subway), `2` (Rail) = **Rail**
    *   **Default**: If neither matches, it defaults to **Road**.

    You can edit keywords to your liking in the `route_mode` function.

    *If your GTFS uses non-standard `route_type` values (e.g. extended types like 700 for bus) or lacks clear names, you may need to edit the `route_mode` function in `main.py`.*

## Troubleshooting

-   **"ModuleNotFoundError: No module named 'pandas'"**: Ensure you have activated your virtual environment and installed requirements.
-   **Straight lines in output**: This usually means the map matching failed for those segments. Check if:
    -   The OSM PBF covers that area.
    -   The stops are close enough to roads/rails.
    -   The correct `--modes` were enabled.
-   **Script crashes with MemoryError**: Try using a smaller PBF (cropped to your region) or a machine with more RAM. Unfortunately, for big areas such as entire regions or even entire nations, not much can be done to not make your computer crash at build time.

## License

CC-BY-NC-SA
@Ciospettw