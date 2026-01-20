from flask import Flask, send_from_directory, jsonify, Response, request
import csv
import os
import json
import queue
import threading

app = Flask(__name__)

# Allow overriding GTFS path via env (e.g. WHATTHESHAPE_GTFS=C:\\path\\to\\gtfs)
GTFS_DIR = os.environ.get('WHATTHESHAPE_GTFS', os.path.join(os.path.dirname(__file__), 'trgtfs'))

# Global queue for streaming graph building events
graph_event_queue = queue.Queue()
graph_building_active = False


@app.get('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'web'), 'index.html')


@app.get('/shapes')
def shapes_list():
    shapes_path = os.path.join(GTFS_DIR, 'shapes.txt')
    shapes = {}
    with open(shapes_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row['shape_id']
            lat = float(row['shape_pt_lat'])
            lon = float(row['shape_pt_lon'])
            seq = int(row['shape_pt_sequence'])
            shapes.setdefault(sid, []).append((seq, lat, lon))
    # sort and trim large polylines optionally
    out = []
    for sid, pts in shapes.items():
        pts.sort(key=lambda x: x[0])
        coords = [[lat, lon] for _, lat, lon in pts]
        out.append({'shape_id': sid, 'coords': coords})
    return jsonify(out)


@app.get('/stops')
def stops():
    stops_path = os.path.join(GTFS_DIR, 'stops.txt')
    out = []
    with open(stops_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('location_type', '0') and str(row.get('location_type', '0')) != '0':
                continue
            try:
                out.append({
                    'stop_id': row['stop_id'],
                    'name': row.get('stop_name', ''),
                    'lat': float(row['stop_lat']),
                    'lon': float(row['stop_lon'])
                })
            except Exception:
                pass
    return jsonify(out)


@app.get('/graph-events')
def graph_events():
    """Server-Sent Events stream for real-time graph building."""
    def generate():
        # Send initial state
        yield f"data: {json.dumps({'type': 'status', 'active': graph_building_active})}\n\n"
        
        # Stream events from queue
        while True:
            try:
                event = graph_event_queue.get(timeout=30)
                if event is None:  # Sentinel to stop
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send keepalive ping
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.post('/start-graph-build')
def start_graph_build():
    """Trigger graph building in background thread."""
    global graph_building_active
    
    if graph_building_active:
        return jsonify({'error': 'Graph building already in progress'}), 409
    
    params = request.get_json() or {}
    osm_pbf = params.get('osm_pbf', 'lazio.osm.pbf')
    
    def build_graphs_thread():
        global graph_building_active
        graph_building_active = True
        
        try:
            # Import here to avoid circular dependency
            from main import build_graphs_for_viewer
            build_graphs_for_viewer(osm_pbf, graph_event_queue)
        except Exception as e:
            graph_event_queue.put({'type': 'error', 'message': str(e)})
        finally:
            graph_event_queue.put({'type': 'complete'})
            graph_event_queue.put(None)  # Sentinel
            graph_building_active = False
    
    thread = threading.Thread(target=build_graphs_thread, daemon=True)
    thread.start()
    
    return jsonify({'status': 'started'})


@app.get('/<path:filename>')
def assets(filename):
    # serve static assets from web/
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'web'), filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=1890, debug=False, threaded=True)
