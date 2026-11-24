"""
Count vehicles on a Pixels Guyana HLS camera feed and log the results to SQLite.

Usage:
    python count_vehicles.py --cam cam1
    python count_vehicles.py --cam cam2 --line 0.0,0.6,1.0,0.6 --show

Writes to vehicles.db:
  - occupancy: every --interval seconds, how many vehicles are visible in frame
  - crossings: one row per vehicle crossing the counting line (with direction)
"""
import argparse
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone

import cv2
import supervision as sv
from ultralytics import YOLO

import os
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("SSL_CERT_DIR", "/etc/ssl/certs")

STREAM_URL = "https://stream.pixelsgy.com/hls/{cam}/main_stream.m3u8"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def setup_db(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS occupancy (
            ts TEXT, camera TEXT, total INTEGER,
            car INTEGER, motorcycle INTEGER, bus INTEGER, truck INTEGER
        );
        CREATE TABLE IF NOT EXISTS crossings (
            ts TEXT, camera TEXT, track_id INTEGER, vehicle TEXT, direction TEXT
        );
    """)
    return db


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(args):
    url = STREAM_URL.format(cam=args.cam)
    model = YOLO(args.model)
    db = setup_db(args.db)
    line_zone = None
    last_log = 0.0

    while True:  # reconnect loop: HLS streams drop now and then
        try:
            print(f"[{now()}] connecting to {url}")
            results = model.track(
                source=url,
                stream=True,
                persist=True,
                classes=list(VEHICLE_CLASSES),
                conf=args.conf,
                vid_stride=args.stride,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            for r in results:
                h, w = r.orig_shape
                if line_zone is None:
                    x1, y1, x2, y2 = args.line
                    line_zone = sv.LineZone(
                        start=sv.Point(int(x1 * w), int(y1 * h)),
                        end=sv.Point(int(x2 * w), int(y2 * h)),
                    )

                dets = sv.Detections.from_ultralytics(r)
                if dets.tracker_id is None:
                    dets = sv.Detections.empty()

                # Line crossings
                if len(dets):
                    crossed_in, crossed_out = line_zone.trigger(dets)
                    ts = now()
                    for i in range(len(dets)):
                        for flag, direction in ((crossed_in[i], "in"), (crossed_out[i], "out")):
                            if flag:
                                db.execute(
                                    "INSERT INTO crossings VALUES (?,?,?,?,?)",
                                    (ts, args.cam, int(dets.tracker_id[i]),
                                     VEHICLE_CLASSES.get(int(dets.class_id[i]), "?"), direction),
                                )

                # Vehicles visible right now, logged every --interval seconds
                if time.time() - last_log >= args.interval:
                    c = Counter(VEHICLE_CLASSES[int(k)] for k in dets.class_id) if len(dets) else Counter()
                    db.execute(
                        "INSERT INTO occupancy VALUES (?,?,?,?,?,?,?)",
                        (now(), args.cam, sum(c.values()),
                         c["car"], c["motorcycle"], c["bus"], c["truck"]),
                    )
                    db.commit()
                    last_log = time.time()
                    print(f"[{now()}] {args.cam}: {sum(c.values())} in frame | "
                          f"line in={line_zone.in_count} out={line_zone.out_count}")

                if args.show:
                    frame = r.plot()
                    cv2.line(frame, (int(line_zone.vector.start.x), int(line_zone.vector.start.y)),
                             (int(line_zone.vector.end.x), int(line_zone.vector.end.y)), (0, 255, 255), 2)
                    cv2.imshow(args.cam, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"[{now()}] stream error: {e}")
        db.commit()
        print(f"[{now()}] stream ended, retrying in {args.retry}s")
        time.sleep(args.retry)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="cam1", help="camera folder in the stream URL, e.g. cam1, cam2")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    p.add_argument("--line", type=lambda s: [float(v) for v in s.split(",")],
                   default=[0.0, 0.5, 1.0, 0.5],
                   help="counting line x1,y1,x2,y2 as fractions of the frame (default: horizontal, mid-frame)")
    p.add_argument("--interval", type=int, default=60, help="seconds between occupancy logs")
    p.add_argument("--db", default="vehicles.db")
    p.add_argument("--retry", type=int, default=10)
    p.add_argument("--show", action="store_true", help="show the annotated video (press q to quit)")
    run(p.parse_args())
