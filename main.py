import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
import csv
from datetime import datetime
import time
import os


# ==============================
# CONFIGURATION
# ==============================

CONF_THRESHOLD = 0.35
STABLE_FRAMES = 5
MIN_POINTS = 1500
CSV_FILE = "hole_measurements.csv"
COOLDOWN_SECONDS = 2.0

IMAGE_DIR = "outputs/images"
PCD_DIR = "outputs/pointclouds"

# 🔥 Depth confirmation threshold (meters)
REAL_HOLE_DEPTH_DIFF = 0.03   # 3 cm

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PCD_DIR, exist_ok=True)


# ==============================
# 🔥 DEPTH CONFIRMATION FUNCTION
# ==============================

def is_real_hole(depth, bbox, depth_scale):
    x1, y1, x2, y2 = bbox

    # Central region (hole interior)
    mx = int((x2 - x1) * 0.25)
    my = int((y2 - y1) * 0.25)

    cx1, cy1 = x1 + mx, y1 + my
    cx2, cy2 = x2 - mx, y2 - my

    center = depth[cy1:cy2, cx1:cx2]
    border = depth[y1:y2, x1:x2]

    center_vals = center[center > 0]
    border_vals = border[border > 0]

    if len(center_vals) < 50 or len(border_vals) < 50:
        return False

    center_depth = np.median(center_vals) * depth_scale
    border_depth = np.median(border_vals) * depth_scale

    depth_diff = center_depth - border_depth

    return depth_diff > REAL_HOLE_DEPTH_DIFF


# ==============================
# Load YOLO model
# ==============================

model = YOLO(
    r"C:\Users\santh\Documents\SANTHOSH\Autonomus_rover\PATH HOLE DETECTION.v1i.yolov8\runs\detect\train\weights\best.pt"
)
model.fuse()


# ==============================
# RealSense setup
# ==============================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)


# ==============================
# CSV setup
# ==============================

with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "width_m",
        "length_m",
        "max_depth_m",
        "volume_m3",
        "volume_liters",
        "concrete_m3",
        "confidence_pct",
        "image_path",
        "pointcloud_path"
    ])

print("✅ System started")
print("▶ Press S to START detection")
print("⏹ Press X to STOP detection")
print("❌ Press Q to QUIT")


# ==============================
# STATE VARIABLES
# ==============================

stable_count = 0
hole_confirmed = False
last_measure_time = 0
detection_enabled = False


try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        detected = False
        bbox = None
        conf = 0.0


        # ==============================
        # YOLO detection
        # ==============================
        if detection_enabled:
            results = model(color, conf=CONF_THRESHOLD, imgsz=640)

            detected = False      # 🔥 reset every frame
            bbox = None
            conf = 0.0

            for r in results:
                for box in r.boxes:

                    candidate_bbox = tuple(map(int, box.xyxy[0]))
                    candidate_conf = float(box.conf[0])

                    # 🔥 CHECK REAL DEPTH FIRST
                    if is_real_hole(depth, candidate_bbox, depth_scale):

                        detected = True
                        bbox = candidate_bbox
                        conf = candidate_conf

                        # ✅ DRAW ONLY REAL HOLE
                        cv2.rectangle(color, bbox[:2], bbox[2:], (0, 255, 0), 2)
                        cv2.putText(
                            color,
                            f"Hole {conf*100:.1f}%",
                            (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        break

                    else:
                        print("❌ YOLO match but no real depth hole")

            # ===== STABILITY =====
            if detected:
                stable_count += 1
            else:
                stable_count = 0

            cv2.putText(
                color,
                f"Stable frames: {stable_count}/{STABLE_FRAMES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


        # ==============================
        # CONFIRM HOLE (UNCHANGED)
        # ==============================

        if (
            detection_enabled
            and detected
            and stable_count >= STABLE_FRAMES
            and not hole_confirmed
            and time.time() - last_measure_time > COOLDOWN_SECONDS
        ):
            hole_confirmed = True
            last_measure_time = time.time()

            x1, y1, x2, y2 = bbox
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            points = []
            for v in range(y1, y2):
                for u in range(x1, x2):
                    d = depth[v, u]
                    if d == 0:
                        continue
                    z = d * depth_scale
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                    points.append([X, Y, Z])

            points = np.array(points)
            if len(points) < MIN_POINTS:
                print("❌ Not enough depth points")
                stable_count = 0
                hole_confirmed = False
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            plane_model, _ = pcd.segment_plane(
                distance_threshold=0.005,
                ransac_n=3,
                num_iterations=1000
            )
            A, B, C, D = plane_model

            Xv, Yv, Zv = points[:, 0], points[:, 1], points[:, 2]
            width = Xv.max() - Xv.min()
            length = Yv.max() - Yv.min()

            volume = 0.0
            depths = []
            fx, fy = intr.fx, intr.fy

            for p in points:
                plane_z = (-A*p[0] - B*p[1] - D) / C
                depth_diff = plane_z - p[2]
                if depth_diff > 0:
                    depths.append(depth_diff)
                    pixel_area = (p[2]/fx) * (p[2]/fy)
                    volume += pixel_area * depth_diff

            max_depth = max(depths)
            volume_liters = volume * 1000.0
            concrete = volume * 1.1

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            image_path = f"{IMAGE_DIR}/hole_{ts}.jpg"
            pcd_path = f"{PCD_DIR}/hole_{ts}.ply"

            cv2.imwrite(image_path, color)
            o3d.io.write_point_cloud(pcd_path, pcd)

            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    round(width, 3),
                    round(length, 3),
                    round(max_depth, 3),
                    round(volume, 4),
                    round(volume_liters, 2),
                    round(concrete, 4),
                    round(conf * 100, 2),
                    image_path,
                    pcd_path
                ])

            print("\n🟢 REAL HOLE MEASURED & SAVED")

            hole_confirmed = False
            stable_count = 0


        # ==============================
        # DISPLAY UI
        # ==============================

        status_text = "DETECTION: ON" if detection_enabled else "DETECTION: OFF"
        status_color = (0, 255, 0) if detection_enabled else (0, 0, 255)

        cv2.putText(color, status_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.putText(
            color,
            "Press S to START | X to STOP | Q to QUIT",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hole Detection", color)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            detection_enabled = True
            stable_count = 0
            hole_confirmed = False
            print("▶ Hole detection ENABLED")

        elif key == ord('x'):
            detection_enabled = False
            stable_count = 0
            hole_confirmed = False
            print("⏹ Hole detection DISABLED")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 System stopped")
