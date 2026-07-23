import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
import csv
from datetime import datetime
import time
import os   
import serial

 
# CONFIGURATION
 
CONF_THRESHOLD = 0.35
STABLE_FRAMES = 1
MIN_POINTS = 800
COOLDOWN_SECONDS = 2.0

YOLO_IMG_SIZE = 256
YOLO_EVERY_N_FRAMES = 8
DEPTH_SAMPLE_STEP = 3

CSV_FILE = "hole_measurements.csv"
IMAGE_DIR = "outputs/images"
PCD_DIR = "outputs/pointclouds"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PCD_DIR, exist_ok=True)

 
# SERIAL CONNECTION (ESP32)
 
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

try:
    rover = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Connected to ESP32")
except Exception as e:
    print("ESP32 Connection Failed:", e)
    rover = None


def send_command(cmd):
    if rover:
        rover.write(cmd.encode())

 
# STATE VARIABLES
 
movement_state = "IDLE"
manual_mode = False
emergency_mode = False
HEARTBEAT_INTERVAL = 0.5
last_heartbeat = time.time()
resume_time = 0

 
# LOAD YOLO MODEL
 
model = YOLO("best.pt")
model.fuse()

 
# REALSENSE SETUP
 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

print("System started")

 
# INITIAL MOVEMENT SEQUENCE
 
print("\nInitial Movement Sequence")

for i in range(2):
    print(f"Forward Move {i+1}")
    send_command('F')
    time.sleep(1)
    send_command('S')
    time.sleep(0.5)

for i in range(2):
    print(f"Backward Move {i+1}")
    send_command('B')
    time.sleep(1)
    send_command('S')
    time.sleep(0.5)

movement_state = "FORWARD"
print("Autonomous Forward Mode Started\n")

 
# CSV HEADER
 
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "width_m",
            "length_m",
            "max_depth_m",
            "volume_m3",
            "concrete_L",
            "confidence_pct"
        ])

 
# LOOP VARIABLES
 
stable_count = 0
hole_logged = False
last_measure_time = 0
frame_id = 0

try:
    while True:

        # Resume movement
        if movement_state == "WAITING" and time.time() >= resume_time:
            movement_state = "FORWARD"
            print("\nRover Resumed Autonomous Movement\n")

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
        real_hole = False

         
        # YOLO DETECTION
         
        if frame_id % YOLO_EVERY_N_FRAMES == 0:
            results = model(color,
                            conf=CONF_THRESHOLD,
                            imgsz=YOLO_IMG_SIZE,
                            verbose=False)

            for r in results:
                for box in r.boxes:
                    detected = True
                    bbox = tuple(map(int, box.xyxy[0]))
                    conf = float(box.conf[0])
                    break

        frame_id += 1
        stable_count = stable_count + 1 if detected else 0

        if not detected:
            hole_logged = False

         
        # HOLE CONFIRMATION (AI + DEPTH)
         
        if (
            detected
            and stable_count >= STABLE_FRAMES
            and not hole_logged
            and time.time() - last_measure_time > COOLDOWN_SECONDS
        ):

            hole_logged = True
            last_measure_time = time.time()

            print("\n_________________________________________________")
            print("HOLE DETECTED BY AI MODEL")
            

            movement_state = "WAITING"
            resume_time = time.time() + 5
            send_command('S')

            x1, y1, x2, y2 = bbox
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            # Collect depth points
            points = []
            for v in range(y1, y2, DEPTH_SAMPLE_STEP):
                for u in range(x1, x2, DEPTH_SAMPLE_STEP):
                    d = depth[v, u]
                    if d == 0:
                        continue
                    z = d * depth_scale
                    points.append(
                        rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                    )

            points = np.asarray(points)

            # ❌ NO REAL DEPTH
            if len(points) < MIN_POINTS:
                print("\nAI VERIFICATION RESULT :")
                print("Visual hole detected but no physical depth found")
                print("Not a real pothole — ignoring detection")
                print("System Action : MOVE FORWARD to search real hole")
                print("_________________________________________________\n")

                # 🔥 COMMAND ROVER TO MOVE FORWARD
                movement_state = "FORWARD"
                send_command('F')

                print("🚗 Rover moving forward to search for real hole")

                hole_logged = False
                continue


             
            # 🟢 REAL HOLE CONFIRMED
             
            print("\nAI VERIFICATION RESULT :")
            print(f"AI Confidence : {conf*100:.2f}%")
            print("Real pothole confirmed")
            print("Sending to Measurement Unit...")
            print("_________________________________________________")

            real_hole = True

            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(points))
            pcd = pcd.voxel_down_sample(voxel_size=0.02)

            plane_model, _ = pcd.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=150
            )

            A, B, C, D = plane_model
            pts = np.asarray(pcd.points)

            Xv, Yv, Zv = pts[:, 0], pts[:, 1], pts[:, 2]

            width = Xv.max() - Xv.min()
            length = Yv.max() - Yv.min()

            volume = 0.0
            depths = []
            fx, fy = intr.fx, intr.fy

            for p in pts:
                plane_z = (-A*p[0] - B*p[1] - D) / C
                depth_diff = plane_z - p[2]
                if depth_diff > 0:
                    depths.append(depth_diff)
                    volume += ((p[2]/fx)*(p[2]/fy)) * depth_diff

            max_depth = max(depths) * 100
            # volume_liters = volume * 1000
            concrete = volume * 1000

            print("\nAI MEASUREMENT RESULTS")
            print("-----------------------------")
            print(f"Width      : {width:.2f} m")
            print(f"Length     : {length:.2f} m")
            print(f"Max Depth  : {max_depth:.2f} Cm")
            print("-----------------------------")
            print(f"Volume     : {volume:.10f} m³")
            print(f"Concrete   : {concrete:.4f} L")

            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    round(width, 3),
                    round(length, 3),
                    round(max_depth, 3),
                    round(volume, 4),
                    round(concrete, 4),
                    round(conf * 100, 2)
                ])

            print("\nMeasurement Stored in CSV")

             
            # SAVE IMAGE + POINT CLOUD
             
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            image_path = f"{IMAGE_DIR}/hole_{ts}.jpg"
            pcd_path = f"{PCD_DIR}/hole_{ts}.ply"

            # Create copy so original frame is untouched
            save_img = color.copy()

            # Draw box on saved image
            x1, y1, x2, y2 = bbox
            cv2.rectangle(save_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Hole {conf*100:.1f}%"
            cv2.putText(save_img, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            # Save annotated image
            cv2.imwrite(image_path, save_img)

            # Save point cloud
            o3d.io.write_point_cloud(pcd_path, pcd)
            o3d.visualization.draw_geometries(
            [pcd],
            window_name="3D Hole Model (Close to Continue)"
        )


            print(f"Image saved : {image_path}")
            print(f"Point cloud saved : {pcd_path}")

         
        # DRAW BOX ONLY IF REAL HOLE
         
        if detected and real_hole:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(color, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            label = f"Hole {conf*100:.1f}%"
            cv2.putText(color, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

         
        # HEARTBEAT CONTROL
         
        if not emergency_mode and not manual_mode:
            if movement_state == "FORWARD":
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    send_command('F')
                    last_heartbeat = time.time()
            else:
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    send_command('S')
                    last_heartbeat = time.time()

        cv2.imshow("Hole Detection", color)
        key = cv2.waitKey(1)

        
        # GLOBAL EMERGENCY STOP (SPACE)
        
        if key == 32:   # SPACE key

            emergency_mode = not emergency_mode

            if emergency_mode:
                movement_state = "IDLE"
                send_command('S')
                print("\nEMERGENCY STOP ACTIVATED — ALL SYSTEMS HALTED\n")
            else:
                movement_state = "FORWARD"
                last_heartbeat = 0
                print("\nEMERGENCY RELEASED — Autonomous Resuming\n")

        
        # MANUAL MODE TOGGLE (M key)
        
        elif key == ord('m') or key == ord('M'):

            manual_mode = not manual_mode

            if manual_mode:
                movement_state = "IDLE"
                send_command('S')
                print("\nMANUAL MODE ACTIVATED")
                print("Use ↑/W Forward | ↓/S Backward | SPACE Emergency")
            else:
                movement_state = "FORWARD"
                print("\nAUTONOMOUS MODE RESUMED")

        
        #  MANUAL CONTROL
        
        #  MANUAL CONTROL

        elif manual_mode and not emergency_mode:

            # Forward
            if key == ord('w') or key == ord('W'):
                send_command('F')
                print("MANUAL: Forward")

            # Backward
            elif key == ord('s') or key == ord('S'):
                send_command('B')
                print("MANUAL: Backward")

            # Optional: Left
            elif key == ord('a') or key == ord('A'):
                send_command('L')
                print("MANUAL: Left")

            # Optional: Right
            elif key == ord('d') or key == ord('D'):
                send_command('R')
                print("MANUAL: Right")


        
        # QUIT
        
        elif key == ord('q'):
            break

finally:
    if rover:
        send_command('S')
        rover.close()

    pipeline.stop()
    cv2.destroyAllWindows()
    print("System stopped")
