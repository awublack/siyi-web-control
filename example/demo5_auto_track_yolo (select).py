#!/usr/bin/env python3
"""实例五（YOLO点击跟踪版·多线程版）：YOLO检测所有类别，鼠标点击框选目标并跟踪（后台YOLO线程）"""
import cv2
import socket
import struct
import time
import os
import threading
from collections import deque
from ultralytics import YOLO

# RTSP 优化
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"

# ====== CRC 和命令帧 ======
def calculate_crc16_xmodem(data):
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
    crc &= 0xFFFF
    return crc.to_bytes(2, 'little')

def build_position_cmd(yaw_deg, pitch_deg, reverse_pitch=True):
    header = bytes.fromhex("556601040000000E")
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    if reverse_pitch:
        pitch_val = -pitch_val
    yaw_val = max(-32768, min(32767, yaw_val))
    pitch_val = max(-32768, min(32767, pitch_val))
    yaw_bytes = struct.pack('<h', yaw_val)
    pitch_bytes = struct.pack('<h', pitch_val)
    payload = header + yaw_bytes + pitch_bytes
    crc = calculate_crc16_xmodem(payload)
    return payload + crc

# ====== 控制参数 ======
K_PX_TO_DEG = 0.22
DEADZONE_PX = 5
SMOOTH_ALPHA = 0.5
DT_MAX = 0.15
REVERSE_PITCH = True
MANUAL_SPEED_DEG_PER_SEC = 80.0
KEY_TIMEOUT = 0.2
MANUAL_IDLE_TIMEOUT = 1.0

# ====== YOLO 初始化 ======
yolo = YOLO('yolov8n.pt')
yolo.conf = 0.5

# ====== 全局状态 ======
mouse_data = {
    'active': False,
    'yaw_speed': 0.0,
    'pitch_speed': 0.0,
    'last_move_time': 0.0
}

tracking_bbox = None        # 当前跟踪框 (x, y, w, h)
tracking_mode = False       # 是否处于跟踪模式
smooth_fx, smooth_fy = 0.0, 0.0

# 共享数据（线程安全）
latest_frame = None
latest_detections = []      # 最新检测结果
detection_lock = threading.Lock()

def yolo_worker():
    """后台线程：持续从最新帧进行YOLO检测，更新检测结果"""
    global latest_detections
    while True:
        frame = None
        with detection_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is not None:
            results = yolo(frame, verbose=False, imgsz=320)
            dets = results[0].boxes
            detections = []
            if dets is not None:
                for box in dets:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.item()
                    cls_id = int(box.cls.item())
                    detections.append((x1, y1, x2, y2, conf, cls_id))
            with detection_lock:
                latest_detections = detections
        else:
            time.sleep(0.01)

# 启动YOLO线程
yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
yolo_thread.start()

def mouse_callback(event, x, y, flags, param):
    global mouse_data, tracking_bbox, tracking_mode, smooth_fx, smooth_fy

    if event == cv2.EVENT_LBUTTONDOWN:
        with detection_lock:
            dets = latest_detections.copy()
        hit = False
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            if x1 <= x <= x2 and y1 <= y <= y2:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracking_bbox = bbox
                tracking_mode = True
                smooth_fx = (x1 + x2) / 2
                smooth_fy = (y1 + y2) / 2
                print(f"✅ 开始跟踪目标: 类别 {cls_id}, 置信度 {conf:.2f}")
                hit = True
                break

        if not hit:
            mouse_data['active'] = True
            mouse_data['last_move_time'] = time.time()
            update_mouse_speed(x, y, param['w'], param['h'])
            if tracking_mode:
                tracking_mode = False
                tracking_bbox = None
                print("🛑 停止跟踪")
    elif event == cv2.EVENT_MOUSEMOVE and mouse_data['active']:
        mouse_data['last_move_time'] = time.time()
        update_mouse_speed(x, y, param['w'], param['h'])
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_data['active'] = False
        mouse_data['yaw_speed'] = 0.0
        mouse_data['pitch_speed'] = 0.0

def update_mouse_speed(x, y, w, h):
    yaw_speed = (x / w - 0.5) * 300
    pitch_speed = (0.5 - y / h) * 300
    mouse_data['yaw_speed'] = max(-100, min(100, yaw_speed))
    mouse_data['pitch_speed'] = max(-100, min(100, pitch_speed))

# ---------- 主循环 ----------
def main():
    global MANUAL_SPEED_DEG_PER_SEC, latest_frame, latest_detections
    global tracking_bbox, tracking_mode, smooth_fx, smooth_fy

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开 RTSP 流")
        return

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    h, w = frame.shape[:2]

    cv2.namedWindow("YOLO Click Track")
    cv2.setMouseCallback("YOLO Click Track", mouse_callback, {'w': w, 'h': h})

    target_yaw = 0.0
    target_pitch = 0.0
    last_control_t = time.time()

    keys_down = set()
    last_key_time = 0.0
    manual_mode = False
    last_manual_activity = 0.0

    print("=" * 55)
    print("YOLO 点击跟踪控制启动（多线程版）")
    print("YOLO 检测所有目标，鼠标点击绿色框开始跟踪")
    print("WASD 键盘控制 | 鼠标拖动控制 | Q加速 E减速 | 空格回中/停止跟踪 | ESC退出")
    print("手动操作后1秒无操作自动恢复跟踪")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # 更新最新帧供YOLO线程使用
        with detection_lock:
            latest_frame = frame.copy()
            current_detections = latest_detections.copy()

        cx, cy = w // 2, h // 2
        now_t = time.time()

        # ---- 键盘输入 ----
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == 27:
                break
            elif key == ord(' '):
                target_yaw = 0.0
                target_pitch = 0.0
                tracking_mode = False
                tracking_bbox = None
                print("回中并停止跟踪")
            elif key == ord('q'):
                MANUAL_SPEED_DEG_PER_SEC = min(150, MANUAL_SPEED_DEG_PER_SEC + 10)
                print(f"加速: {MANUAL_SPEED_DEG_PER_SEC:.0f}°/s")
            elif key == ord('e'):
                MANUAL_SPEED_DEG_PER_SEC = max(10, MANUAL_SPEED_DEG_PER_SEC - 10)
                print(f"减速: {MANUAL_SPEED_DEG_PER_SEC:.0f}°/s")
            elif key in (ord('w'), ord('s'), ord('a'), ord('d')):
                keys_down.add(key)
                last_key_time = now_t
                manual_mode = True
                last_manual_activity = now_t

        if now_t - last_key_time > KEY_TIMEOUT:
            keys_down.clear()

        if mouse_data['active']:
            manual_mode = True
            last_manual_activity = now_t

        if manual_mode and (now_t - last_manual_activity > MANUAL_IDLE_TIMEOUT):
            manual_mode = False
            print("恢复自动模式")

        dt = now_t - last_control_t
        dt = min(dt, DT_MAX)
        if dt > 0:
            if manual_mode:
                dx, dy = 0, 0
                if ord('w') in keys_down: dy = 1
                if ord('s') in keys_down: dy = -1
                if ord('a') in keys_down: dx = -1
                if ord('d') in keys_down: dx = 1
                if mouse_data['active']:
                    dx += mouse_data['yaw_speed'] / 100.0
                    dy += mouse_data['pitch_speed'] / 100.0
                target_yaw += dx * MANUAL_SPEED_DEG_PER_SEC * dt
                target_pitch += dy * MANUAL_SPEED_DEG_PER_SEC * dt
            else:
                if tracking_mode and tracking_bbox is not None:
                    # 寻找与目标框 IoU 最大的检测
                    best_match = None
                    best_iou = 0.2
                    tx, ty, tw, th = tracking_bbox
                    for det in current_detections:
                        x1, y1, x2, y2, _, _ = det
                        ix1 = max(x1, tx); iy1 = max(y1, ty)
                        ix2 = min(x2, tx+tw); iy2 = min(y2, ty+th)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        area1 = (x2-x1)*(y2-y1); area2 = tw*th
                        iou = inter / (area1 + area2 - inter + 1e-5)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = det

                    if best_match is not None:
                        x1, y1, x2, y2, _, _ = best_match
                        tracking_bbox = (x1, y1, x2-x1, y2-y1)
                        fx_raw, fy_raw = (x1+x2)//2, (y1+y2)//2
                        smooth_fx = SMOOTH_ALPHA * fx_raw + (1-SMOOTH_ALPHA) * smooth_fx
                        smooth_fy = SMOOTH_ALPHA * fy_raw + (1-SMOOTH_ALPHA) * smooth_fy
                        fx, fy = int(smooth_fx), int(smooth_fy)
                        dx = fx - cx; dy = fy - cy
                        if abs(dx) > DEADZONE_PX:
                            target_yaw += dx * K_PX_TO_DEG * dt
                        if abs(dy) > DEADZONE_PX:
                            target_pitch -= dy * K_PX_TO_DEG * dt
                        # 绘制跟踪框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, "TRACKING (YOLO)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        tracking_mode = False
                        tracking_bbox = None
                        print("跟踪丢失")
                        cv2.putText(frame, "LOST", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 非跟踪模式：显示所有检测框
                    for det in current_detections:
                        x1, y1, x2, y2, conf, cls_id = det
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        coco_names = yolo.model.names
                        name = coco_names.get(cls_id, str(cls_id))
                        label = f"{name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.putText(frame, "IDLE - Click a green box to track", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            # 限幅
            target_yaw = max(-135.0, min(135.0, target_yaw))
            target_pitch = max(-90.0, min(90.0, target_pitch))
            last_control_t = now_t

        # ---- 发送命令 ----
        cmd = build_position_cmd(target_yaw, target_pitch, reverse_pitch=REVERSE_PITCH)
        sock.sendto(cmd, (CAMERA_IP, UDP_PORT))

        # ---- 显示状态 ----
        mode_str = "MANUAL" if manual_mode else ("TRACK" if tracking_mode else "AUTO")
        info = f"Mode:{mode_str} Yaw:{target_yaw:+6.1f} Pitch:{target_pitch:+6.1f}"
        cv2.putText(frame, info, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("YOLO Click Track", frame)

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    main()
