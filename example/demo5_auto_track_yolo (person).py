#!/usr/bin/env python3
"""实例五（YOLO集成版）：自动跟踪(YOLO检测人) + 键盘控制 + 鼠标控制（修复W/S方向）"""
import cv2
import socket
import struct
import time
import os
from ultralytics import YOLO

# ====== 优化 RTSP 读取参数（TCP传输，加大缓冲区）======
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000"

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"

# ====== CRC 和命令帧（与后端一致）======
def calculate_crc16_xmodem(data: bytes) -> bytes:
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

def build_position_cmd(yaw_deg: float, pitch_deg: float, reverse_pitch: bool = True) -> bytes:
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
yolo.conf = 0.5             # 置信度阈值
yolo.classes = [0]          # 只检测 person

# ---------- 鼠标回调 ----------
mouse_data = {
    'active': False,
    'yaw_speed': 0.0,
    'pitch_speed': 0.0,
    'last_move_time': 0.0
}

def mouse_callback(event, x, y, flags, param):
    global mouse_data
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_data['active'] = True
        mouse_data['last_move_time'] = time.time()
        update_mouse_speed(x, y, param['w'], param['h'])
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
    global MANUAL_SPEED_DEG_PER_SEC

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

    cv2.namedWindow("YOLO Integrated Control")
    cv2.setMouseCallback("YOLO Integrated Control", mouse_callback, {'w': w, 'h': h})

    target_yaw = 0.0
    target_pitch = 0.0
    smooth_fx = 0.0
    smooth_fy = 0.0
    last_control_t = time.time()
    frame_count = 0

    keys_down = set()
    last_key_time = 0.0
    manual_mode = False
    last_manual_activity = 0.0

    print("集成控制启动：YOLO 检测人 | WASD键盘控制 | 鼠标拖动控制 | Q加速 E减速 | 空格回中 | ESC退出")
    print("W上仰 S下俯 A左 D右")
    print("手动操作后1秒无操作自动恢复跟踪")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
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
                print("回中")
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
            print("恢复自动跟踪")

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
                # ====== YOLO 检测（每3帧一次）======
                if frame_count % 3 == 0:
                    results = yolo(frame, verbose=False, imgsz=416)
                    dets = results[0].boxes
                else:
                    dets = []

                if dets is not None and len(dets) > 0:
                    # 取置信度最高的检测
                    best = max(dets, key=lambda b: b.conf.item())
                    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
                    fx_raw, fy_raw = (x1 + x2) // 2, (y1 + y2) // 2
                    # 平滑滤波
                    smooth_fx = SMOOTH_ALPHA * fx_raw + (1 - SMOOTH_ALPHA) * smooth_fx
                    smooth_fy = SMOOTH_ALPHA * fy_raw + (1 - SMOOTH_ALPHA) * smooth_fy
                    fx, fy = int(smooth_fx), int(smooth_fy)

                    dx = fx - cx
                    dy = fy - cy

                    if abs(dx) > DEADZONE_PX:
                        target_yaw += dx * K_PX_TO_DEG * dt
                    if abs(dy) > DEADZONE_PX:
                        target_pitch -= dy * K_PX_TO_DEG * dt

                    # 绘制 YOLO 检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)
                    label = f"Person {best.conf.item():.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, "YOLO TRACK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO PERSON", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            target_yaw = max(-135.0, min(135.0, target_yaw))
            target_pitch = max(-90.0, min(90.0, target_pitch))
            last_control_t = now_t

        # ---- 发送命令 ----
        cmd = build_position_cmd(target_yaw, target_pitch, reverse_pitch=REVERSE_PITCH)
        sock.sendto(cmd, (CAMERA_IP, UDP_PORT))

        # ---- 显示状态 ----
        mode_str = "MANUAL" if manual_mode else "AUTO"
        info = f"Mode:{mode_str} Yaw:{target_yaw:+6.1f} Pitch:{target_pitch:+6.1f}"
        cv2.putText(frame, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("YOLO Integrated Control", frame)

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    main()
