#!/usr/bin/env python3
"""实例五（最终版）：自动目标跟踪（人脸）- 对齐后端优化参数"""
import cv2
import socket
import struct
import time

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"

# ====== 与后端完全一致的 CRC 和命令帧 ======
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
    """构造 CMD 0x0E 位置控制帧（与后端完全一致）"""
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

# ====== 控制参数（与后端优化版一致） ======
K_PX_TO_DEG = 0.22          # 每像素对应角度增量（增大，加快响应）
DEADZONE_PX = 5             # 死区像素（减小，更敏感）
SMOOTH_ALPHA = 0.5          # 平滑系数（0.5，降低滞后）
DT_MAX = 0.15               # dt 上限（放宽，防止突变）
REVERSE_PITCH = True        # 反转俯仰方向（与后端一致）

# ---------- 人脸检测器 ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开 RTSP 流")
        return

    target_yaw = 0.0
    target_pitch = 0.0
    smooth_fx = 0.0
    smooth_fy = 0.0
    last_control_t = time.time()
    frame_count = 0

    print("自动跟踪启动，按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        now_t = time.time()

        # 每隔一帧检测人脸（降低CPU负载）
        if frame_count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
        else:
            faces = []

        if len(faces) > 0:
            # 取最大人脸
            (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
            fx_raw, fy_raw = x + fw // 2, y + fh // 2
            # 平滑滤波（与后端一致：alpha=0.5）
            smooth_fx = SMOOTH_ALPHA * fx_raw + (1 - SMOOTH_ALPHA) * smooth_fx
            smooth_fy = SMOOTH_ALPHA * fy_raw + (1 - SMOOTH_ALPHA) * smooth_fy
            fx, fy = int(smooth_fx), int(smooth_fy)

            dx = fx - cx
            dy = fy - cy

            # 计算时间步长
            dt = now_t - last_control_t
            dt = min(dt, DT_MAX)

            if dt > 0:
                # 死区判断（与后端一致）
                if abs(dx) > DEADZONE_PX:
                    target_yaw += dx * K_PX_TO_DEG * dt
                if abs(dy) > DEADZONE_PX:
                    target_pitch -= dy * K_PX_TO_DEG * dt  # 图像Y轴向下为正，云台向上为正
                last_control_t = now_t

            # 限幅
            target_yaw = max(-135.0, min(135.0, target_yaw))
            target_pitch = max(-90.0, min(90.0, target_pitch))

            # 绘制
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)
            cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 发送位置命令（使用与后端相同的 build_position_cmd）
        cmd = build_position_cmd(target_yaw, target_pitch, reverse_pitch=REVERSE_PITCH)
        sock.sendto(cmd, (CAMERA_IP, UDP_PORT))

        # 显示
        cv2.imshow("Auto Track", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    main()
