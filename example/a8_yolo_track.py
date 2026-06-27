import cv2
import numpy as np
from ultralytics import YOLO
import socket
import struct
import time
from pynput import keyboard

# ====== 配置 ======
A8_IP = '192.168.144.25'
A8_PORT = 37260
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'

MODEL_PATH = 'yolov8n.pt'
TARGET_CLASS = 0          # 0=person, 2=car
GAIN_YAW = 0.04
GAIN_PITCH = 0.03
MAX_SPEED = 15
DEAD_ZONE = 20
YOLO_SKIP = 5             # 每 5 帧检一次，降 CPU

# ====== CRC16-CCITT（SIYI 协议标准）======
def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

SEQ = 0
def send_gimbal_speed(yaw_speed, pitch_speed):
    """SIYI 协议 0x01 0x07：云台速度控制（修正 CRC16）"""
    global SEQ
    yaw_speed = max(-MAX_SPEED, min(MAX_SPEED, yaw_speed))
    pitch_speed = max(-MAX_SPEED, min(MAX_SPEED, pitch_speed))

    payload = struct.pack('<ff', yaw_speed, pitch_speed)  # DATA: 2×float32 LE
    data_len = len(payload)

    pkt = bytearray()
    pkt += struct.pack('<H', 0x5566)      # STX 0x6655 小端 → 0x5566
    pkt.append(0x01)                      # CTRL: need_ack=0
    pkt += struct.pack('<H', data_len)    # DATA_LEN
    pkt += struct.pack('<H', SEQ)         # SEQ
    SEQ = (SEQ + 1) % 65536
    pkt.append(0x07)                      # CMD_ID: GIMBAL_SPEED
    pkt += payload                        # DATA
    crc = crc16_ccitt(pkt)
    pkt += struct.pack('<H', crc)         # CRC16 小端

    sock.sendto(pkt, (A8_IP, A8_PORT))

# ====== 键盘控制 ======
# 模式: 'manual' = 键盘控云台, 'auto' = YOLO 跟踪
mode = 'auto'
manual_yaw, manual_pitch = 0.0, 0.0
manual_zoom_in = False
manual_zoom_out = False
running = True

def on_press(key):
    global mode, manual_yaw, manual_pitch, manual_zoom_in, manual_zoom_out, running
    try:
        if key.char == 'q':
            running = False
            return False  # 停 listener
        if key.char == 't':
            mode = 'manual' if mode == 'auto' else 'auto'
            print(f"[KEY] 模式切换 → {mode}")
        if mode == 'manual':
            if key.char == 'w': manual_pitch =  8   # 抬头
            if key.char == 's': manual_pitch = -8   # 低头
            if key.char == 'a': manual_yaw   =  8   # 左转
            if key.char == 'd': manual_yaw   = -8   # 右转
            if key.char == '+': manual_zoom_in = True
            if key.char == '-': manual_zoom_out = True
    except AttributeError:
        pass

def on_release(key):
    global manual_yaw, manual_pitch, manual_zoom_in, manual_zoom_out
    try:
        if mode == 'manual':
            if key.char in ('w','s'): manual_pitch = 0
            if key.char in ('a','d'): manual_yaw = 0
            if key.char == '+': manual_zoom_in = False
            if key.char == '-': manual_zoom_out = False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ====== UDP + 视频 ======
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.5)  # 非阻塞感，避免 sendto 卡

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("❌ RTSP 打不开，检查 A8 IP 和网线")
    exit(1)

model = YOLO(MODEL_PATH)

print("=" * 50)
print("快捷键:")
print("  t  → 切换 auto(跟踪) / manual(键盘)")
print("  w/s → 俯仰  a/d → 偏航  +/- → 变焦(若固件支持)")
print("  q  → 退出")
print("=" * 50)

frame_cnt = 0
last_bbox = None  # YOLO 降频：保最近一次 bbox

while running:
    ret, frame = cap.read()
    if not ret:
        print("⚠ RTSP 丢帧，重试...")
        time.sleep(0.05)
        continue

    h, w = frame.shape[:2]
    cx_f, cy_f = w // 2, h // 2
    frame_cnt += 1

    # ---- YOLO 降频推理 ----
    if frame_cnt % YOLO_SKIP == 0:
        results = model(frame, classes=[TARGET_CLASS], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        if len(boxes) > 0:
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            best = np.argmax(areas)
            last_bbox = boxes[best]
        else:
            last_bbox = None

    # ---- Auto 模式：YOLO 闭环 ----
    if mode == 'auto':
        if last_bbox is not None:
            x1,y1,x2,y2 = last_bbox
            cx_t = int((x1+x2)/2)
            cy_t = int((y1+y2)/2)
            dx = cx_t - cx_f
            dy = cy_t - cy_f

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.circle(frame, (cx_t, cy_t), 5, (0,0,255), -1)

            if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                send_gimbal_speed(-dx*GAIN_YAW, -dy*GAIN_PITCH)
            else:
                send_gimbal_speed(0, 0)
        else:
            send_gimbal_speed(0, 0)

    # ---- Manual 模式：键盘控 ----
    else:
        send_gimbal_speed(manual_yaw, manual_pitch)
        # 变焦（0x05 MANUAL_ZOOM）：简单起见这里只示意，要完整可再加
        # if manual_zoom_in:  send_zoom(1)
        # if manual_zoom_out: send_zoom(-1)

    # ---- HUD ----
    cv2.putText(frame, f"Mode: {mode}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow('A8 YOLO Track', frame)
    cv2.waitKey(1)  # 仍保留，只用来刷新窗口，不靠它吃按键

# ---- 清理 ----
send_gimbal_speed(0, 0)
cap.release()
cv2.destroyAllWindows()
sock.close()
listener.stop()
print("退出完成")
