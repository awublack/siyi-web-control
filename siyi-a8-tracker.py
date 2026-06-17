#!/usr/bin/env python3
"""
SIYI A8 Mini 人脸跟踪 + 键盘控制融合版
- 默认自动人脸跟踪
- WSAD 手动控制云台
- 空格回中
- 松开按键自动恢复跟踪

运行前需要安装依赖:
pip install opencv-python numpy keyboard

注意：keyboard 库需要管理员/root 权限
"""

import cv2
import numpy as np
import socket
import struct
import subprocess
import threading
import time
import os
import keyboard  # pip install keyboard
from collections import deque

# ==================== CRC16 表 ====================
CRC16_TABLE = [
    0x0000,0x1021,0x2042,0x3063,0x4084,0x50A5,0x60C6,0x70E7,
    0x8108,0x9129,0xA14A,0xB16B,0xC18C,0xD1AD,0xE1CE,0xF1EF,
    0x1231,0x0210,0x3273,0x2252,0x52B5,0x4294,0x72F7,0x62D6,
    0x9339,0x8318,0xB37B,0xA35A,0xD3BD,0xC39C,0xF3FF,0xE3DE,
    0x2462,0x3443,0x0420,0x1401,0x64E6,0x74C7,0x44A4,0x5485,
    0xA56A,0xB54B,0x8528,0x9509,0xE5EE,0xF5CF,0xC5AC,0xD58D,
    0x3653,0x2672,0x1611,0x0630,0x76D7,0x66F6,0x5695,0x46B4,
    0xB75B,0xA77A,0x9719,0x8738,0xF7DF,0xE7FE,0xD79D,0xC7BC,
    0x48C4,0x58E5,0x6886,0x78A7,0x0840,0x1861,0x2802,0x3823,
    0xC9CC,0xD9ED,0xE98E,0xF9AF,0x8948,0x9969,0xA90A,0xB92B,
    0x5AF5,0x4AD4,0x7AB7,0x6A96,0x1A71,0x0A50,0x3A33,0x2A12,
    0xDBFD,0xCBDC,0xFBBF,0xEB9E,0x9B79,0x8B58,0xBB3B,0xAB1A,
    0x6CA6,0x7C87,0x4CE4,0x5CC5,0x2C22,0x3C03,0x0C60,0x1C41,
    0xEDAE,0xFD8F,0xCDEC,0xDDCD,0xAD2A,0xBD0B,0x8D68,0x9D49,
    0x7E97,0x6EB6,0x5ED5,0x4EF4,0x3E13,0x2E32,0x1E51,0x0E70,
    0xFF9F,0xEFBE,0xDFDD,0xCFFC,0xBF1B,0xAF3A,0x9F59,0x8F78,
    0x9188,0x81A9,0xB1CA,0xA1EB,0xD10C,0xC12D,0xF14E,0xE16F,
    0x1080,0x00A1,0x30C2,0x20E3,0x5004,0x4025,0x7046,0x6067,
    0x83B9,0x9398,0xA3FB,0xB3DA,0xC33D,0xD31C,0xE37F,0xF35E,
    0x02B1,0x1290,0x22F3,0x32D2,0x4235,0x5214,0x6277,0x7256,
    0xB5EA,0xA5CB,0x95A8,0x8589,0xF56E,0xE54F,0xD52C,0xC50D,
    0x34E2,0x24C3,0x14A0,0x0481,0x7466,0x6447,0x5424,0x4405,
    0xA7DB,0xB7FA,0x8799,0x97B8,0xE75F,0xF77E,0xC71D,0xD73C,
    0x26D3,0x36F2,0x0691,0x16B0,0x6657,0x7676,0x4615,0x5634,
    0xD94C,0xC96D,0xF90E,0xE92F,0x99C8,0x89E9,0xB98A,0xA9AB,
    0x5844,0x4865,0x7806,0x6827,0x18C0,0x08E1,0x3882,0x28A3,
    0xCB7D,0xDB5C,0xEB3F,0xFB1E,0x8BF9,0x9BD8,0xABBB,0xBB9A,
    0x4A75,0x5A54,0x6A37,0x7A16,0x0AF1,0x1AD0,0x2AB3,0x3A92,
    0xFD2E,0xED0F,0xDD6C,0xCD4D,0xBDAA,0xAD8B,0x9DE8,0x8DC9,
    0x7C26,0x6C07,0x5C64,0x4C45,0x3CA2,0x2C83,0x1CE0,0x0CC1,
    0xEF1F,0xFF3E,0xCF5D,0xDF7C,0xAF9B,0xBFBA,0x8FD9,0x9FF8,
    0x6E17,0x7E36,0x4E55,0x5E74,0x2E93,0x3EB2,0x0ED1,0x1EF0
]

# ==================== 配置参数 ====================
CAMERA_IP = "192.168.144.25"
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
UDP_PORT = 37260
VIDEO_W = 640
VIDEO_H = 480

# ── PID 参数 ──────────────────────────
KP_YAW = 0.25
KI_YAW = 0.001
KD_YAW = 0.04

KP_PITCH = 0.18
KI_PITCH = 0.0008
KD_PITCH = 0.025

MAX_SPEED = 100
DEADZONE_IN = 5
DEADZONE_OUT = 12
ALPHA_ERR = 0.3
OUT_AVG_WIN = 3
CONTROL_DT = 0.025

# ── 手拉跟随参数 ──────────────────────
LOST_DECAY_RATE = 0.95
RECOVERY_FRAMES = 10
MANUAL_INTERVENTION_THRESHOLD = 150

# ── 键盘控制参数 ──────────────────────
MANUAL_SPEED = 50
BOOST_SPEED = 100
HOME_STEPS = 20
HOME_DELAY = 0.1

# ==================== 全局变量 ====================
manual_mode = False
manual_yaw = 0
manual_pitch = 0
lock = threading.Lock()

# ==================== PID 控制器 ====================
class PIDController:
    def __init__(self, kp, ki, kd, max_spd=100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_spd = max_spd
        self.integral = 0.0
        self.prev_err = 0.0
        self.filt_err = 0.0
        self.last_t = time.time()
        self.out_buf = deque(maxlen=OUT_AVG_WIN)
        self.lost_frames = 0
        self.recovery_counter = 0
        self.is_manual_intervention = False

    def update(self, raw_err, has_face=True):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt <= 0:
            dt = CONTROL_DT

        if not has_face:
            self.lost_frames += 1
            self.integral *= LOST_DECAY_RATE
            self.filt_err *= LOST_DECAY_RATE
            self.recovery_counter = 0
            return 0

        if self.lost_frames > 5 and abs(raw_err) > MANUAL_INTERVENTION_THRESHOLD:
            self.is_manual_intervention = True
            self.recovery_counter = RECOVERY_FRAMES
            self.integral = 0
            self.filt_err = 0
            self.prev_err = 0

        self.lost_frames = 0

        if self.recovery_counter > 0:
            self.recovery_counter -= 1
            recovery_factor = 1.0 - (self.recovery_counter / RECOVERY_FRAMES)
        else:
            recovery_factor = 1.0
            self.is_manual_intervention = False

        self.filt_err = ALPHA_ERR * raw_err + (1 - ALPHA_ERR) * self.filt_err
        err = self.filt_err * recovery_factor

        p = self.kp * err
        self.integral += err * dt
        max_i = self.max_spd / (self.ki + 1e-9) if self.ki > 0 else 0
        self.integral = max(-max_i, min(max_i, self.integral))
        i = self.ki * self.integral
        d = self.kd * (err - self.prev_err) / dt if dt > 0 else 0
        self.prev_err = err

        out = p + i + d
        out = max(-self.max_spd, min(self.max_spd, out))
        self.out_buf.append(out)
        return int(np.mean(self.out_buf))

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.filt_err = 0.0
        self.out_buf.clear()
        self.lost_frames = 0
        self.recovery_counter = 0
        self.is_manual_intervention = False

# ==================== 工具函数 ====================
def find_haarcascade():
    paths = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
        os.path.expanduser('~/.local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'),
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"[INFO] Haar: {p}")
            return p
    
    try:
        r = subprocess.run(['find', '/', '-name', 'haarcascade_frontalface_default.xml', '-type', 'f'],
                          capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            path = r.stdout.strip().split()[0]
            print(f"[INFO] Haar: {path}")
            return path
    except:
        pass
    
    return None

def crc16(d):
    crc = 0
    for b in d:
        tmp = ((crc >> 8) ^ b) & 0xFF
        crc = ((crc << 8) ^ CRC16_TABLE[tmp]) & 0xFFFF
    return crc

def send_gimbal(yaw, pitch):
    yaw = max(-100, min(100, int(yaw)))
    pitch = max(-100, min(100, int(pitch)))
    
    pkt = struct.pack('<H B H H B b b', 0x6655, 0x00, 0x0002, 0x0001, 0x07, yaw, pitch)
    pkt += struct.pack('<H', crc16(pkt))
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(pkt, (CAMERA_IP, UDP_PORT))
        s.close()
    except Exception as e:
        print(f"[ERR] 发送失败：{e}")

def send_gimbal_angle(yaw_angle, pitch_angle):
    yaw_angle = max(-36000, min(36000, int(yaw_angle)))
    pitch_angle = max(-36000, min(36000, int(pitch_angle)))
    
    pkt = struct.pack('<H B H H B h h', 0x6655, 0x00, 0x0002, 0x0001, 0x01, yaw_angle, pitch_angle)
    pkt += struct.pack('<H', crc16(pkt))
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(pkt, (CAMERA_IP, UDP_PORT))
        s.close()
    except Exception as e:
        print(f"[ERR] 发送角度失败：{e}")

def initialize_gimbal():
    print("[INIT] 正在初始化云台...")
    for _ in range(5):
        send_gimbal(0, 0)
        time.sleep(0.05)
    
    for step in range(HOME_STEPS):
        speed_factor = 1.0 - (step / HOME_STEPS)
        current_speed = int(40 * speed_factor)
        send_gimbal(current_speed, current_speed)
        time.sleep(HOME_DELAY)
    
    for _ in range(5):
        send_gimbal(0, 0)
        time.sleep(0.05)
    
    for _ in range(3):
        send_gimbal_angle(0, 0)
        time.sleep(0.1)
    
    print("[INIT] 云台初始化完成")

# ==================== 键盘监听线程 ====================
def keyboard_listener():
    global manual_mode, manual_yaw, manual_pitch
    
    print("[KEY] 键盘控制已启动")
    print("  WSAD - 手动控制")
    print("  Shift+WSAD - 快速移动")
    print("  空格 - 回中")
    print("  T - 切换跟踪/手动模式")
    print("  H - 帮助")
    print("  Q - 退出")
    
    while True:
        try:
            with lock:
                yaw = 0
                pitch = 0
                boost = keyboard.is_pressed("shift")
                speed = BOOST_SPEED if boost else MANUAL_SPEED

                if keyboard.is_pressed("w"):
                    pitch = speed
                elif keyboard.is_pressed("s"):
                    pitch = -speed

                if keyboard.is_pressed("a"):
                    yaw = -speed
                elif keyboard.is_pressed("d"):
                    yaw = speed

                if keyboard.is_pressed("space"):
                    print("[CMD] 回中")
                    send_gimbal(0, 0)
                    time.sleep(0.1)
                    for _ in range(3):
                        send_gimbal_angle(0, 0)
                        time.sleep(0.1)
                    manual_yaw = 0
                    manual_pitch = 0
                    time.sleep(0.5)
                    continue

                if yaw != 0 or pitch != 0:
                    manual_mode = True
                    manual_yaw = yaw
                    manual_pitch = pitch
                    
                    direction = []
                    if pitch > 0: direction.append("上")
                    if pitch < 0: direction.append("下")
                    if yaw < 0: direction.append("左")
                    if yaw > 0: direction.append("右")
                    print(f"[MANUAL] {' '.join(direction)} ({yaw}, {pitch})")
                else:
                    if manual_mode:
                        manual_mode = False
                        print("[MODE] 恢复自动跟踪")
                        manual_yaw = 0
                        manual_pitch = 0

                if keyboard.is_pressed("t"):
                    manual_mode = not manual_mode
                    mode_str = "手动控制" if manual_mode else "自动跟踪"
                    print(f"[MODE] 切换到{mode_str}模式")
                    time.sleep(0.3)

                if keyboard.is_pressed("h"):
                    print("\n=== 控制帮助 ===")
                    print("WSAD - 手动控制云台")
                    print("Shift+WSAD - 快速移动")
                    print("空格 - 回中")
                    print("T - 切换跟踪/手动模式")
                    print("Q - 退出")
                    print("================\n")
                    time.sleep(0.5)

                if keyboard.is_pressed("q"):
                    print("[CMD] 准备退出...")
                    send_gimbal(0, 0)
                    os._exit(0)

            time.sleep(0.02)
        except Exception as e:
            print(f"[KEY] Error: {e}")
            time.sleep(0.1)

# ==================== FFmpeg Reader ====================
class FFmpegReader:
    def __init__(self, url, w=VIDEO_W, h=VIDEO_H):
        self.url = url
        self.w = w
        self.h = h
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        print(f"[FFmpeg] {self.url} {self.w}x{self.h}")
        return self

    def _read(self):
        cmd = [
            'ffmpeg', '-rtsp_transport', 'tcp',
            '-i', self.url,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-an',
            '-s', f'{self.w}x{self.h}',
            '-'
        ]
        
        while self.running:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
                sz = self.w * self.h * 3
                
                while self.running:
                    raw = p.stdout.read(sz)
                    if not raw:
                        print('[FFmpeg] 重连…')
                        break
                    with self.lock:
                        self.frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            except Exception as e:
                print(f'[FFmpeg] {e}')
                time.sleep(1)
            finally:
                if 'p' in locals():
                    p.kill()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

# ==================== 多尺度人脸检测 ====================
def detect_face_multi_scale(face_cascade, gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) == 0:
        h, w = gray.shape
        gray_large = cv2.resize(gray, (int(w * 1.2), int(h * 1.2)))
        faces = face_cascade.detectMultiScale(gray_large, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            faces = [(int(x/1.2), int(y/1.2), int(w/1.2), int(h/1.2)) for x, y, w, h in faces]
    
    return faces

# ==================== MAIN ====================
def main():
    global manual_mode, manual_yaw, manual_pitch
    
    print("=" * 60)
    print("  SIYI A8 Mini 人脸跟踪 + 键盘控制 v1.0")
    print("=" * 60)
    
    initialize_gimbal()
    
    pid_y = PIDController(KP_YAW, KI_YAW, KD_YAW, MAX_SPEED)
    pid_p = PIDController(KP_PITCH, KI_PITCH, KD_PITCH, MAX_SPEED)

    print(f"\n{'=' * 55}")
    print(f"Kp_y={KP_YAW} Ki_y={KI_YAW} Kd_y={KD_YAW}")
    print(f"Kp_p={KP_PITCH} Ki_p={KI_PITCH} Kd_p={KD_PITCH}")
    print("=" * 55)
    print("自动跟踪模式 - 按 WSAD 切手动，松开恢复跟踪")
    print("Q 退出 | R 重置 PID | O 切换多尺度")
    print("[/] 水平Kp | ;/' 垂直Kp")
    print("=" * 55 + "\n")

    key_thread = threading.Thread(target=keyboard_listener, daemon=True)
    key_thread.start()

    cascade_path = find_haarcascade()
    if not cascade_path:
        print('[ERR] 未找到 Haar  Cascade 文件！')
        print('请安装 opencv-contrib-python 或手动下载 haarcascade_frontalface_default.xml')
        exit(1)
    
    fc = cv2.CascadeClassifier(cascade_path)
    if fc.empty():
        print('[ERR] Cascade 加载失败')
        exit(1)

    print('[INFO] 正在连接 RTSP 流...')
    rd = FFmpegReader(RTSP_URL).start()
    time.sleep(2)
    
    if rd.get_frame() is None:
        print(f'[ERR] 无法连接 RTSP 流：{RTSP_URL}')
        print('请检查:')
        print('  1. 相机 IP 是否正确')
        print('  2. 相机是否在线')
        print('  3. 网络是否连通')
        rd.stop()
        exit(1)
    
    print('[SUCCESS] RTSP 流连接成功')
    
    last_sent = (None, None)
    fcnt = 0
    t_ctrl = time.time()
    use_multi_scale = True

    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Track", VIDEO_W, VIDEO_H)

    try:
        while True:
            fr = rd.get_frame()
            if fr is None:
                time.sleep(.02)
                continue
            
            fcnt += 1
            h, w = fr.shape[:2]
            cx, cy = w // 2, h // 2
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

            if use_multi_scale:
                faces = detect_face_multi_scale(fc, gray)
            else:
                faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            cv2.line(fr, (cx-25, cy), (cx+25, cy), (255, 255, 255), 1)
            cv2.line(fr, (cx, cy-25), (cx, cy+25), (255, 255, 255), 1)
            cv2.rectangle(fr, (cx-DEADZONE_OUT, cy-DEADZONE_IN), 
                         (cx+DEADZONE_OUT, cy+DEADZONE_IN), (80, 80, 80), 1)

            yaw_cmd = pitch_cmd = 0
            has_face = len(faces) > 0

            with lock:
                is_manual = manual_mode

            if is_manual:
                with lock:
                    yaw_cmd = manual_yaw
                    pitch_cmd = manual_pitch

                cv2.putText(fr, "MANUAL CONTROL", (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.putText(fr, f"yaw={yaw_cmd:+3d} pit={pitch_cmd:+3d}",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)

                if fcnt % 15 == 0:
                    print(f"[MANUAL] yaw={yaw_cmd:+3d} pit={pitch_cmd:+3d}")

            elif has_face:
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, ww, hh = largest
                fx, fy = x + ww // 2, y + hh // 2
                dx, dy = fx - cx, fy - cy

                cv2.rectangle(fr, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                cv2.circle(fr, (fx, fy), 4, (0, 0, 255), -1)
                cv2.line(fr, (cx, cy), (fx, fy), (255, 0, 0), 1)
                cv2.putText(fr, f"SZ:{ww}x{hh}", (x, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)

                if time.time() - t_ctrl >= CONTROL_DT:
                    if abs(dx) <= DEADZONE_IN:
                        yaw_cmd = 0
                    elif abs(dx) > DEADZONE_OUT:
                        yaw_cmd = pid_y.update(dx, has_face=True)
                    else:
                        ratio = (abs(dx) - DEADZONE_IN) / (DEADZONE_OUT - DEADZONE_IN)
                        yaw_cmd = pid_y.update(int(np.sign(dx) * abs(dx)), has_face=True)
                        yaw_cmd = int(yaw_cmd * ratio)
                    if abs(yaw_cmd) < 3:
                        yaw_cmd = 0

                    if abs(dy) <= DEADZONE_IN:
                        pitch_cmd = 0
                    elif abs(dy) > DEADZONE_OUT:
                        pitch_cmd = pid_p.update(-dy, has_face=True)
                    else:
                        ratio = (abs(dy) - DEADZONE_IN) / (DEADZONE_OUT - DEADZONE_IN)
                        pitch_cmd = pid_p.update(int(-np.sign(dy) * abs(dy)), has_face=True)
                        pitch_cmd = int(pitch_cmd * ratio)
                    if abs(pitch_cmd) < 3:
                        pitch_cmd = 0
                    
                    t_ctrl = time.time()

                if pid_y.is_manual_intervention or pid_p.is_manual_intervention:
                    cv2.putText(fr, "MANUAL FOLLOW", (10, 28),
                               cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                else:
                    st = "TRACK" if (yaw_cmd or pitch_cmd) else "CENTERED"
                    cv2.putText(fr, f"{st}", (10, 28),
                               cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

                cv2.putText(fr, f"dx={dx:+3d} dy={dy:+3d}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 0), 1)
                cv2.putText(fr, f"yaw={yaw_cmd:+3d} pit={pitch_cmd:+3d}", (10, 68),
                           cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)

                if fcnt % 15 == 0:
                    manual_tag = " [MANUAL]" if (pid_y.is_manual_intervention or pid_p.is_manual_intervention) else ""
                    print(f"[TRK] dx={dx:+3d} dy={dy:+3d} | yaw={yaw_cmd:+3d} pit={pitch_cmd:+3d}{manual_tag}")
            else:
                if time.time() - t_ctrl >= CONTROL_DT:
                    pid_y.update(0, has_face=False)
                    pid_p.update(0, has_face=False)
                    t_ctrl = time.time()
                cv2.putText(fr, "No Face", (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

            if (yaw_cmd, pitch_cmd) != last_sent:
                send_gimbal(yaw_cmd, pitch_cmd)
                last_sent = (yaw_cmd, pitch_cmd)

            ms = "MS" if use_multi_scale else "SS"
            cv2.putText(fr, f"Ky={pid_y.kp:.2f} Kpy={pid_p.kp:.2f} [{ms}]",
                       (10, VIDEO_H - 14), cv2.FONT_HERSHEY_SIMPLEX, .38, (180, 180, 180), 1)

            cv2.imshow("Track", fr)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            elif k in (ord('r'), ord('R')):
                pid_y.reset()
                pid_p.reset()
                print("[RST] PID 重置")
            elif k in (ord('o'), ord('O')):
                use_multi_scale = not use_multi_scale
                print(f"[TOGGLE] Multi-scale: {use_multi_scale}")
            elif k == ord('['):
                pid_y.kp = max(.01, pid_y.kp - .02)
                print(f"YawKp={pid_y.kp:.2f}")
            elif k == ord(']'):
                pid_y.kp = min(1, pid_y.kp + .02)
                print(f"YawKp={pid_y.kp:.2f}")
            elif k == ord(';'):
                pid_p.kp = max(.01, pid_p.kp - .02)
                print(f"PitKp={pid_p.kp:.2f}")
            elif k == ord("'"):
                pid_p.kp = min(1, pid_p.kp + .02)
                print(f"PitKp={pid_p.kp:.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        send_gimbal(0, 0)
        rd.stop()
        cv2.destroyAllWindows()
        print("[EXIT] 程序退出")

if __name__ == '__main__':
    main()