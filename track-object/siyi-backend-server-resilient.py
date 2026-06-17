#!/usr/bin/env python3
"""
SIYI A8 Mini 网页版后端服务 - 人脸/物体双模跟踪 + 手动按键智能融合版
1. 默认自动人脸跟踪。
2. 用户在前端框选物体 -> 切换到物体跟踪（KCF）。
3. 键盘按键介入 -> 立刻压制所有自动跟踪，切入纯手动。
4. 按键停止后，看门狗切回之前的自动跟踪模式（人脸或物体）。
"""

import asyncio
import json
import socket
import struct
import logging
import os
import sys
import io
import threading
import time
import numpy as np
import cv2
from collections import deque
from aiohttp import web, WSMsgType

# 依赖检查
try:
    import av
except ImportError:
    print("❌ 缺少依赖: av (PyAV)")
    print("   安装: pip install av")
    sys.exit(1)

# 配置环境
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

VIDEO_W = 640
VIDEO_H = 480

# ── PID 参数 ──────────────────────────
KP_YAW, KI_YAW, KD_YAW = 0.25, 0.001, 0.04
KP_PITCH, KI_PITCH, KD_PITCH = 0.18, 0.0008, 0.025
MAX_SPEED = 100
DEADZONE_IN = 5
DEADZONE_OUT = 12
ALPHA_ERR = 0.3
OUT_AVG_WIN = 3
CONTROL_DT = 0.025
LOST_DECAY_RATE = 0.95
RECOVERY_FRAMES = 10
MANUAL_INTERVENTION_THRESHOLD = 150

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('siyi-merged')

# ==================== PID 控制器（同原版）====================
class PIDController:
    def __init__(self, kp, ki, kd, max_spd=100):
        self.kp = kp; self.ki = ki; self.kd = kd; self.max_spd = max_spd
        self.integral = 0.0; self.prev_err = 0.0; self.filt_err = 0.0
        self.last_t = time.time(); self.out_buf = deque(maxlen=OUT_AVG_WIN)
        self.lost_frames = 0
        self.recovery_counter = 0
        self.is_manual_intervention = False

    def update(self, raw_err, has_target=True):
        now = time.time(); dt = now - self.last_t; self.last_t = now
        if dt <= 0: dt = CONTROL_DT

        if not has_target:
            self.lost_frames += 1
            self.integral *= LOST_DECAY_RATE
            self.filt_err *= LOST_DECAY_RATE
            self.recovery_counter = 0
            return 0

        if self.lost_frames > 5 and abs(raw_err) > MANUAL_INTERVENTION_THRESHOLD:
            self.is_manual_intervention = True
            self.recovery_counter = RECOVERY_FRAMES
            self.integral = 0; self.filt_err = 0; self.prev_err = 0

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
        self.integral = 0.0; self.prev_err = 0.0; self.filt_err = 0.0
        self.out_buf.clear(); self.lost_frames = 0; self.recovery_counter = 0
        self.is_manual_intervention = False

# ==================== 全局状态（增强，加入物体跟踪）====================
class HybridControlState:
    def __init__(self):
        self.last_manual_time = 0.0
        self.manual_mode = False       # 手动压制
        self.manual_yaw = 0
        self.manual_pitch = 0
        self.last_sent_cmd = (0, 0)

        # 新增物体跟踪状态
        self.object_tracking_active = False
        self.object_bbox = None        # (x, y, w, h)
        self.last_auto_mode = 'face'   # 记录被手动打断前的自动模式

        self.lock = threading.Lock()

g_state = HybridControlState()

# CRC16 表（略，同原版）
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
def crc16(data):
    crc = 0
    for b in data:
        tmp = ((crc >> 8) ^ b) & 0xFF
        crc = ((crc << 8) ^ CRC16_TABLE[tmp]) & 0xFFFF
    return crc

def _execute_send(yaw, pitch):
    yaw, pitch = max(-100, min(100, int(yaw))), max(-100, min(100, int(pitch)))
    if (yaw, pitch) == g_state.last_sent_cmd:
        return True
    pkt = struct.pack('<H B H H B b b', 0x6655, 0x00, 0x0002, 0x0001, 0x07, yaw, pitch)
    pkt += struct.pack('<H', crc16(pkt))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(pkt, (CAMERA_IP, UDP_PORT))
        g_state.last_sent_cmd = (yaw, pitch)
        return True
    except Exception as e:
        logger.error(f"UDP 发送失败: {e}")
        return False

# ==================== 人脸检测器辅助 ====================
def find_haarcascade():
    paths = [
        'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

# ==================== 核心视频与跟踪线程（融合物体跟踪）====================
class VideoAndTrackingThread(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.last_frame = None
        self.last_frame_time = time.time()
        self.running = True
        self.connected = False
        self.lock = threading.Lock()

        # 人脸检测器
        cascade_p = find_haarcascade()
        if not cascade_p:
            logger.error("❌ 找不到人脸模型")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_p)
            logger.info(f"✅ 人脸模型: {cascade_p}")

        # PID 控制器
        self.pid_y = PIDController(KP_YAW, KI_YAW, KD_YAW, MAX_SPEED)
        self.pid_p = PIDController(KP_PITCH, KI_PITCH, KD_PITCH, MAX_SPEED)
        self.last_control_t = time.time()

        # 物体跟踪器（OpenCV KCF）
        self.tracker = None
        self.tracker_initialized = False

    def init_tracker(self, bbox):
        """用新的 bbox (x,y,w,h) 初始化 KCF 跟踪器"""
        try:
            self.tracker = cv2.TrackerKCF_create()
            self.tracker_initialized = False
            self.tracker_bbox = bbox  # 缓存，下一帧初始化
            logger.info(f"物体跟踪器准备初始化，bbox={bbox}")
        except Exception as e:
            logger.error(f"创建跟踪器失败: {e}")

    def run(self):
        frame_counter = 0
        faces = []
        while self.running:
            container = None
            try:
                logger.info(f"🚀 连接 RTSP: {self.rtsp_url.split('@')[-1] if '@' in self.rtsp_url else self.rtsp_url}")
                container = av.open(self.rtsp_url, options={
                    'rtsp_transport': 'tcp', 'timeout': '3000000',
                    'stimeout': '3000000', 'probesize': '153600',
                    'analyzeduration': '500000'
                })
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                self.connected = True
                self.last_frame_time = time.time()

                for packet in container.demux(stream):
                    if not self.running: break
                    if packet.size == 0: continue
                    for frame in packet.decode():
                        if not self.running: break
                        if not isinstance(frame, av.video.frame.VideoFrame): continue

                        frame_counter += 1
                        bgr_img = frame.to_ndarray(format='bgr24')
                        if bgr_img.shape[1] != VIDEO_W or bgr_img.shape[0] != VIDEO_H:
                            bgr_img = cv2.resize(bgr_img, (VIDEO_W, VIDEO_H))

                        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                        cx, cy = VIDEO_W // 2, VIDEO_H // 2

                        # 读取全局状态
                        with g_state.lock:
                            is_manual = g_state.manual_mode
                            m_yaw, m_pitch = g_state.manual_yaw, g_state.manual_pitch
                            obj_active = g_state.object_tracking_active
                            # 若物体跟踪刚被激活且 tracker 未初始化，则初始化
                            if obj_active and self.tracker is not None and not self.tracker_initialized:
                                try:
                                    self.tracker.init(bgr_img, tuple(self.tracker_bbox))
                                    self.tracker_initialized = True
                                    logger.info("物体跟踪器初始化成功")
                                except Exception as e:
                                    logger.error(f"跟踪器初始化失败: {e}")
                                    g_state.object_tracking_active = False
                                    self.tracker = None

                        yaw_cmd, pitch_cmd = 0, 0
                        now_t = time.time()

                        if is_manual:
                            # 手动模式：完全压制，停止所有跟踪，清除残影
                            faces = []
                            if self.tracker:
                                self.tracker_initialized = False
                            yaw_cmd, pitch_cmd = m_yaw, m_pitch
                            cv2.putText(bgr_img, "MODE: KEYBOARD MANUAL (ALL TRACKING OFF)", (15, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                            cv2.putText(bgr_img, f"Manual: Y:{yaw_cmd} P:{pitch_cmd}", (15, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        elif obj_active and self.tracker is not None and self.tracker_initialized:
                            # 物体跟踪模式
                            success, bbox = self.tracker.update(bgr_img)
                            if success:
                                x, y, w, h = [int(v) for v in bbox]
                                obj_cx, obj_cy = x + w//2, y + h//2
                                dx, dy = obj_cx - cx, obj_cy - cy
                                # 绘制框
                                cv2.rectangle(bgr_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                cv2.circle(bgr_img, (obj_cx, obj_cy), 4, (0, 255, 255), -1)
                                cv2.line(bgr_img, (cx, cy), (obj_cx, obj_cy), (255, 165, 0), 1)

                                # PID 控制
                                if now_t - self.last_control_t >= CONTROL_DT:
                                    yaw_cmd = self._pid_control(self.pid_y, dx, DEADZONE_IN, DEADZONE_OUT)
                                    pitch_cmd = self._pid_control(self.pid_p, -dy, DEADZONE_IN, DEADZONE_OUT)
                                    self.last_control_t = now_t
                                else:
                                    with g_state.lock:
                                        yaw_cmd, pitch_cmd = g_state.last_sent_cmd

                                cv2.putText(bgr_img, "MODE: OBJECT TRACKING", (15, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                cv2.putText(bgr_img, f"Obj Error: dx:{dx} dy:{dy}", (15, 55),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            else:
                                # 跟踪丢失，停止物体跟踪，切回人脸模式
                                with g_state.lock:
                                    g_state.object_tracking_active = False
                                    g_state.last_auto_mode = 'face'
                                self.tracker = None
                                self.tracker_initialized = False
                                logger.info("物体跟踪丢失，切回人脸模式")
                                yaw_cmd, pitch_cmd = 0, 0  # 暂时不动

                        else:
                            # 人脸跟踪模式（默认）
                            if self.tracker:
                                self.tracker_initialized = False  # 确保下次重新初始化

                            if self.face_cascade and frame_counter % 2 == 0:
                                faces = self.face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))

                            has_face = len(faces) > 0
                            # 画十字和死区
                            cv2.line(bgr_img, (cx-20, cy), (cx+20, cy), (255,255,255),1)
                            cv2.line(bgr_img, (cx, cy-20), (cx, cy+20), (255,255,255),1)
                            cv2.rectangle(bgr_img, (cx-DEADZONE_OUT, cy-DEADZONE_IN),
                                          (cx+DEADZONE_OUT, cy+DEADZONE_IN), (100,100,100),1)

                            if has_face:
                                largest = max(faces, key=lambda f: f[2]*f[3])
                                x, y, w, h = largest
                                fx, fy = x+w//2, y+h//2
                                dx, dy = fx - cx, fy - cy
                                cv2.rectangle(bgr_img, (x,y), (x+w,y+h), (0,255,0),2)
                                cv2.circle(bgr_img, (fx,fy), 4, (0,0,255),-1)
                                cv2.line(bgr_img, (cx,cy), (fx,fy), (255,0,0),1)

                                if now_t - self.last_control_t >= CONTROL_DT:
                                    yaw_cmd = self._pid_control(self.pid_y, dx, DEADZONE_IN, DEADZONE_OUT)
                                    pitch_cmd = self._pid_control(self.pid_p, -dy, DEADZONE_IN, DEADZONE_OUT)
                                    self.last_control_t = now_t
                                else:
                                    with g_state.lock:
                                        yaw_cmd, pitch_cmd = g_state.last_sent_cmd

                                cv2.putText(bgr_img, "MODE: AUTO FACE TRACKING", (15,30),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                                cv2.putText(bgr_img, f"Face Err: dx:{dx} dy:{dy}", (15,55),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                            else:
                                if now_t - self.last_control_t >= CONTROL_DT:
                                    self.pid_y.update(0, has_target=False)
                                    self.pid_p.update(0, has_target=False)
                                    self.last_control_t = now_t
                                cv2.putText(bgr_img, "MODE: AUTO (Searching...)", (15,30),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

                        # 统一加锁发送
                        with g_state.lock:
                            if g_state.manual_mode:
                                _execute_send(g_state.manual_yaw, g_state.manual_pitch)
                            else:
                                _execute_send(yaw_cmd, pitch_cmd)

                        # 编码推送
                        _, buf = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                        with self.lock:
                            self.last_frame = buf.tobytes()
                            self.last_frame_time = time.time()

            except Exception as e:
                logger.error(f"视频线程异常: {e}")
                self.connected = False
            finally:
                if container:
                    try: container.close()
                    except: pass
                if self.running:
                    time.sleep(2 if (time.time() - self.last_frame_time > 5) else 0.4)

    def _pid_control(self, pid, err, dead_in, dead_out):
        if abs(err) <= dead_in:
            return 0
        elif abs(err) > dead_out:
            return pid.update(err, has_target=True)
        else:
            ratio = (abs(err) - dead_in) / (dead_out - dead_in + 1e-5)
            return int(pid.update(err, has_target=True) * ratio)

    def get_latest_frame(self):
        if time.time() - self.last_frame_time > 4.0:
            if self.connected:
                logger.warning("⚠️ 视频流超时")
                self.connected = False
        with self.lock:
            return self.last_frame

    def stop(self):
        self.running = False

video_thread = VideoAndTrackingThread(RTSP_URL)
video_thread.start()

# ==================== 看门狗上下文 ====================
async def gimbal_watchdog_ctx(app):
    logger.info("🕵️ 看门狗就绪")
    async def watchdog_loop():
        try:
            while True:
                await asyncio.sleep(0.05)
                now = time.time()
                with g_state.lock:
                    if g_state.manual_mode:
                        if now - g_state.last_manual_time > 0.35:
                            g_state.manual_mode = False
                            g_state.manual_yaw = 0
                            g_state.manual_pitch = 0
                            _execute_send(0, 0)
                            logger.info("手动模式超时，切回自动跟踪")
                        elif now - g_state.last_manual_time > 0.20:
                            if g_state.manual_yaw != 0 or g_state.manual_pitch != 0:
                                g_state.manual_yaw = 0
                                g_state.manual_pitch = 0
                                _execute_send(0, 0)
        except asyncio.CancelledError:
            pass
    task = asyncio.create_task(watchdog_loop())
    yield
    task.cancel()
    await task

# ==================== HTTP API ====================
async def handle_gimbal_control(request):
    """手动键盘控制"""
    try:
        data = await request.json()
        yaw = int(data.get('yaw', 0))
        pitch = int(data.get('pitch', 0))
        with g_state.lock:
            g_state.last_manual_time = time.time()
            g_state.manual_mode = True
            g_state.manual_yaw = yaw
            g_state.manual_pitch = pitch
            _execute_send(yaw, pitch)
        return web.json_response({'status': 'success', 'mode': 'manual'})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_start_tracking(request):
    """前端框选物体后调用，启动物体跟踪"""
    try:
        data = await request.json()
        x = int(data['x'])
        y = int(data['y'])
        w = int(data['w'])
        h = int(data['h'])
        if w <= 0 or h <= 0:
            raise ValueError("框选尺寸无效")
        with g_state.lock:
            g_state.object_tracking_active = True
            g_state.object_bbox = (x, y, w, h)
            g_state.last_auto_mode = 'object'
            # 初始化跟踪器
            video_thread.init_tracker((x, y, w, h))
            # 如果当前是手动模式，我们不立刻退出手动，让看门狗处理。
            # 看门狗在手动结束后会根据 last_auto_mode 恢复物体跟踪
        logger.info(f"启动物体跟踪: {x},{y},{w},{h}")
        return web.json_response({'status': 'success', 'tracking': 'object'})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_stop_tracking(request):
    """停止物体跟踪，切回人脸"""
    with g_state.lock:
        g_state.object_tracking_active = False
        g_state.last_auto_mode = 'face'
    video_thread.tracker = None
    video_thread.tracker_initialized = False
    logger.info("物体跟踪已停止")
    return web.json_response({'status': 'success', 'tracking': 'stopped'})

async def handle_status(request):
    with g_state.lock:
        mode = 'manual' if g_state.manual_mode else ('object_track' if g_state.object_tracking_active else 'auto_face')
    return web.json_response({
        'camera_ip': CAMERA_IP,
        'rtsp_connected': video_thread.connected,
        'system_mode': mode
    })

async def handle_static(request):
    path = request.match_info.get('path', 'index.html') or 'index.html'
    file_path = os.path.join(WORKSPACE_DIR, path)
    if os.path.exists(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket 连接")
    last_sent = 0
    try:
        while not ws.closed:
            frame = video_thread.get_latest_frame()
            t = video_thread.last_frame_time
            if frame and t > last_sent:
                await ws.send_bytes(frame)
                last_sent = t
            await asyncio.sleep(0.01)
    finally:
        logger.info("WebSocket 断开")
    return ws

async def init_app():
    app = web.Application()
    app.router.add_get('/api/ws', websocket_handler)
    app.router.add_post('/api/gimbal/control', handle_gimbal_control)
    app.router.add_post('/api/gimbal/start_tracking', handle_start_tracking)
    app.router.add_post('/api/gimbal/stop_tracking', handle_stop_tracking)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/{path:.*}', handle_static)
    app.cleanup_ctx.append(gimbal_watchdog_ctx)
    return app

if __name__ == '__main__':
    try:
        app = asyncio.get_event_loop().run_until_complete(init_app())
        web.run_app(app, host='0.0.0.0', port=SERVER_PORT)
    except KeyboardInterrupt:
        pass
    finally:
        video_thread.stop()
        with g_state.lock:
            _execute_send(0, 0)
        logger.info("服务安全退出")
