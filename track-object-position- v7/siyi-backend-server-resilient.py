#!/usr/bin/env python3
"""
SIYI A8 Mini 网页版后端服务 - 终极位置控制版 (修复自动跟踪不动问题 + 解除上下限幅)
核心突破：
1. 彻底废弃速度 PID，改用“目标角度积分”架构，根治过冲和迟滞。
2. 视觉偏差与手动速度均积分到目标绝对角度，云台底层电机自动平滑执行。
3. 独立 20Hz 发送线程，使用 CMD 0x0E 绝对位置控制。
4. 全局 UDP Socket 复用，消除频繁创建的系统开销。
5. 【修复】发送线程取消去重，强制每周期发送；跟踪时强制清除手动模式；优化 dt 限制。
6. 【修改】移除所有 pitch 限幅，允许上下无限制运动（硬件限位由云台自身保护）。
"""

import asyncio
import json
import socket
import struct
import logging
import os
import sys
import threading
import time
import numpy as np
import cv2
from aiohttp import web, WSMsgType

try:
    import av
except ImportError:
    print("❌ 缺少依赖: av (PyAV)")
    sys.exit(1)

# ==================== 配置环境 ====================
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DETECTOR = os.environ.get('DETECTOR_TYPE', 'haar') 
DEFAULT_TRACKER = os.environ.get('TRACKER_TYPE', 'hybrid') 

VIDEO_W = 640
VIDEO_H = 480

# ── 位置控制核心参数 ──
K_PX_TO_DEG = 0.12          # 像素转角度系数 (度/秒/像素)。如果跟踪太慢请增大，太猛请减小
DEADZONE_PX = 10            # 死区 (像素)。偏差小于此值时不调整，防止中心微抖
MANUAL_SPEED_DEG_PER_SEC = 40.0  # 手动模式下，满舵(100)对应的最大角速度 (度/秒)
REVERSE_PITCH = True        # ⚠️ 如果上下方向反了，请改为 False

MIN_FRAME_INTERVAL = 0.04   # 25 FPS 视频推送

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('siyi-pos-ctrl')

# ── 全局复用 UDP Socket ──
g_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==================== SIYI 协议封装 (CMD 0x0E) ====================
def calculate_crc16_xmodem(data: bytes) -> bytes:
    """
    标准的 CRC16-XMODEM 算法 (多项式 0x1021)
    返回 2字节的小端序 CRC 校验和
    """
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
    crc &= 0xFFFF
    return crc.to_bytes(2, byteorder='little')

def build_position_cmd(yaw_deg: float, pitch_deg: float, reverse_pitch: bool = False) -> bytes:
    """
    构建 SIYI 云台绝对位置控制指令 (CMD 0x0E)
    
    参数:
    - yaw_deg: 目标偏航角 (度)，范围 -135.0 到 135.0
    - pitch_deg: 目标俯仰角 (度)，范围 -90.0 到 25.0
    - reverse_pitch: 是否反转 Pitch 方向 (如果实际物理方向反了，设为 True)
    
    返回:
    - 完整的 UDP 发送字节流
    """
    # 1. 固定的协议 Header (包含 STX, CTRL, Data_len, SEQ, CMD_ID)
    # 55 66 (STX) | 01 (CTRL) | 04 00 (Data_len=4) | 00 00 (SEQ) | 0E (CMD_ID)
    header = bytes.fromhex("556601040000000E")
    
    # 2. 将角度转换为 0.1度 单位的整数 (例如 20.5度 -> 205)
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    
    # ⚠️ 核心修正：如果上下方向反了，在此处给 Pitch 取反
    if reverse_pitch:
        pitch_val = -pitch_val
    
    # 限制在 int16 范围内防止溢出
    yaw_val = max(-32768, min(32767, yaw_val))
    pitch_val = max(-32768, min(32767, pitch_val))
    
    # 3. 打包为 2字节 有符号小端序整数 (Little Endian)
    yaw_bytes = yaw_val.to_bytes(2, byteorder='little', signed=True)
    pitch_bytes = pitch_val.to_bytes(2, byteorder='little', signed=True)
    
    # 4. 拼接 Header 和 Data (Payload)
    payload = header + yaw_bytes + pitch_bytes
    
    # 5. 计算 CRC16 (计算范围包含整个 Header 和 Data)
    crc = calculate_crc16_xmodem(payload)
    
    # 6. 返回完整指令
    return payload + crc

# ==================== 全局控制状态 ====================
class HybridControlState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # 核心：目标绝对角度 (所有控制逻辑最终都修改这两个值)
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        
        # 手动模式状态
        self.last_manual_time = 0.0
        self.manual_mode = False
        self.manual_yaw = 0   # -100 ~ 100 (速度)
        self.manual_pitch = 0 # -100 ~ 100 (速度)
        
        # 跟踪状态
        self.tracking_mode = 'idle' 
        self.roi_request = None

g_state = HybridControlState()

# ==================== 独立发送线程 (20Hz) ====================
def send_loop():
    """
    独立线程：
    1. 处理手动模式下的速度积分。
    2. 以 20Hz 频率将 target_yaw/pitch 发送给云台。
    【修复】去掉去重逻辑，强制每周期发送，避免自动跟踪时因角度变化微小被跳过。
    【修改】移除了 pitch 限幅，允许上下无限制运动。
    """
    logger.info("🚀 20Hz 位置发送线程启动 (CMD 0x0E)")
    last_time = time.time()
    
    while video_thread.running:
        now = time.time()
        dt = now - last_time
        last_time = now
        
        with g_state.lock:
            # 1. 手动模式：将速度积分到目标角度
            if g_state.manual_mode:
                g_state.target_yaw += (g_state.manual_yaw / 100.0) * MANUAL_SPEED_DEG_PER_SEC * dt
                g_state.target_pitch += (g_state.manual_pitch / 100.0) * MANUAL_SPEED_DEG_PER_SEC * dt
                
                # 物理限位：仅保留 yaw 限幅，移除 pitch 限幅
                g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                # 【移除】pitch 不限幅，由硬件保护
            
            tgt_y = g_state.target_yaw
            tgt_p = g_state.target_pitch
            
        # 【修复】不再检查去重，强制发送位置指令（20Hz足够快，云台会自行处理重复指令）
        cmd = build_position_cmd(tgt_y, tgt_p, reverse_pitch=REVERSE_PITCH)
        try:
            g_udp_socket.sendto(cmd, (CAMERA_IP, UDP_PORT))
        except Exception as e:
            logger.error(f"UDP 发送失败: {e}")
                
        time.sleep(0.05)  # 20Hz

# ==================== 统一人脸检测器封装 ====================
class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        self.detector = None
        self.dnn_net = None
        self.yunet = None
        self.active_method = 'none'
        self._load(method)
        
    def _find_file(self, filename):
        paths = [
            os.path.join(WORKSPACE_DIR, filename),
            os.path.join(os.path.expanduser('~'), '.opencv', filename),
            f'/usr/share/opencv4/haarcascades/{filename}' if 'cascade' in filename else filename,
            f'/usr/local/share/opencv4/haarcascades/{filename}' if 'cascade' in filename else filename,
        ]
        for p in paths:
            if os.path.exists(p): return p
        return None

    def _load(self, method):
        self.method = method
        if method == 'haar':
            path = self._find_file('haarcascade_frontalface_default.xml')
            if path: 
                self.detector = cv2.CascadeClassifier(path)
                self.active_method = 'haar'
                logger.info(f"✅ 加载 Haar 人脸检测器成功")
            else: logger.error("❌ 未找到 Haar 模型文件")
        elif method == 'lbp':
            path = self._find_file('lbpcascade_frontalface.xml')
            if path: 
                self.detector = cv2.CascadeClassifier(path)
                self.active_method = 'lbp'
                logger.info(f"✅ 加载 LBP 人脸检测器成功")
            else:
                logger.warning("⚠️ 未找到 LBP 模型文件，自动降级为 Haar")
                self._load('haar')
        elif method == 'dnn':
            try:
                path = self._find_file('face_detection_yunet_2023mar.onnx')
                if path:
                    self.yunet = cv2.FaceDetectorYN.create(path, "", (VIDEO_W, VIDEO_H), 0.9, 0.3, 5000)
                    self.active_method = 'dnn_yunet'
                    logger.info(f"✅ 加载 DNN (YuNet) 人脸检测器成功 (高精度)")
                    return
            except AttributeError: pass
            prototxt = self._find_file('deploy.prototxt')
            model = self._find_file('res10_300x300_ssd_iter_140000.caffemodel')
            if prototxt and model:
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
                self.active_method = 'dnn_ssd'
                logger.info(f"✅ 加载 DNN (SSD) 人脸检测器成功 (高精度)")
                return
            logger.warning("⚠️ 未找到 DNN 模型文件，自动降级为 Haar。")
            self._load('haar')

    def detect(self, frame):
        if self.active_method in ['haar', 'lbp'] and self.detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
            return faces if len(faces) > 0 else []
        elif self.active_method == 'dnn_yunet' and self.yunet is not None:
            h, w = frame.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is None: return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]
        elif self.active_method == 'dnn_ssd' and self.dnn_net is not None:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    faces.append((int(x), int(y), int(x1-x), int(y1-y)))
            return faces
        return []

# ==================== 统一物体跟踪器封装 ====================
class ObjectTracker:
    def __init__(self, method='hybrid'):
        self.method = method
        self.tracker = None
        self.dnn_net = None
        self.active_method = 'none'
        self.fast_tracker = None      
        self.accurate_tracker = None  
        self.calibrate_interval = 5   
        self.frame_count = 0
        self.last_fast_bbox = None
        self.last_accurate_bbox = None
        self.ssd_blob_params = None 
        self._load(method)

    def _find_file(self, filename):
        paths = [os.path.join(WORKSPACE_DIR, filename), os.path.join(os.path.expanduser('~'), '.opencv', filename)]
        for p in paths:
            if os.path.exists(p): return p
        return None

    def _create_legacy_tracker(self, name):
        try: return getattr(cv2.legacy, f'Tracker{name}_create')()
        except AttributeError: pass
        try: return getattr(cv2, f'Tracker{name}_create')()
        except AttributeError: pass
        return None

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _load(self, method):
        self.method = method
        self.frame_count = 0; self.last_fast_bbox = None; self.last_accurate_bbox = None; self.ssd_blob_params = None
        if method == 'hybrid':
            self.fast_tracker = self._create_legacy_tracker('MOSSE')
            self.accurate_tracker = self._create_legacy_tracker('CSRT')
            if self.fast_tracker and self.accurate_tracker: self.active_method = 'hybrid'; logger.info(f"✅ 加载 Hybrid")
            else: self._fallback_tracker('Hybrid')
        elif method == 'mosse':
            self.tracker = self._create_legacy_tracker('MOSSE')
            if self.tracker: self.active_method = 'mosse'; logger.info("✅ 加载 MOSSE")
            else: self._fallback_tracker('MOSSE')
        elif method == 'kcf':
            self.tracker = self._create_legacy_tracker('KCF')
            if self.tracker: self.active_method = 'kcf'; logger.info("✅ 加载 KCF")
            else: self._fallback_tracker('KCF')
        elif method == 'csrt':
            self.tracker = self._create_legacy_tracker('CSRT')
            if self.tracker: self.active_method = 'csrt'; logger.info("✅ 加载 CSRT")
            else: self._fallback_tracker('CSRT')
        elif method == 'dasiamrpn':
            model = self._find_file('dasiamrpn_model.onnx')
            kernel_cls1 = self._find_file('dasiamrpn_kernel_cls1.onnx')
            kernel_r1 = self._find_file('dasiamrpn_kernel_r1.onnx')
            if model and kernel_cls1 and kernel_r1:
                try:
                    params = cv2.TrackerDaSiamRPN_Params()
                    params.model = model; params.kernel_cls1 = kernel_cls1; params.kernel_r1 = kernel_r1
                    self.tracker = cv2.TrackerDaSiamRPN_create(params)
                    self.active_method = 'dasiamrpn'; logger.info("✅ 加载 DaSiamRPN")
                except: self._fallback_tracker('DaSiamRPN')
            else: self._load('hybrid')
        elif method == 'ssd':
            prototxt = self._find_file('deploy.prototxt')
            model = self._find_file('res10_300x300_ssd_iter_140000.caffemodel')
            if prototxt and model:
                try:
                    self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
                    self.ssd_blob_params = {'scale': 1.0, 'mean': (104.0, 177.0, 123.0)}
                    self.active_method = 'ssd'; logger.info("✅ 加载 SSD")
                except: self._fallback_tracker('SSD')
            else: self._load('hybrid')

    def _fallback_tracker(self, failed_name):
        for fallback in ['hybrid', 'csrt', 'kcf', 'mosse']:
            if fallback == 'hybrid':
                self.fast_tracker = self._create_legacy_tracker('MOSSE')
                self.accurate_tracker = self._create_legacy_tracker('CSRT')
                if self.fast_tracker and self.accurate_tracker: self.active_method = 'hybrid'; return
            else:
                self.tracker = self._create_legacy_tracker(fallback.upper())
                if self.tracker: self.active_method = fallback; return
        self.active_method = 'none'

    def init(self, frame, bbox):
        if self.active_method == 'ssd': self.last_fast_bbox = bbox; return True
        elif self.active_method == 'hybrid':
            try:
                self.fast_tracker.init(frame, bbox); self.accurate_tracker.init(frame, bbox)
                self.last_fast_bbox = bbox; self.last_accurate_bbox = bbox; self.frame_count = 0; return True
            except: return False
        elif self.tracker:
            try: self.tracker.init(frame, bbox); self.last_fast_bbox = bbox; return True
            except: return False
        return False

    def update(self, frame):
        self.frame_count += 1
        if self.active_method == 'ssd' and self.dnn_net:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), self.ssd_blob_params['scale'], (300, 300), self.ssd_blob_params['mean'])
            self.dnn_net.setInput(blob); detections = self.dnn_net.forward()
            valid_boxes = []
            for i in range(detections.shape[2]):
                if detections[0, 0, i, 2] > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int"); valid_boxes.append((int(x), int(y), int(x1-x), int(y1-y)))
            if not valid_boxes: return False, None
            if self.last_fast_bbox:
                best_iou, best_box = 0, None
                for box in valid_boxes:
                    iou = self._calculate_iou(self.last_fast_bbox, box)
                    if iou > best_iou: best_iou = iou; best_box = box
                if best_box and best_iou > 0.2: self.last_fast_bbox = best_box; return True, best_box
            return False, None

        if self.active_method == 'hybrid':
            success_fast, bbox_fast = self.fast_tracker.update(frame)
            if success_fast: self.last_fast_bbox = bbox_fast
            if self.frame_count % self.calibrate_interval == 0:
                success_acc, bbox_acc = self.accurate_tracker.update(frame)
                if success_acc and bbox_acc:
                    self.last_accurate_bbox = bbox_acc
                    try: self.fast_tracker.init(frame, bbox_acc)
                    except: pass
            if self.last_accurate_bbox and self.frame_count % self.calibrate_interval == 0: return True, self.last_accurate_bbox
            elif self.last_fast_bbox: return True, self.last_fast_bbox
            else: return False, None

        elif self.tracker:
            try:
                success, bbox = self.tracker.update(frame)
                if success: self.last_fast_bbox = bbox
                else: self.last_fast_bbox = None
                return success, bbox
            except: self.last_fast_bbox = None; return False, None
        return False, None

# ==================== 智能识别与视频采集核心线程 ====================
class VideoAndTrackingThread(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.last_frame = None
        self.last_frame_time = time.time()
        self.running = True
        self.connected = False
        self.lock = threading.Lock()
        
        self.face_detector = FaceDetector(method=DEFAULT_DETECTOR)
        self.object_tracker = ObjectTracker(method=DEFAULT_TRACKER)
            
        self.last_control_t = time.time()
        
        self.tracking_mode = 'idle' 
        self.roi_to_init = None
        self.obj_lost_count = 0
        
        self.smooth_fx = 0.0
        self.smooth_fy = 0.0

    def run(self):
        frame_counter = 0
        
        while self.running:
            container = None
            try:
                logger.info(f"🚀 RTSP 连接中: {self.rtsp_url.split('@')[-1]}")
                container = av.open(self.rtsp_url, options={
                    'rtsp_transport': 'tcp', 'timeout': '3000000', 'stimeout': '3000000',
                    'probesize': '153600', 'analyzeduration': '500000'
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
                            
                        cx, cy = VIDEO_W // 2, VIDEO_H // 2
                        now_t = time.time()
                        
                        with g_state.lock:
                            is_manual = g_state.manual_mode
                            if g_state.roi_request is not None:
                                self.roi_to_init = g_state.roi_request
                                g_state.roi_request = None

                        if is_manual:
                            cv2.putText(bgr_img, "MODE: MANUAL CONTROL", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        else:
                            if self.roi_to_init is not None:
                                rx, ry, rw, rh = self.roi_to_init
                                bbox = (int(rx*VIDEO_W), int(ry*VIDEO_H), int(rw*VIDEO_W), int(rh*VIDEO_H))
                                bbox = (max(0, bbox[0]), max(0, bbox[1]), min(bbox[2], VIDEO_W-bbox[0]), min(bbox[3], VIDEO_H-bbox[1]))
                                
                                if bbox[2] > 10 and bbox[3] > 10:
                                    roi_img = bgr_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                                    faces_in_roi = self.face_detector.detect(roi_img)
                                    
                                    if len(faces_in_roi) > 0:
                                        self.tracking_mode = 'face'
                                        with g_state.lock: g_state.tracking_mode = 'face'
                                        logger.info(f"👤 框选区域检测到人脸，切入【人脸跟踪模式】")
                                    else:
                                        if self.object_tracker.init(bgr_img, bbox):
                                            self.obj_lost_count = 0
                                            self.smooth_fx = bbox[0] + bbox[2] // 2
                                            self.smooth_fy = bbox[1] + bbox[3] // 2
                                            self.tracking_mode = 'object'
                                            with g_state.lock: g_state.tracking_mode = 'object'
                                            logger.info(f"🎯 切入【物体跟踪模式】")
                                        else:
                                            self.tracking_mode = 'idle'
                                            with g_state.lock: g_state.tracking_mode = 'idle'
                                self.roi_to_init = None

                            if self.tracking_mode == 'idle':
                                cv2.line(bgr_img, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
                                cv2.line(bgr_img, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
                                cv2.putText(bgr_img, "MODE: IDLE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                                
                            elif self.tracking_mode == 'object':
                                success, bbox = self.object_tracker.update(bgr_img)
                                cv2.line(bgr_img, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
                                cv2.line(bgr_img, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

                                if success and bbox is not None:
                                    self.obj_lost_count = 0
                                    px, py, pw, ph = [int(v) for v in bbox]
                                    fx_raw, fy_raw = px + pw // 2, py + ph // 2
                                    
                                    self.smooth_fx = 0.7 * fx_raw + 0.3 * self.smooth_fx
                                    self.smooth_fy = 0.7 * fy_raw + 0.3 * self.smooth_fy
                                    fx, fy = int(self.smooth_fx), int(self.smooth_fy)
                                    
                                    dx, dy = fx - cx, fy - cy
                                    
                                    # ⭐ 核心：位置积分逻辑
                                    dt = now_t - self.last_control_t
                                    # 【修复】限制 dt 最大为 0.1s，避免第一帧因 dt 过大被跳过
                                    dt = min(dt, 0.1)
                                    if dt > 0:  # 保证非零
                                        # 【修复】强制清除手动模式，防止鼠标误触残留
                                        with g_state.lock:
                                            g_state.manual_mode = False
                                            if abs(dx) > DEADZONE_PX:
                                                g_state.target_yaw += dx * K_PX_TO_DEG * dt
                                            if abs(dy) > DEADZONE_PX:
                                                # 图像 Y 向下，云台 Pitch 向上，故取反
                                                g_state.target_pitch += -dy * K_PX_TO_DEG * dt
                                            
                                            # 软限位：仅保留 yaw 限幅，移除 pitch 限幅
                                            g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                                            # 【移除】pitch 不限幅，由硬件保护
                                        self.last_control_t = now_t

                                    cv2.rectangle(bgr_img, (px, py), (px + pw, py + ph), (255, 100, 0), 2)
                                    cv2.circle(bgr_img, (fx, fy), 4, (0, 0, 255), -1)
                                    cv2.putText(bgr_img, f"OBJ POS MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                                else:
                                    self.obj_lost_count += 1
                                    cv2.putText(bgr_img, f"LOST ({self.obj_lost_count}/15)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    if self.obj_lost_count > 15:
                                        self.tracking_mode = 'idle'
                                        with g_state.lock: g_state.tracking_mode = 'idle'
                                        
                            elif self.tracking_mode == 'face':
                                if frame_counter % 2 == 0:
                                    faces = self.face_detector.detect(bgr_img)
                                has_face = len(faces) > 0
                                cv2.line(bgr_img, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
                                cv2.line(bgr_img, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
                                
                                if has_face:
                                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                                    x, y, ww, hh = largest_face
                                    fx, fy = x + ww // 2, y + hh // 2
                                    dx, dy = fx - cx, fy - cy
                                    
                                    dt = now_t - self.last_control_t
                                    dt = min(dt, 0.1)  # 【修复】同样限制 dt
                                    if dt > 0:
                                        # 【修复】强制清除手动模式
                                        with g_state.lock:
                                            g_state.manual_mode = False
                                            if abs(dx) > DEADZONE_PX:
                                                g_state.target_yaw += dx * K_PX_TO_DEG * dt
                                            if abs(dy) > DEADZONE_PX:
                                                g_state.target_pitch += -dy * K_PX_TO_DEG * dt
                                            
                                            # 软限位：仅保留 yaw 限幅，移除 pitch 限幅
                                            g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                                            # 【移除】pitch 不限幅，由硬件保护
                                        self.last_control_t = now_t

                                    cv2.rectangle(bgr_img, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                                    cv2.circle(bgr_img, (fx, fy), 4, (0, 0, 255), -1)
                                    cv2.putText(bgr_img, f"FACE POS MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    cv2.putText(bgr_img, "SEARCHING...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # 绘制 OSD 目标角度
                        with g_state.lock:
                            osd_text = f"Target: Y={g_state.target_yaw:+5.1f} P={g_state.target_pitch:+5.1f}"
                        cv2.putText(bgr_img, osd_text, (15, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        _, buf = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                        with self.lock:
                            self.last_frame = buf.tobytes()
                            self.last_frame_time = time.time()
                            
            except Exception as e:
                logger.error(f"❌ 视频线程异常: {e}")
                self.connected = False
            finally:
                if container:
                    try: container.close()
                    except: pass
                if self.running: time.sleep(1)

    def get_latest_frame(self):
        if time.time() - self.last_frame_time > 4.0: self.connected = False
        with self.lock: return self.last_frame

    def stop(self): self.running = False

video_thread = VideoAndTrackingThread(RTSP_URL)
video_thread.start()

# 启动 20Hz 发送线程
threading.Thread(target=send_loop, daemon=True).start()

# ==================== 后端路由与 API ====================
async def gimbal_watchdog_ctx(app):
    logger.info("🕵️ 智能混合看门狗就绪")
    async def watchdog_loop():
        try:
            while True:
                await asyncio.sleep(0.05)
                now = time.time()
                with g_state.lock:
                    if g_state.manual_mode:
                        # 如果超过 0.35 秒没收到手动指令，自动退出手动模式
                        # 退出后，云台会停在当前的 target 角度，不会乱跑
                        if now - g_state.last_manual_time > 0.35:
                            g_state.manual_mode = False
                            g_state.manual_yaw = 0; g_state.manual_pitch = 0
        except asyncio.CancelledError: pass
    task = asyncio.create_task(watchdog_loop())
    yield
    task.cancel()
    await task

async def handle_gimbal_control(request):
    try:
        data = await request.json()
        yaw = int(data.get('yaw', 0)); pitch = int(data.get('pitch', 0))
        with g_state.lock:
            g_state.last_manual_time = time.time()
            g_state.manual_mode = True
            g_state.manual_yaw = yaw; g_state.manual_pitch = pitch
        return web.json_response({'status': 'success'})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_tracker_init(request):
    try:
        data = await request.json()
        x = float(data.get('x', 0)); y = float(data.get('y', 0))
        w = float(data.get('width', 0)); h = float(data.get('height', 0))
        with g_state.lock:
            g_state.roi_request = (x, y, w, h)
        return web.json_response({'status': 'success'})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_tracker_reset(request):
    with g_state.lock:
        g_state.tracking_mode = 'idle'
        g_state.roi_request = None
    video_thread.tracking_mode = 'idle'
    video_thread.roi_to_init = None
    return web.json_response({'status': 'success'})

async def handle_detector_switch(request):
    try:
        data = await request.json()
        method = data.get('method', 'haar')
        if method not in ['haar', 'lbp', 'dnn']:
            return web.json_response({'status': 'error', 'reason': 'Invalid method.'}, status=400)
        video_thread.face_detector._load(method)
        return web.json_response({'status': 'success', 'active_method': video_thread.face_detector.active_method})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_tracker_switch(request):
    try:
        data = await request.json()
        method = data.get('method', 'hybrid')
        if method not in ['hybrid', 'mosse', 'kcf', 'csrt', 'dasiamrpn', 'ssd']:
            return web.json_response({'status': 'error', 'reason': 'Invalid method.'}, status=400)
        video_thread.object_tracker._load(method)
        return web.json_response({'status': 'success', 'active_method': video_thread.object_tracker.active_method})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_status(request):
    with g_state.lock:
        tgt_y = g_state.target_yaw
        tgt_p = g_state.target_pitch
        
    return web.json_response({
        'rtsp_connected': video_thread.connected, 
        'system_mode': 'manual' if g_state.manual_mode else 'auto',
        'tracking_mode': g_state.tracking_mode,
        'detector_type': video_thread.face_detector.active_method,
        'tracker_type': video_thread.object_tracker.active_method,
        'target_yaw': round(tgt_y, 2),
        'target_pitch': round(tgt_p, 2)
    })

async def handle_static(request):
    path = request.match_info.get('path', 'index.html') or 'index.html'
    file_path = os.path.join(WORKSPACE_DIR, path)
    if os.path.exists(file_path): return web.FileResponse(file_path)
    return web.Response(status=404)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    last_sent_frame_time = 0
    try:
        while not ws.closed:
            current_frame = video_thread.get_latest_frame()
            current_frame_time = video_thread.last_frame_time
            if current_frame and current_frame_time > last_sent_frame_time:
                now = time.time()
                if now - last_sent_frame_time >= MIN_FRAME_INTERVAL:
                    try:
                        await ws.send_bytes(current_frame)
                        last_sent_frame_time = now
                    except: break
                else:
                    await asyncio.sleep(MIN_FRAME_INTERVAL - (now - last_sent_frame_time))
            else:
                await asyncio.sleep(0.01)
    finally:
        return ws

async def init_app():
    app = web.Application()
    app.router.add_get('/api/ws', websocket_handler)
    app.router.add_post('/api/gimbal/control', handle_gimbal_control)
    app.router.add_post('/api/tracker/init', handle_tracker_init)
    app.router.add_post('/api/tracker/reset', handle_tracker_reset)
    app.router.add_post('/api/detector/switch', handle_detector_switch)
    app.router.add_post('/api/tracker/switch', handle_tracker_switch)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/{path:.*}', handle_static)
    app.cleanup_ctx.append(gimbal_watchdog_ctx)
    return app

if __name__ == '__main__':
    try:
        app = asyncio.get_event_loop().run_until_complete(init_app())
        web.run_app(app, host='0.0.0.0', port=SERVER_PORT)
    except KeyboardInterrupt: pass
    finally:
        video_thread.stop()
        g_udp_socket.close()
