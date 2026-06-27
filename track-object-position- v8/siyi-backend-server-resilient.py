#!/usr/bin/env python3
"""
SIYI A8 Mini 网页版后端服务 - 终极位置控制版 (优化响应速度)
优化项：
- K_PX_TO_DEG: 0.12 → 0.22
- DEADZONE_PX: 10 → 5
- 平滑系数: 0.7 → 0.5
- 发送频率: 20Hz → 30Hz
- dt 上限: 0.1s → 0.15s
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
    print("❌ 缺少依赖：av (PyAV)")
    sys.exit(1)

# ==================== 配置环境 ====================
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DETECTOR = os.environ.get('DETECTOR_TYPE', 'haar') 
DEFAULT_TRACKER = os.environ.get('TRACKER_TYPE', 'hybrid')
YOLO_MODEL = os.environ.get('YOLO_MODEL', 'yolov8n.pt')  # YOLOv8nano 

VIDEO_W = 640
VIDEO_H = 480

# ── 位置控制核心参数（优化版） ──
K_PX_TO_DEG = 0.22          # 增大系数，加快响应
DEADZONE_PX = 5             # 减小死区，更敏感
MANUAL_SPEED_DEG_PER_SEC = 80.0  # 手动速度也相应提升
REVERSE_PITCH = True        # 如果上下方向反了，请改为 False

MIN_FRAME_INTERVAL = 0.03   # 约33 FPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('siyi-pos-ctrl')

# YOLO 导入（在 logger 定义之后）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO (ultralytics) 已加载")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLO (ultralytics) 未安装，YOLO 检测功能不可用")

# ── 全局复用 UDP Socket ──
g_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==================== SIYI 协议封装 (CMD 0x0E) ====================
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
    return crc.to_bytes(2, byteorder='little')

def build_position_cmd(yaw_deg: float, pitch_deg: float, reverse_pitch: bool = False) -> bytes:
    header = bytes.fromhex("556601040000000E")
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    if reverse_pitch:
        pitch_val = -pitch_val
    yaw_val = max(-32768, min(32767, yaw_val))
    pitch_val = max(-32768, min(32767, pitch_val))
    yaw_bytes = yaw_val.to_bytes(2, byteorder='little', signed=True)
    pitch_bytes = pitch_val.to_bytes(2, byteorder='little', signed=True)
    payload = header + yaw_bytes + pitch_bytes
    crc = calculate_crc16_xmodem(payload)
    return payload + crc

# ==================== 全局控制状态 ====================
class HybridControlState:
    def __init__(self):
        self.lock = threading.Lock()
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.last_manual_time = 0.0
        self.manual_mode = False
        self.manual_yaw = 0
        self.manual_pitch = 0
        self.tracking_mode = 'idle' 
        self.roi_request = None
        # YOLO 点击选择相关
        self.yolo_click_select = None  # (click_x, click_y) 点击坐标
        self.yolo_enabled = False  # YOLO 是否启用

g_state = HybridControlState()

# ==================== 独立发送线程 (30Hz) ====================
def send_loop():
    logger.info("🚀 30Hz 位置发送线程启动 (CMD 0x0E)")
    last_time = time.time()
    while video_thread.running:
        now = time.time()
        dt = now - last_time
        last_time = now
        
        with g_state.lock:
            if g_state.manual_mode:
                g_state.target_yaw += (g_state.manual_yaw / 100.0) * MANUAL_SPEED_DEG_PER_SEC * dt
                g_state.target_pitch += (g_state.manual_pitch / 100.0) * MANUAL_SPEED_DEG_PER_SEC * dt
                g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                # pitch 不限幅
            tgt_y = g_state.target_yaw
            tgt_p = g_state.target_pitch
        
        cmd = build_position_cmd(tgt_y, tgt_p, reverse_pitch=REVERSE_PITCH)
        try:
            g_udp_socket.sendto(cmd, (CAMERA_IP, UDP_PORT))
        except Exception as e:
            logger.error(f"UDP 发送失败: {e}")
        time.sleep(0.033)  # 30Hz

# ==================== 人脸检测器（不变） ====================
class FaceDetector:
    # ... 保持原样 ...
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

# ==================== 物体跟踪器（不变） ====================
class YoloDetector:
    """YOLO 物体检测器 - 支持单击选择目标"""
    def __init__(self, model_name='yolov8n.pt'):
        self.model = None
        self.model_name = model_name
        self.active = False
        self.detections = []  # 当前检测结果
        if YOLO_AVAILABLE:
            self._load(model_name)
    
    def _load(self, model_name):
        try:
            self.model = YOLO(model_name)
            self.model.to('cpu')  # 使用 CPU
            self.active = True
            logger.info(f"✅ YOLO 检测器加载成功：{model_name}")
        except Exception as e:
            logger.error(f"❌ YOLO 模型加载失败：{e}")
            self.active = False
    
    def detect(self, frame, confidence=0.5):
        """检测帧中的物体，返回检测结果列表"""
        if not self.active or self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=confidence, verbose=False)
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls]
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        'class': class_name,
                        'confidence': conf,
                        'center': (int((x1+x2)/2), int((y1+y2)/2))
                    })
            self.detections = detections
            return detections
        except Exception as e:
            logger.error(f"YOLO 检测异常：{e}")
            return []
    
    def draw_detections(self, frame, detections=None):
        """在帧上绘制检测结果"""
        if detections is None:
            detections = self.detections
        
        for det in detections:
            x, y, w, h = det['bbox']
            cls = det['class']
            conf = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # 绘制标签
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 绘制中心点
            cx, cy = det['center']
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        return frame
    
    def find_nearest_detection(self, click_x, click_y, max_distance=100):
        """找到距离点击位置最近的检测结果"""
        if not self.detections:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for det in self.detections:
            cx, cy = det['center']
            dist = ((cx - click_x)**2 + (cy - click_y)**2)**0.5
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = det
        
        return nearest

class ObjectTracker:
    # ... 保持原样，省略以节省篇幅，实际使用时需复制完整代码 ...
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

# ==================== 视频采集与跟踪线程（优化平滑系数） ====================
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
        self.yolo_detector = YoloDetector(model_name=YOLO_MODEL)
            
        self.last_control_t = time.time()
        self.tracking_mode = 'idle' 
        self.roi_to_init = None
        self.obj_lost_count = 0
        self.smooth_fx = 0.0
        self.smooth_fy = 0.0
        self.yolo_active = False  # YOLO 是否激活

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

                        # 处理 YOLO 点击选择
                        with g_state.lock:
                            if g_state.yolo_click_select is not None and self.yolo_active:
                                click_x, click_y = g_state.yolo_click_select
                                g_state.yolo_click_select = None
                                
                                # 在点击位置附近查找 YOLO 检测结果
                                nearest = self.yolo_detector.find_nearest_detection(click_x, click_y, max_distance=150)
                                if nearest:
                                    bbox = nearest['bbox']
                                    self.roi_to_init = (bbox[0]/VIDEO_W, bbox[1]/VIDEO_H, bbox[2]/VIDEO_W, bbox[3]/VIDEO_H)
                                    logger.info(f"🎯 YOLO 点击选择：{nearest['class']} (conf={nearest['confidence']:.2f})")
                                else:
                                    logger.info(f"⚠️ 点击位置未检测到物体")

                        if is_manual:
                            # YOLO 检测模式 - 显示检测结果
                            if self.yolo_active:
                                detections = self.yolo_detector.detect(bgr_img, confidence=0.5)
                                bgr_img = self.yolo_detector.draw_detections(bgr_img, detections)
                                cv2.putText(bgr_img, "MODE: YOLO DETECTION (Click to Select)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            else:
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
                                    # 【优化】平滑系数从0.7改为0.5，降低滞后
                                    self.smooth_fx = 0.5 * fx_raw + 0.5 * self.smooth_fx
                                    self.smooth_fy = 0.5 * fy_raw + 0.5 * self.smooth_fy
                                    fx, fy = int(self.smooth_fx), int(self.smooth_fy)
                                    dx, dy = fx - cx, fy - cy
                                    dt = now_t - self.last_control_t
                                    dt = min(dt, 0.15)  # 放宽上限
                                    if dt > 0:
                                        with g_state.lock:
                                            g_state.manual_mode = False
                                            if abs(dx) > DEADZONE_PX:
                                                g_state.target_yaw += dx * K_PX_TO_DEG * dt
                                            if abs(dy) > DEADZONE_PX:
                                                g_state.target_pitch += -dy * K_PX_TO_DEG * dt
                                            g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                                            # pitch 不限幅
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
                                    dt = min(dt, 0.15)
                                    if dt > 0:
                                        with g_state.lock:
                                            g_state.manual_mode = False
                                            if abs(dx) > DEADZONE_PX:
                                                g_state.target_yaw += dx * K_PX_TO_DEG * dt
                                            if abs(dy) > DEADZONE_PX:
                                                g_state.target_pitch += -dy * K_PX_TO_DEG * dt
                                            g_state.target_yaw = max(-135.0, min(135.0, g_state.target_yaw))
                                        self.last_control_t = now_t
                                    cv2.rectangle(bgr_img, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                                    cv2.circle(bgr_img, (fx, fy), 4, (0, 0, 255), -1)
                                    cv2.putText(bgr_img, f"FACE POS MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    cv2.putText(bgr_img, "SEARCHING...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        with g_state.lock:
                            osd_text = f"Target: Y={g_state.target_yaw:+5.1f} P={g_state.target_pitch:+5.1f}"
                        cv2.putText(bgr_img, osd_text, (15, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        _, buf = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 48])  # 略提高质量
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

# 启动 30Hz 发送线程
threading.Thread(target=send_loop, daemon=True).start()

# ==================== 后端路由与 API（不变） ====================
async def gimbal_watchdog_ctx(app):
    logger.info("🕵️ 智能混合看门狗就绪")
    async def watchdog_loop():
        try:
            while True:
                await asyncio.sleep(0.05)
                now = time.time()
                with g_state.lock:
                    if g_state.manual_mode and now - g_state.last_manual_time > 0.35:
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

async def handle_yolo_click_select(request):
    """处理 YOLO 点击选择 - 左键单击"""
    try:
        data = await request.json()
        click_x = float(data.get('x', 0))
        click_y = float(data.get('y', 0))
        
        if not video_thread.yolo_detector.active:
            return web.json_response({'status': 'error', 'reason': 'YOLO not available'}, status=503)
        
        with g_state.lock:
            g_state.yolo_click_select = (click_x, click_y)
        
        return web.json_response({'status': 'success'})
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

async def handle_yolo_toggle(request):
    """切换 YOLO 检测开关"""
    try:
        data = await request.json()
        enabled = data.get('enabled', True)
        
        video_thread.yolo_active = enabled
        with g_state.lock:
            g_state.yolo_enabled = enabled
        
        return web.json_response({
            'status': 'success', 
            'yolo_active': video_thread.yolo_active,
            'yolo_available': video_thread.yolo_detector.active
        })
    except Exception as e:
        return web.json_response({'status': 'error', 'reason': str(e)}, status=500)

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
        yolo_en = g_state.yolo_enabled
    return web.json_response({
        'rtsp_connected': video_thread.connected, 
        'system_mode': 'manual' if g_state.manual_mode else 'auto',
        'tracking_mode': g_state.tracking_mode,
        'detector_type': video_thread.face_detector.active_method,
        'tracker_type': video_thread.object_tracker.active_method,
        'target_yaw': round(tgt_y, 2),
        'target_pitch': round(tgt_p, 2),
        'yolo_active': video_thread.yolo_active,
        'yolo_available': video_thread.yolo_detector.active,
        'yolo_enabled': yolo_en
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
    app.router.add_post('/api/yolo/click', handle_yolo_click_select)
    app.router.add_post('/api/yolo/toggle', handle_yolo_toggle)
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
