#!/usr/bin/env python3
"""实例五（框选区域YOLO版·DKF+Kalman增强）：鼠标框选区域→YOLO检测→点击跟踪，集成卡尔曼滤波与预测"""
import cv2
import socket
import struct
import time
import os
import math
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"

# ========== CRC 和命令帧 ==========
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

# ========== 控制参数 ==========
K_PX_TO_DEG = 0.28
DEADZONE_PX = 5
SMOOTH_ALPHA = 0.5
DT_MAX = 0.15
REVERSE_PITCH = True
MANUAL_SPEED_DEG_PER_SEC = 80.0
KEY_TIMEOUT = 0.2
MANUAL_IDLE_TIMEOUT = 1.0

# ========== DKF 滤波器（离散卡尔曼滤波） ==========
class DKFFilter:
    def __init__(self, q=1.0, r=10.0, p_init=100.0):
        self.q = q
        self.r = r
        self.p_init = p_init
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.initialized = False

    def init_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        self.x = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0]
        self.P = [[self.p_init]*8 for _ in range(8)]
        self.initialized = True
        return True

    def predict(self, dt):
        if not self.initialized:
            return
        F = [[1,0,0,0,dt,0,0,0],
             [0,1,0,0,0,dt,0,0],
             [0,0,1,0,0,0,dt,0],
             [0,0,0,1,0,0,0,dt],
             [0,0,0,0,1,0,0,0],
             [0,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,1]]
        new_x = [0.0]*8
        for i in range(8):
            s = 0.0
            for j in range(8):
                s += F[i][j] * self.x[j]
            new_x[i] = s
        self.x = new_x
        dt2 = dt*dt
        for i in range(8):
            self.P[i][i] += self.q * (dt2 if i<4 else dt)

    def update(self, measurement, dt):
        if not self.initialized:
            return
        z = list(measurement)  # [cx, cy, w, h]
        for i in range(4):
            innov = z[i] - self.x[i]
            p = self.P[i][i] + self.r
            k = self.P[i][i] / p
            self.x[i] += k * innov
            self.x[i+4] += k * innov / max(0.01, dt)
            self.P[i][i] = (1 - k) * self.P[i][i]

    def get_bbox(self):
        if not self.initialized:
            return None
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return (x1, y1, x2, y2)

    def get_center(self):
        if not self.initialized:
            return None
        return (self.x[0], self.x[1])

    def get_velocity(self):
        if not self.initialized:
            return (0.0, 0.0)
        return (self.x[4], self.x[5])

# ========== Kalman 预测器 ==========
class KalmanTargetPredictor:
    def __init__(self, process_noise=50.0, measurement_noise=25.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.initialized = False

    def init(self, cx, cy):
        self.x = [cx, cy, 0.0, 0.0]
        self.P = [[100.0,0,0,0],
                  [0,100.0,0,0],
                  [0,0,10000.0,0],
                  [0,0,0,10000.0]]
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        F = [[1,0,dt,0],
             [0,1,0,dt],
             [0,0,1,0],
             [0,0,0,1]]
        new_x = [0.0]*4
        for i in range(4):
            s = 0.0
            for j in range(4):
                s += F[i][j] * self.x[j]
            new_x[i] = s
        self.x = new_x
        for i in range(4):
            self.P[i][i] += self.Q * (dt*dt if i<2 else dt)

    def update(self, cx, cy, dt):
        if not self.initialized:
            self.init(cx, cy)
            return
        innov_x = cx - self.x[0]
        innov_y = cy - self.x[1]
        p00 = self.P[0][0] + self.R
        p11 = self.P[1][1] + self.R
        kx = self.P[0][0] / p00
        ky = self.P[1][1] / p11
        self.x[0] += kx * innov_x
        self.x[1] += ky * innov_y
        self.x[2] += kx * innov_x / max(0.01, dt)
        self.x[3] += ky * innov_y / max(0.01, dt)
        self.P[0][0] = (1-kx) * self.P[0][0]
        self.P[1][1] = (1-ky) * self.P[1][1]

    def predict_future(self, ahead_sec):
        if not self.initialized:
            return None
        cx = self.x[0] + self.x[2] * ahead_sec
        cy = self.x[1] + self.x[3] * ahead_sec
        return (cx, cy)

# ========== YOLO 初始化 ==========
yolo = YOLO('yolov8n.pt')
yolo.conf = 0.5

# ========== 全局状态 ==========
mouse_data = {'active': False, 'yaw_speed': 0.0, 'pitch_speed': 0.0, 'last_move_time': 0.0}
tracking_bbox = None
tracking_mode = False
smooth_fx, smooth_fy = 0.0, 0.0

selecting = False
selection_rect = None
waiting_select = False
region_detections = []
region_detected = False

latest_frame = None

# DKF 和 Kalman 预测器实例
dkf = DKFFilter(q=0.5, r=5.0, p_init=100.0)
kalman = KalmanTargetPredictor(process_noise=30.0, measurement_noise=15.0)

# 丢失容忍
lost_count = 0
MAX_LOST_FRAMES = 8
PREDICT_AHEAD_MS = 80

def run_yolo_on_region(frame, rect):
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    results = yolo(roi, verbose=False, imgsz=320)
    dets = results[0].boxes
    detections = []
    if dets is not None:
        for box in dets:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            detections.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, conf, cls_id))
    return detections

def mouse_callback(event, x, y, flags, param):
    global mouse_data, tracking_bbox, tracking_mode, smooth_fx, smooth_fy
    global selecting, selection_rect, waiting_select, region_detections, region_detected, latest_frame, lost_count

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if tracking_mode:
            tracking_mode = False; tracking_bbox = None; lost_count = 0; dkf.reset(); kalman.reset(); print("双击取消跟踪")
        if waiting_select or selecting:
            selecting = False; waiting_select = False; selection_rect = None
            region_detections = []; region_detected = False; print("双击清除框选")
        if mouse_data['active']:
            mouse_data['active'] = False; mouse_data['yaw_speed'] = 0.0; mouse_data['pitch_speed'] = 0.0
            print("双击退出手动控制")
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if waiting_select:
            hit = False
            for det in region_detections:
                x1, y1, x2, y2, conf, cls_id = det
                if x1 <= x <= x2 and y1 <= y <= y2:
                    tracking_bbox = (x1, y1, x2-x1, y2-y1)
                    tracking_mode = True
                    lost_count = 0
                    smooth_fx = (x1+x2)/2; smooth_fy = (y1+y2)/2
                    dkf.init_from_bbox(tracking_bbox)
                    kalman.init(smooth_fx, smooth_fy)
                    print(f"✅ 开始跟踪目标: 类别 {cls_id}, 置信度 {conf:.2f}")
                    waiting_select = False; selection_rect = None; region_detections = []; region_detected = False
                    hit = True; break
            if not hit:
                waiting_select = False; selection_rect = None; region_detections = []; region_detected = False
                print("点击空白，取消框选")
            return

        selecting = True
        selection_rect = (x, y, x, y)
        if tracking_mode:
            tracking_mode = False; tracking_bbox = None; lost_count = 0; dkf.reset(); kalman.reset(); print("框选开始，停止跟踪")
        region_detections = []; region_detected = False
        return

    if event == cv2.EVENT_MOUSEMOVE and selecting:
        x1, y1, _, _ = selection_rect
        selection_rect = (x1, y1, x, y)
        return

    if event == cv2.EVENT_LBUTTONUP and selecting:
        selecting = False
        x1, y1, x2, y2 = selection_rect
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(param['w'], x2); y2 = min(param['h'], y2)
        selection_rect = (x1, y1, x2, y2)
        waiting_select = True
        if latest_frame is not None and x2 > x1 and y2 > y1:
            region_detections = run_yolo_on_region(latest_frame, selection_rect)
            region_detected = True
            print(f"框选区域: ({x1},{y1})-({x2},{y2})，检测到 {len(region_detections)} 个目标")
        else:
            region_detections = []; region_detected = False
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        if waiting_select or selecting:
            selecting = False; waiting_select = False; selection_rect = None
            region_detections = []; region_detected = False; print("右键取消框选")
        elif tracking_mode:
            tracking_mode = False; tracking_bbox = None; lost_count = 0; dkf.reset(); kalman.reset(); print("右键取消跟踪")
        return

    if not selecting and not waiting_select:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_data['active'] = True; mouse_data['last_move_time'] = time.time()
            update_mouse_speed(x, y, param['w'], param['h'])
        elif event == cv2.EVENT_MOUSEMOVE and mouse_data['active']:
            mouse_data['last_move_time'] = time.time()
            update_mouse_speed(x, y, param['w'], param['h'])
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_data['active'] = False; mouse_data['yaw_speed'] = 0.0; mouse_data['pitch_speed'] = 0.0

def update_mouse_speed(x, y, w, h):
    yaw_speed = (x/w - 0.5)*300; pitch_speed = (0.5 - y/h)*300
    mouse_data['yaw_speed'] = max(-100, min(100, yaw_speed))
    mouse_data['pitch_speed'] = max(-100, min(100, pitch_speed))

def match_target(current_detections, prev_bbox):
    tx, ty, tw, th = prev_bbox
    prev_cx = tx + tw/2
    prev_cy = ty + th/2
    diag = (tw*tw + th*th)**0.5
    max_dist = diag * 2.0
    best_match = None
    best_score = 0.05
    for det in current_detections:
        x1, y1, x2, y2, _, _ = det
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        dist = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
        if dist > max_dist:
            continue
        ix1 = max(x1, tx); iy1 = max(y1, ty)
        ix2 = min(x2, tx+tw); iy2 = min(y2, ty+th)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area1 = (x2-x1)*(y2-y1); area2 = tw*th
        iou = inter / (area1 + area2 - inter + 1e-5)
        dist_score = 1.0 - dist / max_dist if max_dist > 0 else 0
        score = 0.6 * iou + 0.4 * dist_score
        if score > best_score:
            best_score = score
            best_match = det
    return best_match

def main():
    global MANUAL_SPEED_DEG_PER_SEC, latest_frame
    global tracking_bbox, tracking_mode, smooth_fx, smooth_fy
    global selecting, selection_rect, waiting_select, region_detections, region_detected, lost_count

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开 RTSP 流"); return

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧"); return
    h, w = frame.shape[:2]
    latest_frame = frame.copy()

    cv2.namedWindow("YOLO DKF Track")
    cv2.setMouseCallback("YOLO DKF Track", mouse_callback, {'w': w, 'h': h})

    target_yaw = 0.0; target_pitch = 0.0; last_control_t = time.time()
    keys_down = set(); last_key_time = 0.0; manual_mode = False; last_manual_activity = 0.0

    print("=" * 86)
    print("YOLO 框选区域检测跟踪（DKF+Kalman增强版）")
    print("左键拖拽框选 → 区域YOLO检测 → 点击目标跟踪 | 双击/右键取消 | WASD/鼠标手动")
    print("Q加速 E减速 | 空格回中/停止跟踪 | ESC退出")
    print("手动操作后1秒无操作自动恢复跟踪")
    print("集成卡尔曼滤波与预测，丢失容忍8帧")
    print("=" * 86)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue
        latest_frame = frame.copy()

        cx, cy = w//2, h//2
        now_t = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == 27: break
            elif key == ord(' '):
                target_yaw = 0.0; target_pitch = 0.0
                tracking_mode = False; tracking_bbox = None; lost_count = 0
                dkf.reset(); kalman.reset()
                waiting_select = False; selection_rect = None; region_detections = []; region_detected = False
                print("回中并重置")
            elif key == ord('q'):
                MANUAL_SPEED_DEG_PER_SEC = min(150, MANUAL_SPEED_DEG_PER_SEC+10)
                print(f"加速: {MANUAL_SPEED_DEG_PER_SEC:.0f}°/s")
            elif key == ord('e'):
                MANUAL_SPEED_DEG_PER_SEC = max(10, MANUAL_SPEED_DEG_PER_SEC-10)
                print(f"减速: {MANUAL_SPEED_DEG_PER_SEC:.0f}°/s")
            elif key in (ord('w'),ord('s'),ord('a'),ord('d')):
                keys_down.add(key); last_key_time = now_t; manual_mode = True; last_manual_activity = now_t

        if now_t - last_key_time > KEY_TIMEOUT: keys_down.clear()
        if mouse_data['active']: manual_mode = True; last_manual_activity = now_t
        if manual_mode and (now_t - last_manual_activity > MANUAL_IDLE_TIMEOUT):
            manual_mode = False; print("恢复自动模式")

        dt = now_t - last_control_t; dt = min(dt, DT_MAX)
        if dt > 0:
            if manual_mode:
                dx, dy = 0, 0
                if ord('w') in keys_down: dy = 1
                if ord('s') in keys_down: dy = -1
                if ord('a') in keys_down: dx = -1
                if ord('d') in keys_down: dx = 1
                if mouse_data['active']:
                    dx += mouse_data['yaw_speed']/100.0
                    dy += mouse_data['pitch_speed']/100.0
                target_yaw += dx * MANUAL_SPEED_DEG_PER_SEC * dt
                target_pitch += dy * MANUAL_SPEED_DEG_PER_SEC * dt
            else:
                if tracking_mode and tracking_bbox is not None:
                    # DKF 预测
                    if dkf.initialized:
                        dkf.predict(dt)
                        kalman.predict(dt)

                    # YOLO 检测
                    results = yolo(frame, verbose=False, imgsz=320)
                    dets = results[0].boxes
                    current_dets = []
                    if dets is not None:
                        for box in dets:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf.item(); cls_id = int(box.cls.item())
                            current_dets.append((x1,y1,x2,y2,conf,cls_id))

                    best_match = match_target(current_dets, tracking_bbox)

                    if best_match is not None:
                        lost_count = 0
                        x1,y1,x2,y2,_,_ = best_match
                        if dkf.initialized:
                            cx_m = (x1+x2)/2.0
                            cy_m = (y1+y2)/2.0
                            w_m = x2-x1
                            h_m = y2-y1
                            dkf.update((cx_m, cy_m, w_m, h_m), dt)
                            kalman.update(cx_m, cy_m, dt)
                            smoothed = dkf.get_bbox()
                            if smoothed:
                                x1, y1, x2, y2 = smoothed
                                x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
                        tracking_bbox = (x1, y1, x2-x1, y2-y1)
                        fx_raw, fy_raw = (x1+x2)//2, (y1+y2)//2
                        smooth_fx = SMOOTH_ALPHA*fx_raw + (1-SMOOTH_ALPHA)*smooth_fx
                        smooth_fy = SMOOTH_ALPHA*fy_raw + (1-SMOOTH_ALPHA)*smooth_fy
                        fx, fy = int(smooth_fx), int(smooth_fy)
                        predicted = kalman.predict_future(PREDICT_AHEAD_MS / 1000.0)
                        if predicted:
                            px, py = predicted
                        else:
                            px, py = fx, fy
                        dx = px - cx; dy = py - cy
                        if abs(dx) > DEADZONE_PX: target_yaw += dx*K_PX_TO_DEG*dt
                        if abs(dy) > DEADZONE_PX: target_pitch -= dy*K_PX_TO_DEG*dt
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                        cv2.circle(frame,(int(px),int(py)),4,(0,0,255),-1)
                        cv2.putText(frame,"TRACKING+DKF",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    else:
                        lost_count += 1
                        if lost_count >= MAX_LOST_FRAMES:
                            tracking_mode = False; tracking_bbox = None; lost_count = 0
                            dkf.reset(); kalman.reset()
                            print(f"跟踪丢失（连续{MAX_LOST_FRAMES}帧无匹配）")
                            cv2.putText(frame,"LOST",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                        else:
                            if dkf.initialized:
                                predicted_bbox = dkf.get_bbox()
                                if predicted_bbox:
                                    x1, y1, x2, y2 = predicted_bbox
                                    x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
                                    tracking_bbox = (x1, y1, x2-x1, y2-y1)
                                    fx_raw, fy_raw = (x1+x2)//2, (y1+y2)//2
                                    smooth_fx = SMOOTH_ALPHA*fx_raw + (1-SMOOTH_ALPHA)*smooth_fx
                                    smooth_fy = SMOOTH_ALPHA*fy_raw + (1-SMOOTH_ALPHA)*smooth_fy
                                    fx, fy = int(smooth_fx), int(smooth_fy)
                                    predicted = kalman.predict_future(PREDICT_AHEAD_MS / 1000.0)
                                    if predicted:
                                        px, py = predicted
                                    else:
                                        px, py = fx, fy
                                    dx = px - cx; dy = py - cy
                                    if abs(dx) > DEADZONE_PX: target_yaw += dx*K_PX_TO_DEG*dt
                                    if abs(dy) > DEADZONE_PX: target_pitch -= dy*K_PX_TO_DEG*dt
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
                                    cv2.circle(frame,(int(px),int(py)),4,(0,0,255),-1)
                                    cv2.putText(frame,f"PREDICT ({lost_count}/{MAX_LOST_FRAMES})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                                else:
                                    x,y,bw,bh = tracking_bbox
                                    cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,255),1)
                                    cv2.putText(frame,f"HOLD ({lost_count}/{MAX_LOST_FRAMES})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                            else:
                                x,y,bw,bh = tracking_bbox
                                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,255),1)
                                cv2.putText(frame,f"HOLD ({lost_count}/{MAX_LOST_FRAMES})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                else:
                    if waiting_select and selection_rect is not None and region_detected:
                        x1,y1,x2,y2 = selection_rect
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)
                        for det in region_detections:
                            dx1,dy1,dx2,dy2,conf,cls_id = det
                            cv2.rectangle(frame,(dx1,dy1),(dx2,dy2),(0,255,0),2)
                            name = yolo.model.names.get(cls_id,str(cls_id))
                            cv2.putText(frame,f"{name} {conf:.2f}",(dx1,dy1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                        cv2.putText(frame,"SELECT - Click a green box to track",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
                    else:
                        cv2.putText(frame,"IDLE - Drag to select region",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(128,128,128),2)

            if selecting and selection_rect is not None:
                x1,y1,x2,y2 = selection_rect
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            target_yaw = max(-135.0,min(135.0,target_yaw))
            target_pitch = max(-90.0,min(90.0,target_pitch))
            last_control_t = now_t

        cmd = build_position_cmd(target_yaw,target_pitch,reverse_pitch=REVERSE_PITCH)
        sock.sendto(cmd,(CAMERA_IP,UDP_PORT))

        mode_str = "MANUAL" if manual_mode else ("TRACK" if tracking_mode else "IDLE")
        info = f"Mode:{mode_str} Yaw:{target_yaw:+6.1f} Pitch:{target_pitch:+6.1f}"
        cv2.putText(frame,info,(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        cv2.imshow("YOLO DKF Track",frame)

    cap.release(); cv2.destroyAllWindows(); sock.close()

if __name__ == '__main__':
    main()
