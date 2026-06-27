#!/usr/bin/env python3
"""实例五（框选区域YOLO版·修复跟踪丢失）：鼠标框选区域→对该区域YOLO检测→点击目标跟踪（改进跟踪匹配）"""
import cv2
import socket
import struct
import time
import os
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"

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

K_PX_TO_DEG = 0.22
DEADZONE_PX = 5
SMOOTH_ALPHA = 0.5
DT_MAX = 0.15
REVERSE_PITCH = True
MANUAL_SPEED_DEG_PER_SEC = 80.0
KEY_TIMEOUT = 0.2
MANUAL_IDLE_TIMEOUT = 1.0

yolo = YOLO('yolov8n.pt')
yolo.conf = 0.5  # 框选检测用的阈值

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

# 跟踪丢失容忍
lost_count = 0
MAX_LOST_FRAMES = 5
track_started = False  # 标记是否刚进入跟踪状态（前几帧特殊处理）
track_frame_count = 0  # 跟踪开始的帧计数

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
    global selecting, selection_rect, waiting_select, region_detections, region_detected, latest_frame, lost_count, track_started, track_frame_count

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if tracking_mode:
            tracking_mode = False; tracking_bbox = None; lost_count = 0; track_started = False; track_frame_count = 0; print("双击取消跟踪")
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
                    track_started = True
                    track_frame_count = 0
                    smooth_fx = (x1+x2)/2; smooth_fy = (y1+y2)/2
                    print(f"✅ 开始跟踪目标: 类别 {cls_id}, 置信度 {conf:.2f}, 初始框 {tracking_bbox}")
                    waiting_select = False; selection_rect = None; region_detections = []; region_detected = False
                    hit = True; break
            if not hit:
                waiting_select = False; selection_rect = None; region_detections = []; region_detected = False
                print("点击空白，取消框选")
            return

        selecting = True
        selection_rect = (x, y, x, y)
        if tracking_mode:
            tracking_mode = False; tracking_bbox = None; lost_count = 0; track_started = False; track_frame_count = 0; print("框选开始，停止跟踪")
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
            tracking_mode = False; tracking_bbox = None; lost_count = 0; track_started = False; track_frame_count = 0; print("右键取消跟踪")
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

def match_target(current_detections, prev_bbox, is_first_frame=False):
    """
    改进匹配：结合 IoU 和中心距离，返回最佳匹配或 None
    is_first_frame: 刚进入跟踪的前几帧，使用更宽松的策略
    """
    tx, ty, tw, th = prev_bbox
    prev_cx = tx + tw/2
    prev_cy = ty + th/2
    diag = (tw*tw + th*th)**0.5
    if is_first_frame:
        max_dist = diag * 3.0  # 第一帧放宽到3倍
        iou_thresh = 0.02
    else:
        max_dist = diag * 2.5
        iou_thresh = 0.05

    best_match = None
    best_score = iou_thresh

    for det in current_detections:
        x1, y1, x2, y2, _, _ = det
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        dist = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
        if dist > max_dist:
            continue
        # 计算 IoU
        ix1 = max(x1, tx); iy1 = max(y1, ty)
        ix2 = min(x2, tx+tw); iy2 = min(y2, ty+th)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area1 = (x2-x1)*(y2-y1); area2 = tw*th
        iou = inter / (area1 + area2 - inter + 1e-5)
        # 综合评分
        dist_score = 1.0 - dist / max_dist if max_dist > 0 else 0
        score = 0.6 * iou + 0.4 * dist_score
        if score > best_score:
            best_score = score
            best_match = det
    return best_match

def main():
    global MANUAL_SPEED_DEG_PER_SEC, latest_frame
    global tracking_bbox, tracking_mode, smooth_fx, smooth_fy
    global selecting, selection_rect, waiting_select, region_detections, region_detected, lost_count, track_started, track_frame_count

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开 RTSP 流"); return

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧"); return
    h, w = frame.shape[:2]
    latest_frame = frame.copy()

    cv2.namedWindow("YOLO Region Track")
    cv2.setMouseCallback("YOLO Region Track", mouse_callback, {'w': w, 'h': h})

    target_yaw = 0.0; target_pitch = 0.0; last_control_t = time.time()
    keys_down = set(); last_key_time = 0.0; manual_mode = False; last_manual_activity = 0.0

    print("=" * 78)
    print("YOLO 框选区域检测跟踪（修复跟踪丢失版）")
    print("左键拖拽框选 → 区域YOLO检测 → 点击目标跟踪 | 双击/右键取消 | WASD/鼠标手动")
    print("Q加速 E减速 | 空格回中/停止跟踪 | ESC退出")
    print("跟踪阶段使用更低置信度阈值(0.25)，提高召回率")
    print("=" * 78)

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
                tracking_mode = False; tracking_bbox = None; lost_count = 0; track_started = False; track_frame_count = 0
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
                    # 跟踪模式：使用更低置信度阈值
                    original_conf = yolo.conf
                    yolo.conf = 0.25  # 临时降低阈值
                    results = yolo(frame, verbose=False, imgsz=320)
                    yolo.conf = original_conf  # 恢复
                    dets = results[0].boxes
                    current_dets = []
                    if dets is not None:
                        for box in dets:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf.item(); cls_id = int(box.cls.item())
                            current_dets.append((x1,y1,x2,y2,conf,cls_id))

                    # 判断是否为跟踪初期（前3帧）
                    is_first = track_started and track_frame_count < 3
                    best_match = match_target(current_dets, tracking_bbox, is_first_frame=is_first)

                    if best_match is not None:
                        lost_count = 0
                        track_started = False  # 一旦匹配成功，退出特殊模式
                        x1,y1,x2,y2,_,_ = best_match
                        tracking_bbox = (x1,y1,x2-x1,y2-y1)
                        fx_raw,fy_raw = (x1+x2)//2,(y1+y2)//2
                        smooth_fx = SMOOTH_ALPHA*fx_raw + (1-SMOOTH_ALPHA)*smooth_fx
                        smooth_fy = SMOOTH_ALPHA*fy_raw + (1-SMOOTH_ALPHA)*smooth_fy
                        fx,fy = int(smooth_fx),int(smooth_fy)
                        dx = fx-cx; dy = fy-cy
                        if abs(dx) > DEADZONE_PX: target_yaw += dx*K_PX_TO_DEG*dt
                        if abs(dy) > DEADZONE_PX: target_pitch -= dy*K_PX_TO_DEG*dt
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                        cv2.circle(frame,(fx,fy),4,(0,0,255),-1)
                        cv2.putText(frame,"TRACKING",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    else:
                        lost_count += 1
                        if lost_count >= MAX_LOST_FRAMES:
                            tracking_mode = False; tracking_bbox = None; lost_count = 0; track_started = False; track_frame_count = 0
                            print(f"跟踪丢失（连续{MAX_LOST_FRAMES}帧无匹配）")
                            cv2.putText(frame,"LOST",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                        else:
                            x,y,bw,bh = tracking_bbox
                            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,255),1)
                            cv2.putText(frame,f"HOLD ({lost_count}/{MAX_LOST_FRAMES})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                    track_frame_count += 1
                else:
                    # 非跟踪模式
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
        cv2.imshow("YOLO Region Track",frame)

    cap.release(); cv2.destroyAllWindows(); sock.close()

if __name__ == '__main__':
    main()
