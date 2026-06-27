import cv2
import numpy as np
from ultralytics import YOLO
import socket
import struct
import time
import signal
import sys
import threading

# ====== 配置参数 ======
A8_IP = '192.168.144.25'
A8_PORT = 37260
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'  # A8 默认 RTSP 地址

MODEL_PATH = 'yolov8n.pt'          # 可换成 yolov8s.pt 等
TARGET_CLASS = 0                   # COCO 中 person 为 0；car 为 2
GAIN_YAW = 0.05                    # 偏航速度增益（度/像素）
GAIN_PITCH = 0.03                  # 俯仰速度增益（度/像素）
MAX_SPEED = 15                     # 最大角速度（度/秒）
DEAD_ZONE = 20                     # 死区像素半径，避免微抖

# 键盘控制参数
KEYBOARD_YAW_SPEED = 10.0          # 键盘控制偏航速度（度/秒）
KEYBOARD_PITCH_SPEED = 8.0         # 键盘控制俯仰速度（度/秒）

# ====== 全局变量 ======
running = True
cap = None
sock = None
keyboard_mode = False              # 键盘控制模式开关
manual_yaw_speed = 0.0             # 手动偏航速度
manual_pitch_speed = 0.0           # 手动俯仰速度

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global running
    print("\n收到退出信号，正在清理...")
    running = False

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def send_gimbal_speed(yaw_speed, pitch_speed, sock_obj):
    """向 A8 发送云台速度指令（协议 0x01 0x07）"""
    if sock_obj is None:
        return False
        
    # 速度限制
    yaw_speed = max(-MAX_SPEED, min(MAX_SPEED, yaw_speed))
    pitch_speed = max(-MAX_SPEED, min(MAX_SPEED, pitch_speed))

    # 构造数据包（SIYI 二进制协议）
    data = bytearray()
    data.append(0x55)              # 包头
    data.append(0x66)              # 包头
    data.append(0x01)              # 命令集：云台控制
    data.append(0x07)              # 命令 ID：设置云台速度
    data.extend(struct.pack('<f', yaw_speed))     # 偏航速度 float32
    data.extend(struct.pack('<f', pitch_speed))   # 俯仰速度 float32
    # 校验和（从第 3 字节到末尾异或）
    checksum = 0
    for b in data[2:]:
        checksum ^= b
    data.append(checksum)
    
    try:
        sock_obj.sendto(data, (A8_IP, A8_PORT))
        return True
    except Exception as e:
        print(f"UDP 发送失败：{e}")
        return False

def cleanup():
    """清理资源"""
    global cap, sock
    print("正在清理资源...")
    
    # 停止云台
    if sock:
        send_gimbal_speed(0, 0, sock)
        time.sleep(0.2)  # 给云台一点时间响应
        sock.close()
        sock = None
    
    # 释放摄像头
    if cap:
        cap.release()
        cap = None
    
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
    print("清理完成")

def print_keyboard_help():
    """打印键盘控制帮助"""
    print("\n" + "="*50)
    print("🎮 键盘控制模式")
    print("="*50)
    print("  方向键 ↑/↓  : 云台俯仰 (上/下)")
    print("  方向键 ←/→  : 云台偏航 (左/右)")
    print("  空格键       : 云台回中/停止")
    print("  Tab 键       : 切换 自动跟踪 ↔ 手动控制")
    print("  q 键         : 退出程序")
    print("="*50)
    print("按 Tab 切换模式，当前：自动跟踪")
    print("-"*50)

# ====== 主程序 ======
try:
    # ====== UDP 通信初始化 ======
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # ====== 加载模型 ======
    print(f"正在加载 YOLO 模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载完成")
    
    # ====== 初始化视频捕获 ======
    print(f"正在连接 RTSP 流：{RTSP_URL}")
    
    # 使用 GStreamer 后端，添加丢帧和错误容忍参数
    gst_pipeline = (
        f"rtspsrc location={RTSP_URL} latency=0 drop-on-latency=true do-retransmission=false "
        f"! queue leaky=2 max-size-buffers=2 "
        f"! rtph265depay ! h265parse ! nvvidconv ! videoconvert "
        f"! appsink sync=false emit-signals=false"
    )
    
    # 先尝试标准方式
    cap = cv2.VideoCapture(RTSP_URL)
    
    # 设置 RTSP 超时参数（关键！）
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 连接超时 5 秒
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 500)   # 读取超时 0.5 秒
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # 只缓冲 1 帧，减少延迟
    
    # 验证是否成功打开
    if not cap.isOpened():
        print("⚠️  标准方式失败，尝试 GStreamer 后端...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ 无法打开 RTSP 流！请检查：")
        print("   1. A8 设备是否在线 (ping 192.168.144.25)")
        print("   2. RTSP 服务是否运行")
        print("   3. 网络连接是否正常")
        print("   4. 防火墙是否阻止")
        cleanup()
        sys.exit(1)
    
    print("✅ 视频流连接成功")
    print_keyboard_help()
    
    # ====== 主循环 ======
    frame_count = 0
    start_time = time.time()
    error_frame_count = 0
    max_consecutive_errors = 10  # 最多连续错误次数
    
    while running:
        # 使用超时读取（避免阻塞）
        ret, frame = cap.read()
        
        if not ret:
            error_frame_count += 1
            if error_frame_count % 30 == 0:
                print(f"⚠️  视频流错误 (连续 {error_frame_count} 帧)，尝试恢复...")
            
            # 如果连续错误太多，尝试重连
            if error_frame_count >= max_consecutive_errors:
                print("⚠️  视频流断开，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 500)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                error_frame_count = 0
            continue
        
        # 重置错误计数
        error_frame_count = 0
        frame_count += 1
        
        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2
        
        # 显示模式指示
        mode_text = "手动控制" if keyboard_mode else "自动跟踪"
        mode_color = (0, 0, 255) if keyboard_mode else (0, 255, 0)
        cv2.putText(frame, f"Mode: {mode_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        # ====== 键盘控制模式 ======
        if keyboard_mode:
            # 显示手动速度
            if abs(manual_yaw_speed) > 0.1 or abs(manual_pitch_speed) > 0.1:
                speed_text = f"Yaw: {manual_yaw_speed:.1f} | Pitch: {manual_pitch_speed:.1f}"
                cv2.putText(frame, speed_text, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            send_gimbal_speed(manual_yaw_speed, manual_pitch_speed, sock)
        
        # ====== 自动跟踪模式 ======
        else:
            # YOLOv8 推理（只检测指定类别）
            results = model(frame, classes=[TARGET_CLASS], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            
            target_found = False
            if len(boxes) > 0:
                # 选择面积最大的目标（简单策略）
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = np.argmax(areas)
                x1, y1, x2, y2 = boxes[best_idx]
                cx_target = int((x1 + x2) / 2)
                cy_target = int((y1 + y2) / 2)
                
                # 绘制目标框和中心点
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.circle(frame, (cx_target, cy_target), 5, (0,0,255), -1)
                
                # 显示目标信息
                cv2.putText(frame, "Target Locked", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 计算偏移量（像素）
                dx = cx_target - cx_frame
                dy = cy_target - cy_frame
                
                # 死区判断
                if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                    # 速度 = 增益 × 偏移（负号：云台方向与偏移相反）
                    yaw_speed = -dx * GAIN_YAW
                    pitch_speed = -dy * GAIN_PITCH
                    send_gimbal_speed(yaw_speed, pitch_speed, sock)
                    target_found = True
            
            if not target_found and not keyboard_mode:
                # 没有目标时停止云台
                send_gimbal_speed(0, 0, sock)
        
        # 显示画面（带超时，避免窗口卡死）
        cv2.imshow('A8 YOLO Track', frame)
        
        # 处理键盘输入
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            print("收到退出指令")
            break
        
        elif key == 9:  # Tab 键 - 切换模式
            keyboard_mode = not keyboard_mode
            if keyboard_mode:
                manual_yaw_speed = 0.0
                manual_pitch_speed = 0.0
                print("🎮 切换到手动控制模式")
            else:
                print("🤖 切换到自动跟踪模式")
            # 清空键盘缓冲区
            cv2.waitKey(1)
        
        elif keyboard_mode:
            # 方向键控制（OpenCV 方向键码）
            if key == 81 or key == 2:      # 左箭头 / 'a'
                manual_yaw_speed = -KEYBOARD_YAW_SPEED
                manual_pitch_speed = 0.0
                print(f"← 偏航左：{manual_yaw_speed}")
            elif key == 83 or key == 3:    # 右箭头 / 'd'
                manual_yaw_speed = KEYBOARD_YAW_SPEED
                manual_pitch_speed = 0.0
                print(f"→ 偏航右：{manual_yaw_speed}")
            elif key == 82 or key == 1:    # 上箭头 / 'w'
                manual_pitch_speed = -KEYBOARD_PITCH_SPEED
                manual_yaw_speed = 0.0
                print(f"↑ 俯仰上：{manual_pitch_speed}")
            elif key == 84 or key == 4:    # 下箭头 / 's'
                manual_pitch_speed = KEYBOARD_PITCH_SPEED
                manual_yaw_speed = 0.0
                print(f"↓ 俯仰下：{manual_pitch_speed}")
            elif key == 32:  # 空格键 - 回中/停止
                manual_yaw_speed = 0.0
                manual_pitch_speed = 0.0
                print("⏹ 云台停止")
            else:
                # 其他键，保持当前速度（允许惯性）
                pass
        
        # 性能监控（每 60 帧打印一次）
        if frame_count % 60 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            mode_status = "MANUAL" if keyboard_mode else "AUTO"
            print(f"[{mode_status}] FPS: {fps:.1f} | 运行时间：{elapsed:.0f}s")
    
    print("退出主循环")
    
except Exception as e:
    print(f"❌ 程序异常：{e}")
    import traceback
    traceback.print_exc()
finally:
    # 确保 cleanup 一定会执行
    cleanup()
    print("程序已退出")