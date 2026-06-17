#!/usr/bin/env python3
"""
SIYI A8 Mini 网页版后端服务 - 超低延迟优化版
优化了 FFmpeg 缓存队列、引入了多线程/队列隔离机制，实现画面零延迟
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
import queue
from aiohttp import web, WSMsgType

# 检查依赖
try:
    import av
except ImportError:
    print("❌ 缺少依赖: av (PyAV)")
    print("   安装: pip install av")
    sys.exit(1)

# 配置
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('siyi-backend')

# ==================== 视频采集与处理线程 ====================
class VideoCaptureThread(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.last_frame = None
        self.last_frame_time = time.time()
        self.running = True
        self.connected = False
        self.lock = threading.Lock()
        
        # 引入高效帧队列，容量设为 1，只保留最新的一帧，旧帧直接丢弃
        self.frame_queue = queue.Queue(maxsize=1)
        
        # 启动独立的图像处理线程，将“收包”与“图像处理”并行化
        self.processor_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processor_thread.start()

    def run(self):
        while self.running:
            container = None
            try:
                logger.info(f"🚀 正在连接 RTSP (低延迟模式): {self.rtsp_url}")
                
                # 核心低延迟 FFmpeg 配置参数
                container = av.open(self.rtsp_url, options={
                    'rtsp_transport': 'tcp',
                    'timeout': '3000000',
                    'stimeout': '3000000',
                    'probesize': '32768',          # 进一步紧缩，只要能解析出 H.264 头即可
                    'analyzeduration': '100000',    # 缩短至 0.1 秒分析时间
                    'fflags': 'nobuffer',          # 彻底禁用 FFmpeg 内部缓存
                    'flags': 'low_delay',          # 开启低延迟标记
                    'strict': 'experimental'
                })
                
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                
                self.connected = True
                logger.info("✅ RTSP 视频流已连接 (已开启零缓存机制)")
                self.last_frame_time = time.time()
                
                for packet in container.demux(stream):
                    if not self.running: 
                        break
                    if packet.size == 0: 
                        continue
                        
                    for frame in packet.decode():
                        if not self.running: 
                            break
                        if not isinstance(frame, av.video.frame.VideoFrame): 
                            continue
                        
                        # 快速丢弃处理线来不及处理的旧帧，防止画面堆积造成回放式卡顿
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.frame_queue.put(frame)
                            
            except Exception as e:
                logger.error(f"❌ 视频采集错误: {e}")
                self.connected = False
            finally:
                if container:
                    try: container.close()
                    except: pass
                if self.running:
                    time.sleep(1)

    def _process_frames(self):
        """ 独立的耗时图像压缩线程：确保不阻塞底层网卡接收 RTSP 包 """
        while self.running:
            try:
                # 阻塞式等待新帧，超时设为 1 秒防止死锁
                frame = self.frame_queue.get(timeout=1.0)
                
                # 图像转换与 JPEG 压缩（极消耗 CPU，独立出来后不会引发延迟）
                img = frame.to_image().resize((640, 480))
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=45) 
                
                with self.lock:
                    self.last_frame = buf.getvalue()
                    self.last_frame_time = time.time()
                    
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ 图像处理器异常: {e}")

    def get_latest_frame(self):
        if time.time() - self.last_frame_time > 4.0:
            if self.connected:
                self.connected = False
        with self.lock:
            return self.last_frame

    def stop(self):
        self.running = False

# 启动采集线程
video_thread = VideoCaptureThread(RTSP_URL)
video_thread.start()

# ==================== 云台控制逻辑 ====================
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

def send_gimbal_command(yaw, pitch, ip=CAMERA_IP, port=UDP_PORT):
    yaw, pitch = max(-100, min(100, int(yaw))), max(-100, min(100, int(pitch)))
    pkt = struct.pack('<H B H H B b b', 0x6655, 0x00, 0x0002, 0x0001, 0x07, yaw, pitch)
    pkt += struct.pack('<H', crc16(pkt))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(pkt, (ip, port))
        return True
    except: 
        return False

# ==================== HTTP API 处理器 ====================
async def handle_gimbal_control(request):
    try:
        data = await request.json()
        send_gimbal_command(data.get('yaw', 0), data.get('pitch', 0), data.get('ip', CAMERA_IP), data.get('port', UDP_PORT))
        return web.json_response({'status': 'success'})
    except: 
        return web.json_response({'status': 'error'}, status=500)

async def handle_status(request):
    return web.json_response({'camera_ip': CAMERA_IP, 'rtsp_connected': video_thread.connected, 'status': 'running'})

async def handle_static(request):
    path = request.match_info.get('path', 'index.html') or 'index.html'
    file_path = os.path.join(WORKSPACE_DIR, path)
    if os.path.exists(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404)

# ==================== WebSocket 视频流 ====================
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("新的 WebSocket 客户端已连接")
    
    last_sent_frame_time = 0
    try:
        while not ws.closed:
            current_frame = video_thread.get_latest_frame()
            current_frame_time = video_thread.last_frame_time
            
            if current_frame and current_frame_time > last_sent_frame_time:
                try:
                    await ws.send_bytes(current_frame)
                    last_sent_frame_time = current_frame_time
                except:
                    break
            
            # 缩短 WebSocket 轮询休眠，加快推送响应速率
            await asyncio.sleep(0.005) 
    finally:
        logger.info("WebSocket 客户端已断开")
    return ws

# ==================== 主程序 ====================
async def init_app():
    app = web.Application()
    app.router.add_get('/api/ws', websocket_handler)
    app.router.add_post('/api/gimbal/control', handle_gimbal_control)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/{path:.*}', handle_static)
    return app

if __name__ == '__main__':
    try:
        app = asyncio.get_event_loop().run_until_complete(init_app())
        web.run_app(app, host='0.0.0.0', port=SERVER_PORT)
    except KeyboardInterrupt:
        pass
    finally:
        video_thread.stop()
        logger.info("👋 服务已停止")
