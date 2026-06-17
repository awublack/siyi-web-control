#!/usr/bin/env python3
"""
SIYI A8 Mini 网页版后端服务 - 增强稳定性版
提供 WebSocket 视频流和 HTTP API 云台控制
支持 RTSP 自动重连和异常恢复
"""

import asyncio
import json
import socket
import struct
import logging
import os
import sys
import io
import time

# 检查依赖
try:
    import av
except ImportError:
    print("❌ 缺少依赖: av (PyAV)")
    print("   安装: pip install av")
    sys.exit(1)

try:
    from aiohttp import web, WSMsgType
except ImportError:
    print("❌ 缺少依赖: aiohttp")
    print("   安装: pip install aiohttp")
    sys.exit(1)

# 配置
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('siyi-backend')

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

def crc16(data):
    crc = 0
    for b in data:
        tmp = ((crc >> 8) ^ b) & 0xFF
        crc = ((crc << 8) ^ CRC16_TABLE[tmp]) & 0xFFFF
    return crc

def send_gimbal_command(yaw, pitch, ip=CAMERA_IP, port=UDP_PORT):
    """发送云台控制命令"""
    yaw = max(-100, min(100, int(yaw)))
    pitch = max(-100, min(100, int(pitch)))
    
    pkt = struct.pack('<H B H H B b b', 0x6655, 0x00, 0x0002, 0x0001, 0x07, yaw, pitch)
    pkt += struct.pack('<H', crc16(pkt))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(pkt, (ip, port))
        sock.close()
        return True
    except Exception as e:
        logger.error(f"发送命令失败：{e}")
        return False

def send_gimbal_angle_command(yaw_angle, pitch_angle, ip=CAMERA_IP, port=UDP_PORT):
    """发送角度控制命令"""
    yaw_angle = max(-36000, min(36000, int(yaw_angle)))
    pitch_angle = max(-36000, min(36000, int(pitch_angle)))
    
    pkt = struct.pack('<H B H H B h h', 0x6655, 0x00, 0x0002, 0x0001, 0x01, yaw_angle, pitch_angle)
    pkt += struct.pack('<H', crc16(pkt))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(pkt, (ip, port))
        sock.close()
        return True
    except Exception as e:
        logger.error(f"发送角度命令失败：{e}")
        return False

# ==================== CORS中间件 ====================
@web.middleware
async def cors_middleware(request, handler):
    if request.method == 'OPTIONS':
        response = web.Response(status=200)
    else:
        response = await handler(request)
    
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    return response

# ==================== HTTP API 处理器 ====================
async def handle_gimbal_control(request):
    try:
        data = await request.json()
        yaw = data.get('yaw', 0)
        pitch = data.get('pitch', 0)
        ip = data.get('ip', CAMERA_IP)
        port = data.get('port', UDP_PORT)
        success = send_gimbal_command(yaw, pitch, ip, port)
        if success:
            return web.json_response({'status': 'success'})
        return web.json_response({'status': 'error'}, status=500)
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)

async def handle_gimbal_angle(request):
    try:
        data = await request.json()
        yaw = data.get('yaw', 0)
        pitch = data.get('pitch', 0)
        ip = data.get('ip', CAMERA_IP)
        success = send_gimbal_angle_command(yaw, pitch, ip)
        if success:
            return web.json_response({'status': 'success'})
        return web.json_response({'status': 'error'}, status=500)
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)

async def handle_status(request):
    return web.json_response({
        'camera_ip': CAMERA_IP,
        'rtsp_url': RTSP_URL,
        'status': 'running'
    })

async def handle_static(request):
    try:
        path = request.match_info.get('path', 'index.html')
        if not path: path = 'index.html'
        file_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
        if not file_path.startswith(os.path.abspath(WORKSPACE_DIR)) or not os.path.exists(file_path):
            return web.Response(status=404)
        
        ext = os.path.splitext(path)[1].lower()
        content_type = 'text/html' if ext in ['.html', '.htm'] else 'application/octet-stream'
        if ext == '.js': content_type = 'application/javascript'
        if ext == '.css': content_type = 'text/css'
        
        with open(file_path, 'rb') as f:
            return web.Response(body=f.read(), content_type=content_type)
    except:
        return web.Response(status=500)

# ==================== WebSocket 视频流 ====================
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket 连接已建立")
    
    async def push_video_frames():
        while not ws.closed:
            container = None
            try:
                logger.info(f"正在打开 RTSP 流: {RTSP_URL}")
                # 优化 RTSP 打开参数
                container = av.open(RTSP_URL, options={
                    'rtsp_transport': 'tcp',
                    'stimeout': '5000000', # 5秒超时
                    'buffer_size': '1024000',
                    'max_delay': '500000'
                })
                stream = container.streams.video[0]
                logger.info(f"视频流已打开: {stream.width}x{stream.height}")
                
                for packet in container.demux(stream):
                    if ws.closed: break
                    if packet.size == 0: continue
                    
                    for frame in packet.decode():
                        if not isinstance(frame, av.video.frame.VideoFrame):
                            continue
                        
                        img = frame.to_image()
                        img_resized = img.resize((640, 480))
                        buf = io.BytesIO()
                        img_resized.save(buf, format='JPEG', quality=60)
                        jpeg_data = buf.getvalue()
                        
                        try:
                            await ws.send_bytes(jpeg_data)
                        except:
                            return # WebSocket 已关闭
                        
                        await asyncio.sleep(0.01) # 尽量减少延迟
                        
            except Exception as e:
                logger.error(f"推流错误: {e}")
                if container:
                    try: container.close()
                    except: pass
                
                if not ws.closed:
                    logger.info("5秒后尝试重新连接 RTSP...")
                    await asyncio.sleep(5)
                else:
                    break
            finally:
                if container:
                    try: container.close()
                    except: pass
                    
    push_task = asyncio.create_task(push_video_frames())
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if json.loads(msg.data).get('action') == 'stop': break
            elif msg.type == WSMsgType.ERROR: break
    finally:
        push_task.cancel()
        logger.info("WebSocket 连接已关闭")
    
    return ws

async def init_app():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get('/api/ws', websocket_handler)
    app.router.add_post('/api/gimbal/control', handle_gimbal_control)
    app.router.add_post('/api/gimbal/angle', handle_gimbal_angle)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/{path:.*}', handle_static)
    return app

if __name__ == '__main__':
    app = asyncio.get_event_loop().run_until_complete(init_app())
    web.run_app(app, host='0.0.0.0', port=SERVER_PORT)
