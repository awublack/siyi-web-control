import asyncio
import logging
import socket
import struct
import os  # 导入系统环境变量模块
from aiohttp import web
import cv2
import av
import queue
import threading

# ==================== 全局动态配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("GimbalSystem")

# 优先读取系统环境变量，若无则使用 SIYI 默认出厂参数
CAMERA_IP = os.environ.get('CAMERA_IP', "192.168.144.25")
RTSP_URL = os.environ.get('RTSP_URL', "rtsp://192.168.144.25:8554/main.264")
UDP_PORT = int(os.environ.get('UDP_PORT', "37260"))
SERVER_PORT = int(os.environ.get('SERVER_PORT', "8080"))

logger.info(f"⚙️ 系统配置加载成功: 相机IP={CAMERA_IP} | RTSP地址={RTSP_URL} | 视频服务端口={SERVER_PORT}")

# 全局状态跟踪器
class SystemState:
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1) # 严格控制无缓存队列
        self.websockets = set()
        self.current_yaw = 0
        self.current_pitch = 0

state = SystemState()

# ==================== SIYI A8 Mini 协议底层驱动 ====================
def crc16(data: bytes) -> int:
    """SIYI 协议标准 CRC16 校验"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

def send_gimbal_speed_command(yaw_speed, pitch_speed, ip=CAMERA_IP, port=UDP_PORT):
    """
    1. 速度控制接口 (用于键盘连续操控)
    范围: -100 到 100
    """
    pkt_header = struct.pack('<H B H H B', 0x6655, 0x00, 0x0002, 0x0001, 0x07)
    pkt_data = struct.pack('<b b', int(yaw_speed), int(pitch_speed))
    payload = pkt_header + pkt_data
    packet = payload + struct.pack('<H', crc16(payload))
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(packet, (ip, port))
    except Exception as e:
        logger.error(f"发送速度指令异常: {e}")

def send_gimbal_angle_command(yaw_deg, pitch_deg, ip=CAMERA_IP, port=UDP_PORT):
    """
    2. 绝对角度/增量角度精准控制接口 (用于指哪打哪)
    单位: 0.1度。输入 15.5 度传入 155
    """
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    
    # 结合 A8 Mini 物理极限做安全限幅
    yaw_val = max(-1350, min(1350, yaw_val))     # 航向角限制 -135° 到 135°
    pitch_val = max(-900, min(250, pitch_val))   # 俯仰角限制 -90° 到 25°
    
    # 构造角度指令包 (0x0B 角度同步控制命令)
    pkt_header = struct.pack('<H B H H B', 0x6655, 0x00, 0x0004, 0x0001, 0x0B)
    pkt_data = struct.pack('<h h', yaw_val, pitch_val)
    payload = pkt_header + pkt_data
    packet = payload + struct.pack('<H', crc16(payload))
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(packet, (ip, port))
        return True
    except Exception as e:
        logger.error(f"发送角度指令异常: {e}")
        return False

# ==================== 高频低延迟实时流核心引擎 ====================
def video_capture_worker():
    """独占线程：高频网络拉流并彻底压榨 FFmpeg 缓存"""
    logger.info("🎬 RTSP 超低延迟解码线程已点火...")
    
    options = {
        'fflags': 'nobuffer',
        'flags': 'low_delay',
        'analyzeduration': '100000',
        'probesize': '32768',
        'rtsp_transport': 'udp'
    }
    
    while state.running:
        container = None
        try:
            container = av.open(RTSP_URL, options=options)
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO' # 开启底层多线程并行解码
            
            for frame in container.decode(stream):
                if not state.running:
                    break
                    
                bgr_img = frame.to_nd_array(format='bgr24')
                
                # 如果队列满，剔除旧帧，永远保持最新鲜的画面
                if state.frame_queue.full():
                    try:
                        state.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                state.frame_queue.put(bgr_img)
                
        except Exception as e:
            logger.error(f"视频流解码中断或无法建立连接: {e}，3秒后尝试自愈重连...")
            if container:
                container.close()
            from time import sleep
            sleep(3)
    
    if container:
        container.close()
    logger.info("🎬 RTSP 解码线程已安全终止")

# ==================== HTTP & WebSocket 异步网络处理器 ====================
async def image_push_loop(app):
    """异步任务：将最新的图片高速压缩并分发"""
    logger.info("📡 WebSocket 高频图传广播核心已就绪")
    
    while True:
        if not state.running or not state.websockets:
            await asyncio.sleep(0.1)
            continue
            
        try:
            loop = asyncio.get_event_loop()
            
            def fetch_and_encode():
                try:
                    img = state.frame_queue.get_nowait()
                    h, w = img.shape[:2]
                    target_w = 800
                    target_h = int(h * (target_w / w))
                    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    return buffer.tobytes()
                except queue.Empty:
                    return None
                    
            jpeg_bytes = await loop.run_in_executor(None, fetch_and_encode)
            
            if jpeg_bytes:
                disconnected = set()
                for ws in state.websockets:
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception:
                        disconnected.add(ws)
                if disconnected:
                    state.websockets -= disconnected
                    
            await asyncio.sleep(0.005)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"图传发送环路异动: {e}")
            await asyncio.sleep(0.1)

async def handle_ws(request):
    """处理前端视频流 WebSocket"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    state.websockets.add(ws)
    logger.info(f"🔗 前端控制台 WebSocket 连接成功，当前总连接数: {len(state.websockets)}")
    
    try:
        async for msg in ws:
            pass
    finally:
        state.websockets.discard(ws)
        logger.info("❌ 前端控制台 WebSocket 连接已安全切断")
    return ws

async def handle_control(request):
    """处理键盘速度控制 API"""
    try:
        data = await request.json()
        yaw = data.get('yaw', 0)
        pitch = data.get('pitch', 0)
        state.current_yaw = yaw
        state.current_pitch = pitch
        
        send_gimbal_speed_command(yaw, pitch)
        return web.json_response({'status': 'success'})
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)}, status=400)

async def handle_angle(request):
    """处理点击画面后指哪打哪的角度增量 API"""
    try:
        data = await request.json()
        yaw_delta = data.get('yaw', 0)
        pitch_delta = data.get('pitch', 0)
        
        logger.info(f"📐 接收到指哪打哪微调指令 - 偏航: {yaw_delta}°, 俯仰: {pitch_delta}°")
        send_gimbal_angle_command(yaw_delta, pitch_delta)
        return web.json_response({'status': 'success'})
    except Exception as e:
        return web.json_response({'status': 'error', 'message': str(e)}, status=400)

async def handle_status(request):
    """服务存活状态心跳检查"""
    return web.json_response({
        'status': 'running' if state.running else 'stopped',
        'camera_ip': CAMERA_IP,
        'current_yaw_speed': state.current_yaw,
        'current_pitch_speed': state.current_pitch
    })

async def index(request):
    """静态主页托管处理器（执行安全干净的字符串字面值替换）"""
    rendered_template = HTML_TEMPLATE.replace('{{CAMERA_IP}}', CAMERA_IP)
    return web.Response(text=rendered_template, content_type='text/html')

# ==================== 生命周期管理 ====================
async def start_background_tasks(app):
    state.running = True
    app['capture_thread'] = threading.Thread(target=video_capture_worker, daemon=True)
    app['capture_thread'].start()
    app['pusher_task'] = asyncio.create_task(image_push_loop(app))

async def cleanup_background_tasks(app):
    state.running = False
    
    if 'pusher_task' in app and app['pusher_task']:
        app['pusher_task'].cancel()
        try:
            await app['pusher_task']
        except asyncio.CancelledError:
            pass
        app['pusher_task'] = None
        
    if 'capture_thread' in app:
        app['capture_thread'].join(timeout=2)
    logger.info("🛑 全局生命周期安全关闭完成")

# ==================== 前端原汁原味 HTML 交互套件（完全脱离 f-string 冲突） ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIYI A8 Mini 云台控制系统 Pro</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f111a 0%, #16192b 100%);
            min-height: 100vh; color: #fff; padding: 20px;
        }
        .container { max-width: 1300px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 25px; color: #00d9ff; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }
        .main-grid { display: grid; grid-template-columns: 1fr 360px; gap: 20px; }
        .video-section {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px; padding: 15px; backdrop-filter: blur(20px); display: flex; flex-direction: column; gap: 15px;
        }
        .video-container {
            position: relative; width: 100%; aspect-ratio: 4/3; background: #05070f;
            border-radius: 12px; overflow: hidden; box-shadow: inset 0 0 40px rgba(0,0,0,0.8); cursor: crosshair;
        }
        #videoStream { width: 100%; height: 100%; object-fit: contain; display: none; }
        .crosshair-guide {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 40px; height: 40px; border: 2px solid rgba(0, 217, 255, 0.3); border-radius: 50%; pointer-events: none; display: none;
        }
        .crosshair-guide::before, .crosshair-guide::after { content: ''; position: absolute; background: rgba(0, 217, 255, 0.4); }
        .crosshair-guide::before { top: 19px; left: -10px; width: 60px; height: 2px; }
        .crosshair-guide::after { left: 19px; top: -10px; width: 2px; height: 60px; }
        .status-overlay { position: absolute; text-align: center; padding: 20px; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .status-overlay h2 { font-size: 22px; margin-bottom: 12px; color: #00d9ff; }
        .status-item { margin: 8px 0; font-size: 15px; color: #ccc; }
        .status-value { color: #00ff88; font-weight: 600; }
        .control-panel {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px; padding: 20px; backdrop-filter: blur(20px); display: flex; flex-direction: column;
        }
        .status-card { background: rgba(0, 0, 0, 0.4); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #00d9ff; }
        .status-item-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; }
        .status-label { color: #99a2bd; }
        .mode-indicator { padding: 12px 15px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 14px; }
        .mode-auto { background: linear-gradient(135deg, #05c46b, #0be881); color: #fff; }
        .mode-manual { background: linear-gradient(135deg, #ef5777, #f53b57); color: #fff; }
        .control-buttons { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px; }
        .btn {
            padding: 12px 15px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;
            font-size: 14px; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 6px;
        }
        .btn-primary { background: linear-gradient(135deg, #00d9ff, #0084ff); color: #fff; }
        .btn-danger { background: linear-gradient(135deg, #ff5e62, #ff2525); color: #fff; }
        .btn-success { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: #fff; }
        .btn:hover:not(:disabled) { transform: translateY(-2px); filter: brightness(1.1); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4); }
        .btn:disabled { opacity: 0.3; cursor: not-allowed; }
        .keyboard-help { background: rgba(0, 0, 0, 0.25); border-radius: 10px; padding: 12px 15px; margin-top: auto; }
        .keyboard-help h3 { margin-bottom: 8px; color: #00d9ff; font-size: 13px; letter-spacing: 0.5px;}
        .key-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
        .key-row:last-child { border-bottom: none; }
        .key { background: rgba(255, 255, 255, 0.12); padding: 2px 8px; border-radius: 4px; font-family: monospace; border: 1px solid rgba(255,255,255,0.1); color: #00d9ff;}
        .log-container { background: rgba(5, 7, 15, 0.8); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; height: 140px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        .log-entry { margin-bottom: 4px; line-height: 1.4; border-left: 2px solid transparent; padding-left: 6px;}
        .log-info { color: #00d9ff; border-left-color: #00d9ff; }
        .log-success { color: #00ff88; border-left-color: #00ff88; }
        .log-warning { color: #ffd43b; border-left-color: #ffd43b; }
        .log-error { color: #ff6b6b; border-left-color: #ff6b6b; }
        .settings-section { margin-bottom: 15px; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 10px;}
        .setting-item { margin-bottom: 10px; }
        .setting-label { display: block; margin-bottom: 4px; font-size: 12px; color: #8e9aaf; }
        .setting-input { width: 100%; padding: 8px 12px; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; background: rgba(0, 0, 0, 0.4); color: #fff; font-size: 13px; }
        .connection-status { padding: 10px; border-radius: 8px; margin-bottom: 15px; text-align: center; font-weight: bold; font-size: 13px; }
        .conn-ok { background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; }
        .conn-error { background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); color: #ff6b6b; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 SIYI A8 Mini 云台控制系统 Pro</h1>
        <div class="main-grid">
            <div class="video-section">
                <div class="video-container" id="videoContainer" onclick="handleVideoClick(event)">
                    <img id="videoStream" alt="Video Stream">
                    <div class="crosshair-guide" id="crosshairGuide"></div>
                    <div class="status-overlay" id="videoOverlay">
                        <h2>📡 控制系统就绪</h2>
                        <div class="status-item">相机 IP: <span class="status-value" id="displayCameraIp">{{CAMERA_IP}}</span></div>
                        <div class="status-item">状态：<span class="status-value" id="mainStatus" style="color:#ffd43b">等待服务拉起...</span></div>
                    </div>
                </div>
                <div class="log-container" id="logContainer"></div>
            </div>
            
            <div class="control-panel">
                <div class="connection-status conn-error" id="connStatus">🔴 系统未连接</div>
                <div class="mode-indicator mode-auto" id="modeIndicator">🤖 自动跟踪模式</div>
                
                <div class="status-card">
                    <div class="status-item-row"><span class="status-label">发送 Yaw 指令</span><span class="status-value" id="statusYaw">0</span></div>
                    <div class="status-item-row"><span class="status-label">发送 Pitch 指令</span><span class="status-value" id="statusPitch">0</span></div>
                    <div class="status-item-row"><span class="status-label">后端通信状态</span><span class="status-value" id="backendStatus" style="color: #ff6b6b">离线</span></div>
                </div>
                
                <div class="settings-section">
                    <div class="setting-item">
                        <label class="setting-label">摄像头 IP</label>
                        <input type="text" class="setting-input" id="cameraIp" value="{{CAMERA_IP}}" disabled>
                    </div>
                </div>

                <div class="control-buttons">
                    <button class="btn btn-success" id="btnStart" onclick="startService()">▶ 启动系统</button>
                    <button class="btn btn-danger" id="btnStop" onclick="stopService()" disabled>⏹ 停止系统</button>
                    <button class="btn btn-primary" onclick="centerGimbal()">🎯 云台回中</button>
                    <button class="btn btn-primary" onclick="toggleMode()">🔄 切换模式</button>
                </div>
                
                <div class="keyboard-help">
                    <h3>⌨️ 控制技巧</h3>
                    <div class="key-row"><span><span class="key">W</span> / <span class="key">S</span> 俯仰控制 | <span class="key">A</span> / <span class="key">D</span> 航向控制</span></div>
                    <div class="key-row"><span><span class="key">Shift</span> 键盘调速(全速) | <span class="key">Space</span> 一键刹车回中</span></div>
                    <div class="key-row"><span style="color: #00d9ff">🖱️ 鼠标左键点击画面任意处，云台全速锁定对齐</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let manualMode = false;
        let serviceRunning = false;
        let controlInterval = null;
        let ws = null;
        const keysPressed = {};
        let lastSentYaw = 0;
        let lastSentPitch = 0;
        let isImageLoading = false;

        function log(message, type = 'info') {
            const logContainer = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
            if (logContainer.children.length > 50) logContainer.removeChild(logContainer.firstChild);
        }

        function initWebSocket() {
            const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`;
            if (ws) ws.close();
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'blob';
            ws.onopen = () => {
                log('🚀 实时高清视频链路已建立', 'success');
                document.getElementById('videoStream').style.display = 'block';
                document.getElementById('crosshairGuide').style.display = 'block';
                document.getElementById('videoOverlay').style.display = 'none';
            };
            ws.onmessage = (event) => {
                if (!serviceRunning || isImageLoading) return;
                if (event.data instanceof Blob) {
                    isImageLoading = true;
                    const img = document.getElementById('videoStream');
                    const url = URL.createObjectURL(event.data);
                    const oldUrl = img.src;
                    img.onload = () => {
                        isImageLoading = false;
                        if (oldUrl && oldUrl.startsWith('blob:')) URL.revokeObjectURL(oldUrl);
                    };
                    img.src = url;
                }
            };
            ws.onclose = () => {
                document.getElementById('videoStream').style.display = 'none';
                document.getElementById('crosshairGuide').style.display = 'none';
                document.getElementById('videoOverlay').style.display = 'block';
                if (serviceRunning) setTimeout(initWebSocket, 2000);
            };
        }

        async function sendGimbalSpeed(yaw, pitch, isStopCmd = false) {
            if (!isStopCmd && yaw === lastSentYaw && pitch === lastSentPitch) return;
            lastSentYaw = yaw; lastSentPitch = pitch;
            document.getElementById('statusYaw').textContent = yaw;
            document.getElementById('statusPitch').textContent = pitch;
            try {
                await fetch('/api/gimbal/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ yaw, pitch })
                });
            } catch (e) {}
        }

        // 指哪打哪几何转换
        async function handleVideoClick(event) {
            if (!serviceRunning || !manualMode) return;
            
            const container = document.getElementById('videoContainer');
            const rect = container.getBoundingClientRect();
            
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;
            
            const deltaX = clickX - (rect.width / 2);
            const deltaY = (rect.height / 2) - clickY;
            
            const HFOV = 55.0;
            const degreesPerPixel = HFOV / rect.width;
            
            const targetYawDelta = deltaX * degreesPerPixel;
            const targetPitchDelta = deltaY * degreesPerPixel;
            
            log(`🎯 指哪打哪锁定 -> 航向修正: ${targetYawDelta.toFixed(1)}°, 俯仰修正: ${targetPitchDelta.toFixed(1)}°`, 'success');
            createTargetMarker(clickX, clickY);

            try {
                await fetch('/api/gimbal/angle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ yaw: targetYawDelta, pitch: targetPitchDelta })
                });
            } catch (error) {
                log('指令抛送链路阻塞', 'error');
            }
        }

        function createTargetMarker(x, y) {
            const container = document.getElementById('videoContainer');
            const marker = document.createElement('div');
            marker.style.position = 'absolute'; marker.style.left = `${x - 15}px`; marker.style.top = `${y - 15}px`;
            marker.style.width = '30px'; marker.style.height = '30px'; marker.style.border = '2px dashed #ff3838';
            marker.style.borderRadius = '50%'; marker.style.pointerEvents = 'none';
            marker.style.animation = 'ping 0.4s ease-out forwards';
            container.appendChild(marker);
            setTimeout(() => marker.remove(), 400);
        }

        function centerGimbal() {
            log('🎯 云台下达快速刹车复位指令', 'info');
            sendGimbalSpeed(0, 0, true);
            setTimeout(() => sendGimbalSpeed(0, 0, true), 50);
        }

        function toggleMode() {
            manualMode = !manualMode;
            const indicator = document.getElementById('modeIndicator');
            if (manualMode) {
                indicator.className = 'mode-indicator mode-manual'; indicator.textContent = '🎮 手动控制激活 (支持键盘及鼠标点击)';
            } else {
                indicator.className = 'mode-indicator mode-auto'; indicator.textContent = '🤖 机器视觉自动跟踪中';
                centerGimbal();
            }
        }

        function startService() {
            serviceRunning = true; manualMode = true; toggleMode(); manualMode = true;
            document.getElementById('btnStart').disabled = true; document.getElementById('btnStop').disabled = false;
            document.getElementById('mainStatus').textContent = '● 正常运行'; document.getElementById('mainStatus').style.color = '#00ff88';
            log('🚀 控制台后台引擎全面激活', 'success');
            document.getElementById('connStatus').className = 'connection-status conn-ok';
            document.getElementById('connStatus').textContent = '🟢 系统运行中 | 已锁定 A8 Mini 设备';
            initWebSocket(); centerGimbal(); startKeyboardControl();
        }

        function stopService() {
            serviceRunning = false;
            if (ws) { ws.close(); ws = null; }
            if (controlInterval) clearInterval(controlInterval);
            document.getElementById('btnStart').disabled = false; document.getElementById('btnStop').disabled = true;
            document.getElementById('mainStatus').textContent = '挂起停止'; document.getElementById('mainStatus').style.color = '#ff6b6b';
            document.getElementById('connStatus').className = 'connection-status conn-error';
            document.getElementById('connStatus').textContent = '🔴 系统已关闭';
            centerGimbal(); log('⏹ 核心控制服务成功安全挂起', 'warning');
        }

        function startKeyboardControl() {
            if (controlInterval) clearInterval(controlInterval);
            Object.keys(keysPressed).forEach(k => keysPressed[k] = false);
            
            document.addEventListener('keydown', (e) => {
                if (!serviceRunning) return;
                keysPressed[e.key.toLowerCase()] = true;
                if (e.code === 'Space') { e.preventDefault(); centerGimbal(); }
            });
            document.addEventListener('keyup', (e) => { keysPressed[e.key.toLowerCase()] = false; });

            controlInterval = setInterval(() => {
                if (!serviceRunning || !manualMode) return;
                const boost = keysPressed['shift'];
                const speed = boost ? 100 : 45;
                let yaw = 0, pitch = 0;
                if (keysPressed['w']) pitch = speed;
                if (keysPressed['s']) pitch = -speed;
                if (keysPressed['a']) yaw = -speed;
                if (keysPressed['d']) yaw = speed;

                if (yaw === 0 && pitch === 0) {
                    if (lastSentYaw !== 0 || lastSentPitch !== 0) sendGimbalSpeed(0, 0, true);
                } else {
                    sendGimbalSpeed(yaw, pitch);
                }
            }, 50);
        }

        window.addEventListener('load', () => {
            setInterval(async () => {
                try {
                    const r = await fetch('/api/status'); const d = await r.json();
                    if (d.status) {
                        document.getElementById('backendStatus').textContent = '正常通信';
                        document.getElementById('backendStatus').style.color = '#00ff88';
                    }
                } catch (e) {
                    document.getElementById('backendStatus').textContent = '断开';
                    document.getElementById('backendStatus').style.color = '#ff6b6b';
                }
            }, 3000);
        });
    </script>
</body>
</html>
"""

# ==================== Web 容器启动入口 ====================
def main():
    app = web.Application()
    
    # 挂载路由
    app.router.add_get('/', index)
    app.router.add_get('/api/ws', handle_ws)
    app.router.add_post('/api/gimbal/control', handle_control)
    app.router.add_post('/api/gimbal/angle', handle_angle)
    app.router.add_get('/api/status', handle_status)
    
    # 注入全局生命周期事件
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    logger.info("==================================================")
    logger.info(f"🚀 低延迟控制服务准备就绪！请在浏览器中访问: http://localhost:{SERVER_PORT}")
    logger.info("==================================================")
    
    web.run_app(app, host='0.0.0.0', port=SERVER_PORT, access_log=None)

if __name__ == '__main__':
    main()
