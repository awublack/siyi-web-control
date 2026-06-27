# YOLO 目标跟踪使用说明

## 新增功能

### 1. YOLO 物体识别
- 集成 YOLOv8n 模型，支持实时物体检测
- 可识别 80+ 种常见物体类别（人、车、动物等）
- 检测框实时显示在视频流上

### 2. 鼠标交互
- **左键单击**：在 YOLO 模式下选择要跟踪的目标
- **右键单击**：立即取消/停止当前跟踪

## 快速开始

### 启动服务器
```bash
cd "/home/awu/.openclaw/workspace/siyi_web_control/track-object-position- v8"
python3 siyi-backend-server-resilient.py
```

### 访问 Web 界面
在浏览器中打开：
```
http://localhost:8080/index.html
```

## 使用流程

### 方法 1: YOLO 点击选择（推荐）
1. 点击 **▶ 启动服务** 按钮
2. 点击 **🔍 YOLO 检测** 按钮启用 YOLO
3. 等待 YOLO 检测到物体（屏幕上会显示黄色检测框）
4. **左键单击**想要跟踪的物体
5. 系统会自动开始跟踪该物体

### 方法 2: 框选目标
1. 点击 **🖱️ 框选目标** 按钮
2. 在视频上拖拽框选目标区域
3. 系统自动识别框选区域内的人脸或物体并开始跟踪

### 停止跟踪
- **右键单击**视频画面 - 立即停止所有跟踪
- 或点击 **🔄 重置跟踪** 按钮

## 界面说明

### 状态指示器
- **⏸️ 待机中** - 未进行跟踪
- **🎯 物体跟踪中** - 正在跟踪物体
- **🤖 人脸自动跟踪** - 正在跟踪人脸
- **🔍 YOLO 检测 ✓** - YOLO 已启用并显示检测框
- **🎮 手动控制中** - 正在使用键盘/鼠标手动控制

### 控制按钮
| 按钮 | 功能 |
|------|------|
| ▶ 启动服务 | 启动后端服务和视频流 |
| 🎯 云台回中 | 将云台复位到中心位置 |
| 🖱️ 框选目标 | 进入框选模式，拖拽选择目标 |
| 🔄 重置跟踪 | 停止当前跟踪，进入待机 |
| 🔍 YOLO 检测 | 开启/关闭 YOLO 物体识别 |
| ⏹ 停止服务 | 完全停止服务 |

### 键盘控制
- **W/S/A/D** - 控制云台上/下/左/右
- **Shift** - 加速控制
- **Space** - 云台回中

## 算法切换

### 人脸检测算法
- **Haar** - 极速，基础检测
- **LBP** - 抗光照变化
- **DNN YuNet** - 高精度（推荐）

### 物体跟踪算法
- **Hybrid** - MOSSE+CSRT 融合（强烈推荐，默认）
- **MOSSE** - 极速，但易漂移
- **KCF** - 平衡速度和精度
- **CSRT** - 高精度，可能卡顿
- **DaSiamRPN** - 深度学习，极高精度

## 环境变量配置

```bash
# 相机配置
export CAMERA_IP="192.168.144.25"
export RTSP_URL="rtsp://192.168.144.25:8554/main.264"

# YOLO 模型（可选）
export YOLO_MODEL="yolov8n.pt"  # nano 版本，最快

# 检测器类型
export DETECTOR_TYPE="haar"  # haar, lbp, dnn

# 跟踪器类型
export TRACKER_TYPE="hybrid"  # hybrid, mosse, kcf, csrt, dasiamrpn, ssd
```

## 依赖检查

运行前确保已安装：
```bash
pip3 list | grep -iE "ultralytics|opencv|aiohttp"
```

必需包：
- `ultralytics` - YOLO 模型
- `opencv-python` - 图像处理
- `aiohttp` - Web 服务器
- `PyAV` - 视频流处理
- `numpy` - 数值计算

## API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/yolo/click` | POST | 左键点击选择目标 |
| `/api/yolo/toggle` | POST | 切换 YOLO 开关 |
| `/api/status` | GET | 获取系统状态 |
| `/api/tracker/init` | POST | 初始化跟踪器 |
| `/api/tracker/reset` | POST | 重置跟踪器 |

### YOLO 点击选择 API 示例
```bash
curl -X POST http://localhost:8080/api/yolo/click \
  -H "Content-Type: application/json" \
  -d '{"x": 320, "y": 240}'
```

## 故障排除

### YOLO 不可用
- 检查 `ultralytics` 是否安装：`pip3 show ultralytics`
- 首次运行会下载模型文件（约 6MB）

### 视频流无法连接
- 检查相机 IP 是否正确
- 确认 RTSP 地址可访问
- 检查网络连接

### 跟踪不稳定
- 切换到 Hybrid 或 CSRT 跟踪器
- 确保目标在画面中有足够纹理
- 调整光照条件

## 文件结构

```
track-object-position- v8/
├── siyi-backend-server-resilient.py   # 后端服务器（含 YOLO）
├── index.html                         # 前端界面
├── yolov8n.pt                         # YOLO 模型文件
├── [其他模型文件...]
└── YOLO_USAGE.md                      # 本文件
```

## 更新日志

**v8.1 - 2026-06-27**
- ✅ 新增 YOLOv8 物体识别
- ✅ 左键单击选择目标
- ✅ 右键取消跟踪
- ✅ 实时检测框显示
- ✅ YOLO 开关控制
- ✅ 状态指示和日志