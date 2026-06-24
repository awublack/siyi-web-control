# SIYI Web Control

SIYI A8 Mini 云台相机的 Web 控制界面和后端服务器。提供实时视频流、云台控制和目标跟踪功能。

## 项目结构

```
siyi_web_control/
├── main.py                      # 主服务器（集成前后端）
├── siyi-backend-server-resilient.py  # 后端服务器（弹性版本）
├── siyi-backend-server-resilient (Copy 3).py  # 后端服务器备份
├── index.html                   # Web 控制界面
├── siyi-a8-tracker.py          # A8 跟踪器脚本
├── siyi-a8-tracker (Copy).py   # A8 跟踪器备份
├── testsiyi.py                 # 测试脚本
├── track-object/               # 目标跟踪 v1
├── track-object-v2/            # 目标跟踪 v2
├── track-object-v3fft/         # 目标跟踪 v3 (FFT)
├── track-object-v4select/      # 目标跟踪 v4 (选择器)
├── track-objectv5/             # 目标跟踪 v5
├── track-object-v6/            # 目标跟踪 v6 - 改进版本
├── track-object-position- v7/  # 目标跟踪 v7 - 位置跟踪增强
├── track-object-position- v8/  # 目标跟踪 v8 - 最新版本
└── tracking_log.txt            # 跟踪日志
```

## 快速开始

### 启动后端服务器

```bash
python3 siyi-backend-server-resilient.py
```

### 访问 Web 界面

在浏览器中打开：
```
http://localhost:8080/index.html
```

### 启动目标跟踪

```bash
# v6 版本
cd track-object-v6
python3 siyi-backend-server-resilient.py

# v7 版本（位置跟踪增强）
cd track-object-position-\ v7
python3 siyi-backend-server-resilient.py

# v8 版本（最新）
cd track-object-position-\ v8
python3 siyi-backend-server-resilient.py
```

## 功能特性

- 📹 **实时视频流** - 通过 RTSP 接收 SIYI A8 Mini 的视频流
- 🎮 **云台控制** - Web 界面控制云台俯仰、偏航、变焦
- 🎯 **目标跟踪** - 多版本目标跟踪算法（v1-v8）
  - v6: 改进的跟踪稳定性
  - v7: 位置跟踪增强功能
  - v8: 最新版本，优化性能
- 🔧 **弹性设计** - 自动重连、错误恢复机制
- 📊 **跟踪日志** - 自动记录跟踪数据和性能指标
- 🌐 **Web 界面** - 无需安装客户端，浏览器即可控制

## 硬件要求

- SIYI A8 Mini 云台相机
- Ubuntu/Linux 系统
- Python 3.8+

## 配置

编辑 `siyi-backend-server-resilient.py` 中的以下参数：

- `SIYI_IP`: 云台相机 IP 地址（默认：192.168.144.1）
- `SIYI_PORT`: 云台相机端口
- `RTSP_URL`: RTSP 视频流地址
- `WEB_PORT`: Web 服务端口（默认：8080）

## 版本历史

- **v8**: 最新版本，位置跟踪优化（2026-06-24）
- **v7**: 位置跟踪增强功能（2026-06-24）
- **v6**: 改进的跟踪算法和稳定性（2026-06-24）
- **v5**: 最新跟踪算法，优化选择器
- **v4**: 添加选择器功能
- **v3**: FFT 频域分析
- **v2**: 改进跟踪稳定性
- **v1**: 初始目标跟踪版本

## 最近更新 (2026-06-24)

- 新增 v7 和 v8 目标跟踪版本
- 添加位置跟踪增强功能
- 改进后端服务器弹性机制
- 添加跟踪日志记录功能
- 多个备份版本便于回滚测试

## 许可证

MIT License

## 联系方式

- GitHub: [@awublack](https://github.com/awublack)
- Email: awublack@126.com