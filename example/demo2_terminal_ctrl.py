#!/usr/bin/env python3
"""实例二：终端命令行控制云台（单次定位或交互模式）"""
import socket
import argparse
import struct
import sys
import time

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260

# ====== 完整的 CRC 和命令帧构造函数 ======
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
    return crc.to_bytes(2, 'little')

def build_position_cmd(yaw_deg: float, pitch_deg: float) -> bytes:
    """构造 CMD 0x0E 位置控制帧"""
    header = bytes.fromhex("556601040000000E")  # 固定头部
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    # 限幅
    yaw_val = max(-32768, min(32767, yaw_val))
    pitch_val = max(-32768, min(32767, pitch_val))
    yaw_bytes = struct.pack('<h', yaw_val)      # 小端有符号16位
    pitch_bytes = struct.pack('<h', pitch_val)
    payload = header + yaw_bytes + pitch_bytes
    crc = calculate_crc16_xmodem(payload)
    return payload + crc

# ====== 发送函数 ======
def send_once(sock, yaw, pitch):
    cmd = build_position_cmd(yaw, pitch)
    sock.sendto(cmd, (CAMERA_IP, UDP_PORT))
    print(f"→ 发送至 yaw={yaw:.1f}°, pitch={pitch:.1f}°")

def interactive_mode(sock):
    """交互式：输入 'yaw pitch' 发送，输入 q 退出"""
    print("交互模式：输入 'yaw pitch' 发送（如 30 -10），输入 q 退出")
    while True:
        try:
            line = input(">>> ").strip()
            if line.lower() == 'q':
                break
            parts = line.split()
            if len(parts) == 2:
                yaw = float(parts[0])
                pitch = float(parts[1])
                send_once(sock, yaw, pitch)
            else:
                print("格式错误，请输入 'yaw pitch'")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="SIYI A8 Mini 终端控制")
    parser.add_argument('--yaw', type=float, help='目标偏航角')
    parser.add_argument('--pitch', type=float, help='目标俯仰角')
    parser.add_argument('--interactive', '-i', action='store_true', help='进入交互模式')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    if args.interactive:
        interactive_mode(sock)
    elif args.yaw is not None and args.pitch is not None:
        send_once(sock, args.yaw, args.pitch)
    else:
        print("请指定 --yaw 和 --pitch，或使用 --interactive 进入交互模式")
    
    sock.close()

if __name__ == '__main__':
    main()
