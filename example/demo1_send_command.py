#!/usr/bin/env python3
"""实例一：发送一个位置命令帧，让云台转到指定角度"""
import socket
import struct
import time

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260

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

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 示例1：让云台转到 (yaw=30°, pitch=-10°)
    cmd = build_position_cmd(30.0, -10.0)
    print(f"发送命令帧 (共 {len(cmd)} 字节): {cmd.hex()}")
    sock.sendto(cmd, (CAMERA_IP, UDP_PORT))
    print("✓ 云台应转向 yaw=30°, pitch=-10°")
    
    time.sleep(2)
    
    # 示例2：回中 (yaw=0°, pitch=0°)
    cmd = build_position_cmd(0.0, 0.0)
    sock.sendto(cmd, (CAMERA_IP, UDP_PORT))
    print("✓ 云台回中")
    
    sock.close()

if __name__ == '__main__':
    main()
