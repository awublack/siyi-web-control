#!/usr/bin/env python3
"""实例三（curses版）：键盘控制云台，按住持续移动，松开停止"""
import socket
import struct
import curses
import time

CAMERA_IP = "192.168.144.25"
UDP_PORT = 37260
SPEED_DEG_PER_SEC = 50.0  # 移动速度

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

def build_position_cmd(yaw_deg, pitch_deg):
    header = bytes.fromhex("556601040000000E")
    yaw_val = int(yaw_deg * 10)
    pitch_val = int(pitch_deg * 10)
    yaw_val = max(-32768, min(32767, yaw_val))
    pitch_val = max(-32768, min(32767, pitch_val))
    yaw_bytes = struct.pack('<h', yaw_val)
    pitch_bytes = struct.pack('<h', pitch_val)
    payload = header + yaw_bytes + pitch_bytes
    crc = calculate_crc16_xmodem(payload)
    return payload + crc

def main(stdscr):
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(1)   # 非阻塞
    stdscr.clear()
    stdscr.addstr(0, 0, "键盘控制：W上 S下 A左 D右 | 空格回中 | ESC退出")
    stdscr.refresh()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_yaw = 0.0
    target_pitch = 0.0
    pressed = {'w': False, 's': False, 'a': False, 'd': False}
    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = now - last_time
        last_time = now
        dt = min(dt, 0.1)

        # 处理按键
        c = stdscr.getch()
        if c != -1:
            if c == ord('w'): pressed['w'] = True
            elif c == ord('s'): pressed['s'] = True
            elif c == ord('a'): pressed['a'] = True
            elif c == ord('d'): pressed['d'] = True
            elif c == ord(' '): 
                target_yaw = 0.0
                target_pitch = 0.0
            elif c == 27:  # ESC
                running = False
                break
        else:
            # 无按键时，将所有方向键标记为 False（模拟释放）
            # 注意：curses 无法检测按键释放，所以我们采用一个简单的策略：
            # 每次循环如果没有新的按键输入，就认为所有键都已释放。
            # 但这会导致一个问题：如果用户一直按住一个键，curses 会持续返回该键码，
            # 所以只要 getch 返回 -1 就代表没有键被按住。
            pressed = {'w': False, 's': False, 'a': False, 'd': False}

        # 计算方向
        dx, dy = 0, 0
        if pressed['w']: dy = -1
        if pressed['s']: dy = 1
        if pressed['a']: dx = -1
        if pressed['d']: dx = 1

        target_yaw += dx * SPEED_DEG_PER_SEC * dt
        target_pitch += dy * SPEED_DEG_PER_SEC * dt
        target_yaw = max(-135.0, min(135.0, target_yaw))
        target_pitch = max(-90.0, min(90.0, target_pitch))

        cmd = build_position_cmd(target_yaw, target_pitch)
        sock.sendto(cmd, (CAMERA_IP, UDP_PORT))

        # 显示状态
        stdscr.addstr(2, 0, f"Yaw: {target_yaw:+7.1f}°  Pitch: {target_pitch:+7.1f}°  ")
        stdscr.refresh()

        time.sleep(0.033)  # 30Hz

    sock.close()
    stdscr.clear()
    stdscr.addstr(0, 0, "已退出")
    stdscr.refresh()
    time.sleep(1)

if __name__ == '__main__':
    curses.wrapper(main)
