import cv2
import os

base_dir = "/home/awu/.openclaw/workspace/siyi_web_control"
files = {
    "model": "dasiamrpn_model.onnx",
    "kernel_cls1": "dasiamrpn_kernel_cls1.onnx",
    "kernel_r1": "dasiamrpn_kernel_r1.onnx"
}

print("=== 1. 检查文件是否存在 ===")
all_exist = True
for name, filename in files.items():
    path = os.path.join(base_dir, filename)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"{filename}: {'✅ 存在' if exists else '❌ 缺失'} (大小: {size} bytes)")
    if not exists: all_exist = False

if not all_exist:
    print("\n❌ 结论：文件缺失！请确保这三个文件都在当前目录下，且名字完全一致。")
else:
    print("\n=== 2. 尝试加载模型 ===")
    model = os.path.join(base_dir, files["model"])
    k1 = os.path.join(base_dir, files["kernel_cls1"])
    k2 = os.path.join(base_dir, files["kernel_r1"])
    
    try:
        tracker = cv2.TrackerDaSiamRPN_create(model, k1, k2)
        print("✅ 结论：使用 cv2 API 加载成功！")
    except Exception as e:
        print(f"❌ cv2 API 失败: {e}")
        try:
            tracker = cv2.legacy.TrackerDaSiamRPN_create(model, k1, k2)
            print("✅ 结论：使用 cv2.legacy API 加载成功！")
        except Exception as e2:
            print(f"❌ cv2.legacy API 也失败: {e2}")
            print("💡 结论：可能是模型文件损坏，或 OpenCV 4.13.0 不兼容此版本的 ONNX 模型。")
