import carla
import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime

# ========== 配置参数 ==========
# 摄像头分辨率
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 数据保存根目录
SAVE_ROOT = "carla_data_collect"
# 自车蓝图名称（可替换为其他车型）
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
# Carla服务器地址和端口
CARLA_HOST = "localhost"
CARLA_PORT = 2000

# 初始化全局变量，用于存储数据和控制状态
image_save_paths = {
    "front_image": None, "back_image": None, "left_image": None, "right_image": None
}  # 关键修改：键名与CSV表头一致
vehicle_data = {}
data_lock = False  # 数据保存锁，避免多线程冲突
csv_writer = None
csv_file = None

def init_save_directory():
    """初始化数据保存目录"""
    global SAVE_ROOT
    # 按时间戳创建子目录，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_ROOT = os.path.join(SAVE_ROOT, timestamp)
    for dir_name in ["images/front", "images/back", "images/left", "images/right", "csv"]:
        os.makedirs(os.path.join(SAVE_ROOT, dir_name), exist_ok=True)
    return SAVE_ROOT

def init_csv_writer():
    """初始化CSV文件写入器"""
    global csv_writer, csv_file
    csv_path = os.path.join(SAVE_ROOT, "csv/global_vehicle_data.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    # CSV表头：时间戳+图像路径+车辆状态+底盘控制信息
    fieldnames = [
        "timestamp",
        "front_image", "back_image", "left_image", "right_image",
        "global_x", "global_y", "global_z",  # 全局坐标
        "velocity_x", "velocity_y", "velocity_z",  # 速度（m/s）
        "acceleration_x", "acceleration_y", "acceleration_z",  # 加速度（m/s²）
        "steer",  # 方向盘转角（-1~1）
        "throttle",  # 油门（0~1）
        "brake",  # 刹车（0~1）
        "speed_kmh"  # 车辆速度（km/h）
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    return csv_writer

def process_image(image, direction):
    """摄像头图像回调函数：处理并保存图像"""
    global image_save_paths, data_lock
    if data_lock:  # 避免数据写入冲突
        return
    # 转换Carla图像为numpy数组（RGBA→RGB）
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    # 生成图像文件名（时间戳+方向）
    timestamp = f"{image.timestamp:.6f}".replace(".", "_")
    image_name = f"{timestamp}_{direction}.png"
    image_path = os.path.join(SAVE_ROOT, f"images/{direction}/{image_name}")
    # 保存图像
    cv2.imwrite(image_path, array)
    # 记录图像路径：关键修改→使用带_image后缀的键名
    image_key = f"{direction}_image"
    image_save_paths[image_key] = image_path

def get_vehicle_state(vehicle):
    """获取车辆的全局状态和底盘控制信息"""
    global vehicle_data
    if not vehicle or not vehicle.is_alive:
        return {}
    # 1. 全局坐标（Carla世界坐标系）
    transform = vehicle.get_transform()
    location = transform.location
    # 2. 速度和加速度（线性）
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    # 3. 底盘控制信息（油门、刹车、方向盘转角）
    control = vehicle.get_control()
    # 4. 车辆速度（km/h）：将m/s转换为km/h
    speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    # 组装数据
    vehicle_data = {
        "timestamp": time.time(),
        "global_x": location.x,
        "global_y": location.y,
        "global_z": location.z,
        "velocity_x": velocity.x,
        "velocity_y": velocity.y,
        "velocity_z": velocity.z,
        "acceleration_x": acceleration.x,
        "acceleration_y": acceleration.y,
        "acceleration_z": acceleration.z,
        "steer": control.steer,
        "throttle": control.throttle,
        "brake": control.brake,
        "speed_kmh": speed_kmh
    }
    return vehicle_data

def save_vehicle_data():
    """保存车辆数据到CSV，关联图像路径"""
    global csv_writer, image_save_paths, vehicle_data, data_lock
    if data_lock:
        return
    # 关键优化：校验车辆数据和图像路径是否完整（无None）
    if not vehicle_data:
        return
    for path in image_save_paths.values():
        if path is None:
            return
    data_lock = True  # 加锁
    try:
        # 合并图像路径和车辆数据
        save_data = {**vehicle_data, **image_save_paths}
        # 写入CSV
        csv_writer.writerow(save_data)
        csv_file.flush()  # 强制刷新缓冲区
        print(f"成功保存一组数据：时间戳={save_data['timestamp']:.2f}")
    except Exception as e:
        print(f"数据保存失败：{e}")
    finally:
        # 重置图像路径和解锁
        image_save_paths = {k: None for k in image_save_paths.keys()}
        data_lock = False  # 解锁

def spawn_cameras(vehicle, world, blueprint_library):
    """在自车上挂载前后左右四个摄像头"""
    cameras = []
    # 摄像头配置：方向、挂载位置、旋转角度
    camera_configs = [
        ("front", carla.Transform(carla.Location(x=2.0, z=1.8)), carla.Rotation(pitch=0)),  # 前视
        ("back", carla.Transform(carla.Location(x=-2.0, z=1.8)), carla.Rotation(pitch=0, yaw=180)),  # 后视
        ("left", carla.Transform(carla.Location(y=1.0, z=1.8)), carla.Rotation(pitch=0, yaw=-90)),  # 左视
        ("right", carla.Transform(carla.Location(y=-1.0, z=1.8)), carla.Rotation(pitch=0, yaw=90))  # 右视
    ]
    # 创建摄像头蓝图
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1")  # 采集频率：10Hz（可调整）
    # 生成摄像头并注册回调
    for direction, transform, rotation in camera_configs:
        transform.rotation = rotation
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        camera.listen(lambda image, d=direction: process_image(image, d))
        cameras.append(camera)
        print(f"已挂载{direction}摄像头")
    return cameras

def manual_control(vehicle):
    """手动控制车辆：键盘操作（基于Pygame）"""
    import pygame
    pygame.init()
    # 优化：设置Pygame窗口为非阻塞模式
    pygame.display.set_mode((200, 200))
    pygame.display.set_caption("Carla Manual Control")
    clock = pygame.time.Clock()
    control = carla.VehicleControl()
    running = True  # 运行标志位
    print("===== 手动控制按键说明 =====")
    print("W：前进  S：后退  A：左转  D：右转")
    print("空格：刹车  Q：退出采集")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 按键按下事件（避免持续触发）
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        # 键盘按键检测（持续按键）
        keys = pygame.key.get_pressed()
        # 初始化控制参数
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        control.hand_brake = False
        # 前进/后退
        if keys[pygame.K_w]:
            control.throttle = 1.0
        elif keys[pygame.K_s]:
            control.throttle = -1.0
        # 转向
        if keys[pygame.K_a]:
            control.steer = -0.5
        elif keys[pygame.K_d]:
            control.steer = 0.5
        # 刹车
        if keys[pygame.K_SPACE]:
            control.brake = 1.0
        # 应用控制
        vehicle.apply_control(control)
        # 获取车辆状态并保存
        get_vehicle_state(vehicle)
        save_vehicle_data()
        # 帧率控制
        clock.tick(30)
        pygame.display.flip()
    # 退出Pygame
    pygame.quit()

def main():
    # 1. 初始化数据保存目录和CSV
    init_save_directory()
    init_csv_writer()
    print(f"数据将保存到：{SAVE_ROOT}")

    # 2. 连接Carla服务器
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 3. 生成自车
    vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT)
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("未找到车辆生成点！")
        return
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print(f"已生成自车：{VEHICLE_BLUEPRINT}")

    # 4. 挂载四方向摄像头
    cameras = spawn_cameras(vehicle, world, blueprint_library)

    try:
        # 5. 启动手动控制和数据采集
        manual_control(vehicle)
    except KeyboardInterrupt:
        print("\n采集被用户中断！")
    except Exception as e:
        print(f"手动控制异常：{e}")
    finally:
        # 6. 销毁所有Actor，关闭文件
        print("开始清理资源...")
        for camera in cameras:
            if camera.is_alive:
                camera.destroy()
        if vehicle.is_alive:
            vehicle.destroy()
        if csv_file:
            csv_file.close()
        print("资源清理完成，数据采集结束！")

if __name__ == "__main__":
    main()
