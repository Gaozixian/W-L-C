import carla
import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from threading import Lock


class CarlaDataCollector:
    def __init__(self, client, vehicle, save_root="carla_data_collect", camera_res=(640, 480), sensor_tick=0.1):
        """
        初始化数据采集器
        :param client: Carla客户端对象（已连接服务器）
        :param vehicle: 待采集数据的车辆Actor（你的特斯拉车辆）
        :param save_root: 数据保存根目录
        :param camera_res: 摄像头分辨率 (width, height)
        :param sensor_tick: 传感器采集频率（秒/次）
        """
        self.client = client
        self.vehicle = vehicle
        self.world = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.camera_width, self.camera_height = camera_res
        self.sensor_tick = sensor_tick

        # 数据保存相关
        self.save_root = self._init_save_directory(save_root)
        self.csv_file, self.csv_writer = self._init_csv_writer()

        # 全局变量与锁（避免多线程冲突）
        self.image_save_paths = {
            "front_image": None, "back_image": None,
            "left_image": None, "right_image": None
        }
        self.vehicle_data = {}
        self.dimensions = {}
        self.data_lock = Lock()
        self.cameras = []  # 存储摄像头Actor，用于后续销毁

    def _init_save_directory(self, root):
        """初始化数据保存目录（按时间戳创建）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(root, timestamp)
        for dir_name in ["images/front", "images/back", "images/left", "images/right", "csv"]:
            os.makedirs(os.path.join(save_root, dir_name), exist_ok=True)
        print(f"数据保存目录已创建：{save_root}")
        return save_root

    def _init_csv_writer(self):
        """初始化CSV写入器"""
        csv_path = os.path.join(self.save_root, "csv/global_vehicle_data.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp",
            "front_image", "back_image", "left_image", "right_image",
            "global_x", "global_y", "global_z",
            "velocity_x", "velocity_y", "velocity_z",
            "acceleration_x", "acceleration_y", "acceleration_z",
            "steer", "throttle", "brake", "speed_kmh"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        return csv_file, csv_writer

    def _process_image(self, image, direction):
        """摄像头图像回调函数：保存图像并记录路径"""
        with self.data_lock:
            # 转换Carla图像为RGB格式
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            # 生成图像文件名
            timestamp = f"{image.timestamp:.6f}".replace(".", "_")
            image_name = f"{timestamp}_{direction}.png"
            image_path = os.path.join(self.save_root, f"images/{direction}/{image_name}")
            # 保存图像
            cv2.imwrite(image_path, array)
            # 更新图像路径
            self.image_save_paths[f"{direction}_image"] = image_path

    def _get_vehicle_state(self):
        """获取车辆的全局状态和底盘控制信息"""
        if not self.vehicle or not self.vehicle.is_alive:
            return {}
        # 全局坐标
        transform = self.vehicle.get_transform()
        location = transform.location
        # 速度和加速度
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        # 底盘控制信息
        control = self.vehicle.get_control()
        # 速度转换（m/s → km/h）
        speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        # 组装数据
        self.vehicle_data = {
            "timestamp": time.time(),
            "global_x": location.x, "global_y": location.y, "global_z": location.z,
            "velocity_x": velocity.x, "velocity_y": velocity.y, "velocity_z": velocity.z,
            "acceleration_x": acceleration.x, "acceleration_y": acceleration.y, "acceleration_z": acceleration.z,
            "steer": control.steer, "throttle": control.throttle, "brake": control.brake,
            "speed_kmh": speed_kmh
        }

    def _save_vehicle_data(self):
        """保存车辆数据到CSV（关联图像路径）"""
        with self.data_lock:
            # 校验数据完整性
            if not self.vehicle_data or None in self.image_save_paths.values():
                return
            # 合并数据
            save_data = {**self.vehicle_data, **self.image_save_paths}
            # 写入CSV
            self.csv_writer.writerow(save_data)
            self.csv_file.flush()
            # 重置图像路径
            self.image_save_paths = {k: None for k in self.image_save_paths.keys()}
            print(f"成功保存数据：时间戳={save_data['timestamp']:.2f} | 车速={save_data['speed_kmh']:.1f} km/h")

    def _get_vehicle_dimensions(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return {}
        length = float(self.vehicle.attributes.get('length', 0.0))
        width = float(self.vehicle.attributes.get('width', 0.0))
        height = float(self.vehicle.attributes.get('height', 0.0))
        self.dimensions = {
            "length": length, "width": width, "height": height
        }


    def spawn_cameras(self):
        """挂载前后左右四个摄像头"""
        # 摄像头配置：方向、挂载位置、旋转角度
        self._get_vehicle_dimensions()
        bounding_box = self.vehicle.bounding_box
        # 包围盒的 extent 是半长/半宽/半高，因此需要乘以 2 得到实际尺寸
        length = bounding_box.extent.x * 2.0  # 长度（X轴）
        width = bounding_box.extent.y * 2.0   # 宽度（Y轴）
        height = bounding_box.extent.z * 2.0  # 高度（Z轴）
        print(length, width, height)
        front_distance = length / 2 + 0.5
        back_distance = length / 2 + 0.5
        camera_height = height * 1.2
        side_distance = width / 2 + 0.1
        camera_configs = [
            # ("front", carla.Transform(
            #     carla.Location(x=front_distance, z=camera_height),
            #     carla.Rotation(pitch=0)
            # )),
            ("front", carla.Transform(
                carla.Location(x=2.0, y=0.0, z=2.5),
                carla.Rotation(pitch=-10)
            )),
            ("back", carla.Transform(
                carla.Location(x=-back_distance, z=camera_height),
                carla.Rotation(pitch=0, yaw=180)
            )),
            ("left", carla.Transform(
                carla.Location(y=side_distance, z=camera_height),
                carla.Rotation(pitch=0, yaw=-90)
            )),
            ("right", carla.Transform(
                carla.Location(y=-side_distance, z=camera_height),
                carla.Rotation(pitch=0, yaw=90)
            ))
        ]
        # 创建摄像头蓝图
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.camera_height))
        camera_bp.set_attribute("fov", "110")
        camera_bp.set_attribute("sensor_tick", str(self.sensor_tick))
        camera_bp.set_attribute("motion_blur_intensity", "0.0")  # 禁用运动模糊
        camera_bp.set_attribute("motion_blur_max_distortion", "0.0")
        camera_bp.set_attribute("motion_blur_min_object_screen_size", "0.0")
        camera_bp.set_attribute("lens_circle_multiplier", "0.0")  # 禁用镜头效果
        camera_bp.set_attribute("lens_circle_falloff", "0.0")
        camera_bp.set_attribute("lens_k", "-1.0")
        camera_bp.set_attribute("lens_kcube", "0.0")
        camera_bp.set_attribute("lens_x_size", "0.0")
        camera_bp.set_attribute("lens_y_size", "0.0")
        # 生成摄像头并注册回调
        for direction, transform in camera_configs:
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
            camera.listen(lambda image, d=direction: self._process_image(image, d))
            self.cameras.append(camera)
            print(f"已挂载{direction}摄像头")

    def collect_data(self):
        """单次数据采集（建议在循环中调用）"""
        self._get_vehicle_state()
        self._save_vehicle_data()

    def stop_collect(self):
        """停止采集并清理资源"""
        print("开始清理采集资源...")
        # 销毁摄像头
        for camera in self.cameras:
            if camera.is_alive:
                camera.destroy()
        # 关闭CSV文件
        self.csv_file.close()
        print("资源清理完成，数据采集停止！")
