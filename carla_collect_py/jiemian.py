#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Carla Autonomous Vehicle Control Application

功能说明:
1. 连接到Carla仿真器
2. 查找并控制已生成的指定类型车辆
3. 使用Carla内置自动驾驶功能
4. 实时显示车辆状态（速度、转角）和相机视角
5. 相机支持位置和俯仰角调节
6. 安全的资源清理机制

作者: Modified Version
"""

import carla
import random
import time
import threading
import math
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np


# ============================================================================
# 相机传感器类 - 用于获取车辆视角图像
# ============================================================================

class CameraSensor:
    """
    相机传感器类，负责在车辆上安装相机并捕获图像

    功能:
    - 在车辆上安装RGB相机传感器
    - 回调方式实时获取图像数据
    - 图像数据转换为PIL图像供tkinter显示
    - 支持动态调整相对位置和俯仰角
    """

    def __init__(self, vehicle, world, width=640, height=480):
        """
        初始化相机传感器

        Args:
            vehicle: Carla车辆对象
            world: Carla世界对象
            width: 图像宽度（默认640）
            height: 图像高度（默认480）
        """
        self.vehicle = vehicle
        self.world = world
        self.width = width
        self.height = height
        self.sensor = None
        self.image = None
        self.pil_image = None
        self.lock = threading.Lock()
        self.running = False

        # 摄像头相对位置和角度（相对于车辆中心）
        self.camera_offset = {'x': 2, 'y': 0.0, 'z': 2.5}
        self.camera_rotation = {'pitch': -10, 'yaw': 0, 'roll': 0}  # 默认稍微向下倾斜5度

    def set_position(self, x=2, y=0.0, z=2.5, pitch=-10, yaw=0, roll=0):
        """
        设置摄像头相对于车辆的位置和角度

        Args:
            x: 前后偏移（米），正值为车前，负值为车后
            y: 左右偏移（米），正值为向左，负值为向右
            z: 高度偏移（米），正值为向上
            pitch: 俯仰角（度），正值为向上看，负值为向下看
            yaw: 偏航角（度），正值为向左转，负值为向右转
            roll: 横滚角（度）
        """
        self.camera_offset = {'x': x, 'y': y, 'z': z}
        self.camera_rotation = {'pitch': pitch, 'yaw': yaw, 'roll': roll}

        # 如果传感器已启动，重新启动以应用新位置
        if self.sensor is not None and self.running:
            self.stop()
            self.start()

    def start(self):
        """启动相机传感器"""
        if self.sensor is not None:
            return

        try:
            # 获取蓝图库中的RGB相机
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')

            # 设置相机属性
            blueprint.set_attribute('image_size_x', str(self.width))
            blueprint.set_attribute('image_size_y', str(self.height))
            blueprint.set_attribute('fov', '110')

            # 在车辆上安装相机（相对于车辆中心）
            sensor_transform = carla.Transform(
                carla.Location(
                    x=self.camera_offset['x'],
                    y=self.camera_offset['y'],
                    z=self.camera_offset['z']
                ),
                carla.Rotation(
                    pitch=self.camera_rotation['pitch'],
                    yaw=self.camera_rotation['yaw'],
                    roll=self.camera_rotation['roll']
                )
            )

            # 创建并启动传感器（attach_to确保跟随车辆）
            self.sensor = self.world.spawn_actor(blueprint, sensor_transform, attach_to=self.vehicle)

            # 设置图像回调
            self.sensor.listen(self._on_image)

            self.running = True

        except Exception as e:
            print(f"Failed to start camera sensor: {e}")

    def _on_image(self, image):
        """图像回调函数"""
        if not self.running:
            return

        try:
            # 将CARLA图像转换为numpy数组
            # CARLA图像格式：BGRA
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))

            # 转换为RGB格式（去掉alpha通道并交换B和R）
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # BGR to RGB

            with self.lock:
                self.image = array.copy()
                # 同时转换为PIL图像供tkinter显示
                self.pil_image = Image.fromarray(array)

        except Exception as e:
            print(f"Image processing error: {e}")

    def get_image(self):
        """获取当前numpy图像"""
        with self.lock:
            return self.image.copy() if self.image is not None else None

    def get_pil_image(self):
        """获取当前PIL图像"""
        with self.lock:
            return self.pil_image.copy() if self.pil_image is not None else None

    def stop(self):
        """停止并销毁传感器"""
        self.running = False

        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
                self.sensor = None
            except:
                pass

        self.image = None
        self.pil_image = None

    def is_active(self):
        """检查传感器是否正在运行"""
        return self.sensor is not None and self.running


# ============================================================================
# Tkinter相机显示窗口类 - 用于显示车辆视角
# ============================================================================

class TkCameraWindow:
    """
    Tkinter相机显示窗口类，负责实时显示车辆相机视角
    功能:
    - 创建独立的tkinter窗口显示相机图像
    - 使用Canvas和PIL图像实现稳定显示
    - 显示车辆速度和转角信息
    - 窗口关闭时自动清理资源
    """

    def __init__(self, camera_sensor, vehicle, update_callback=None):
        """
        初始化相机显示窗口

        Args:
            camera_sensor: CameraSensor对象
            vehicle: Carla车辆对象
            update_callback: 速度更新回调函数（可选）
        """
        self.camera_sensor = camera_sensor
        self.vehicle = vehicle
        self.update_callback = update_callback
        self.window = None
        self.canvas = None
        self.photo_image = None
        self.running = False
        self.update_id = None

        # 图像尺寸
        self.image_width = camera_sensor.width
        self.image_height = camera_sensor.height

    def show(self):
        """显示相机窗口"""
        if self.window is not None:
            return

        # 创建新窗口
        self.window = tk.Toplevel()
        self.window.title("Vehicle Camera View")
        self.window.geometry(f"{self.image_width}x{self.image_height + 80}")
        self.window.resizable(False, False)

        # 窗口关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        # 创建Canvas用于显示图像
        self.canvas = tk.Canvas(
            self.window,
            width=self.image_width,
            height=self.image_height,
            bg='black'
        )
        self.canvas.pack(padx=10, pady=5)

        # 创建状态标签框架
        frame_status = tk.Frame(self.window)
        frame_status.pack(pady=5)

        # 速度显示标签
        self.speed_label = tk.Label(
            frame_status,
            text="Speed: 0.0 km/h",
            font=('Arial', 12, 'bold'),
            fg='blue'
        )
        self.speed_label.pack(side='left', padx=10)

        # 转角显示标签
        self.steer_label = tk.Label(
            frame_status,
            text="Steering: 0.0°",
            font=('Arial', 12, 'bold'),
            fg='green'
        )
        self.steer_label.pack(side='left', padx=10)

        # 相机状态标签
        self.status_label = tk.Label(
            self.window,
            text="Waiting for camera...",
            font=('Arial', 10),
            fg='gray'
        )
        self.status_label.pack()

        self.running = True

        # 启动更新循环
        self._update_display()

    def _update_display(self):
        """更新显示循环"""
        if not self.running or self.window is None:
            return

        try:
            # 获取PIL图像
            pil_image = self.camera_sensor.get_pil_image()

            if pil_image is not None:
                # 转换为PhotoImage
                self.photo_image = ImageTk.PhotoImage(pil_image)

                # 在Canvas上显示图像
                self.canvas.create_image(
                    0, 0,
                    anchor=tk.NW,
                    image=self.photo_image
                )

                # 获取并显示速度和转角
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * math.sqrt(
                    velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                )
                self.speed_label.config(
                    text=f"Speed: {speed:.1f} km/h"
                )

                # 获取并显示转角
                control = self.vehicle.get_control()
                steer_angle = control.steer * 100  # 转换为角度显示（-100° 到 100° 表示左右）
                self.steer_label.config(
                    text=f"Steering: {steer_angle:+.1f}°"
                )

                self.status_label.config(
                    text="Camera Active",
                    fg='green'
                )

                # 如果有回调函数，调用它
                if self.update_callback:
                    self.update_callback(speed)

            else:
                self.status_label.config(
                    text="Waiting for camera...",
                    fg='gray'
                )

        except Exception as e:
            print(f"Display update error: {e}")

        # 继续更新（30fps）
        self.update_id = self.window.after(33, self._update_display)

    def close(self):
        """关闭窗口"""
        self.running = False

        if self.update_id is not None:
            try:
                self.window.after_cancel(self.update_id)
            except:
                pass
            self.update_id = None

        if self.window is not None:
            try:
                self.window.destroy()
            except:
                pass
            self.window = None
            self.canvas = None
            self.photo_image = None

    def is_running(self):
        """检查窗口是否正在运行"""
        return self.running and self.window is not None


# ============================================================================
# Carla自动驾驶应用主类
# ============================================================================

class CarlaAutoPilotApp:
    """
    Carla自动驾驶控制应用程序主类

    提供图形界面用于：
    - 连接Carla服务器
    - 查找已生成的指定类型车辆
    - 启动/停止自动驾驶
    - 实时显示车辆状态（速度、转角）
    """

    def __init__(self, root):
        """
        初始化应用程序

        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("Carla Autonomous Vehicle Controller")
        self.root.geometry("720x720")  # 增加高度以显示更多控件
        self.root.resizable(False, False)

        # Carla相关变量
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.available_vehicles = []  # 存储可用车辆列表

        # 相机相关变量
        self.camera_sensor = None
        self.camera_window = None

        # 控制状态
        self.autopilot_enabled = False
        self.running = True

        # 创建界面
        self._create_widgets()

        # 启动控制循环线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def _create_widgets(self):
        """创建GUI组件"""

        # 样式配置
        style = ttk.Style()
        style.configure('TLabel', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', font=('Microsoft YaHei UI', 10))

        # 1. 连接设置区域
        frame_connection = ttk.LabelFrame(self.root, text="Connection", padding=10)
        frame_connection.pack(fill="x", padx=10, pady=5)

        self.btn_connect = ttk.Button(
            frame_connection,
            text="Connect to Carla Server",
            command=self._connect_to_carla
        )
        self.btn_connect.pack(fill="x", pady=5)

        self.lbl_connection_status = ttk.Label(
            frame_connection,
            text="Status: Disconnected",
            foreground="red"
        )
        self.lbl_connection_status.pack()

        # 2. 车辆选择区域（改为查找已生成车辆）
        frame_vehicle = ttk.LabelFrame(self.root, text="Vehicle Selection", padding=10)
        frame_vehicle.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_vehicle, text="Vehicle Type Filter:").pack(anchor="w")

        self.entry_blueprint = ttk.Entry(frame_vehicle, width=40)
        self.entry_blueprint.insert(0, "vehicle.tesla.cybertruck")
        self.entry_blueprint.pack(fill="x", pady=5)

        self.btn_find_vehicles = ttk.Button(
            frame_vehicle,
            text="Find Vehicles",
            command=self._find_vehicles,
            state="disabled"
        )
        self.btn_find_vehicles.pack(side="left", padx=5)

        self.btn_select_vehicle = ttk.Button(
            frame_vehicle,
            text="Select Vehicle",
            command=self._select_vehicle,
            state="disabled"
        )
        self.btn_select_vehicle.pack(side="left", padx=5)

        # 车辆列表显示
        self.vehicle_listbox = tk.Listbox(frame_vehicle, height=4)
        self.vehicle_listbox.pack(fill="x", pady=5)

        # 3. 自动驾驶控制区域
        frame_control = ttk.LabelFrame(self.root, text="Autopilot Control", padding=10)
        frame_control.pack(fill="x", padx=10, pady=5)

        # 目标速度设置
        ttk.Label(frame_control, text="Target Speed (km/h):").pack(side="left", padx=5)

        self.spin_speed = ttk.Spinbox(
            frame_control,
            from_=10,
            to=200,
            increment=5,
            width=8
        )
        self.spin_speed.set(50)
        self.spin_speed.pack(side="left", padx=5)

        # 自动驾驶开关按钮
        self.btn_autopilot = ttk.Button(
            frame_control,
            text="START AUTOPILOT",
            command=self._toggle_autopilot,
            state="disabled"
        )
        self.btn_autopilot.pack(fill="x", pady=10)

        # 4. 相机视角控制区域（增加俯仰角调节）
        frame_camera = ttk.LabelFrame(self.root, text="Camera View", padding=10)
        frame_camera.pack(fill="x", padx=10, pady=5)

        self.btn_camera = ttk.Button(
            frame_camera,
            text="OPEN CAMERA VIEW",
            command=self._toggle_camera_view,
            state="disabled"
        )
        self.btn_camera.pack(fill="x", pady=5)

        self.lbl_camera_status = ttk.Label(
            frame_camera,
            text="Status: Camera Closed",
            foreground="gray"
        )
        self.lbl_camera_status.pack()

        # 摄像头位置和角度调整区域
        frame_camera_pos = ttk.LabelFrame(frame_camera, text="Camera Position & Angle", padding=5)
        frame_camera_pos.pack(fill="x", pady=5)

        # 第一行：X, Y, Z位置
        ttk.Label(frame_camera_pos, text="X (Front/Back):").grid(row=0, column=0, padx=2, sticky="e")
        self.spin_cam_x = ttk.Spinbox(
            frame_camera_pos,
            from_=-2.0,
            to=5.0,
            increment=0.1,
            width=6
        )
        self.spin_cam_x.set(2)
        self.spin_cam_x.grid(row=0, column=1, padx=2)

        ttk.Label(frame_camera_pos, text="Y (Left/Right):").grid(row=0, column=2, padx=2, sticky="e")
        self.spin_cam_y = ttk.Spinbox(
            frame_camera_pos,
            from_=-2.0,
            to=2.0,
            increment=0.1,
            width=6
        )
        self.spin_cam_y.set(0.0)
        self.spin_cam_y.grid(row=0, column=3, padx=2)

        ttk.Label(frame_camera_pos, text="Z (Height):").grid(row=0, column=4, padx=2, sticky="e")
        self.spin_cam_z = ttk.Spinbox(
            frame_camera_pos,
            from_=0.5,
            to=5.0,
            increment=0.1,
            width=6
        )
        self.spin_cam_z.set(2.5)
        self.spin_cam_z.grid(row=0, column=5, padx=2)

        # 第二行：俯仰角、偏航角、横滚角
        ttk.Label(frame_camera_pos, text="Pitch (俯仰):").grid(row=1, column=0, padx=2, sticky="e", pady=5)
        self.spin_cam_pitch = ttk.Spinbox(
            frame_camera_pos,
            from_=-90.0,
            to=90.0,
            increment=5.0,
            width=6
        )
        self.spin_cam_pitch.set(-10)  # 默认向下10度
        self.spin_cam_pitch.grid(row=1, column=1, padx=2, pady=5)

        ttk.Label(frame_camera_pos, text="Yaw (偏航):").grid(row=1, column=2, padx=2, sticky="e", pady=5)
        self.spin_cam_yaw = ttk.Spinbox(
            frame_camera_pos,
            from_=-180.0,
            to=180.0,
            increment=5.0,
            width=6
        )
        self.spin_cam_yaw.set(0)
        self.spin_cam_yaw.grid(row=1, column=3, padx=2, pady=5)

        ttk.Label(frame_camera_pos, text="Roll (横滚):").grid(row=1, column=4, padx=2, sticky="e", pady=5)
        self.spin_cam_roll = ttk.Spinbox(
            frame_camera_pos,
            from_=-180.0,
            to=180.0,
            increment=5.0,
            width=6
        )
        self.spin_cam_roll.set(0)
        self.spin_cam_roll.grid(row=1, column=5, padx=2, pady=5)

        # 应用位置按钮
        self.btn_cam_apply = ttk.Button(
            frame_camera,
            text="APPLY CAMERA SETTINGS",
            command=self._apply_camera_settings,
            state="disabled"
        )
        self.btn_cam_apply.pack(fill="x", pady=5)

        # 5. 状态显示区域（增加转角显示）
        frame_status = ttk.LabelFrame(self.root, text="Status", padding=10)
        frame_status.pack(fill="x", padx=10, pady=5)

        # 速度显示
        self.lbl_speed = ttk.Label(
            frame_status,
            text="Speed: 0.0 km/h",
            font=('Microsoft YaHei UI', 14, 'bold')
        )
        self.lbl_speed.pack(pady=2)

        # 转角显示
        self.lbl_steering = ttk.Label(
            frame_status,
            text="Steering: 0.0°",
            font=('Microsoft YaHei UI', 14, 'bold'),
            foreground="green"
        )
        self.lbl_steering.pack(pady=2)

        self.lbl_mode = ttk.Label(
            frame_status,
            text="Mode: Manual",
            font=('Microsoft YaHei UI', 12)
        )
        self.lbl_mode.pack(pady=2)

        # 6. 退出按钮
        ttk.Button(
            self.root,
            text="Exit & Cleanup",
            command=self._on_close
        ).pack(pady=10)

        # 窗口关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _connect_to_carla(self):
        """连接到Carla服务器"""
        try:
            # 创建Carla客户端
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)

            # 获取世界和地图
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            # 更新界面状态
            self.lbl_connection_status.config(
                text="Status: Connected",
                foreground="green"
            )
            self.btn_connect.config(state="disabled")
            self.btn_find_vehicles.config(state="normal")

            messagebox.showinfo("Success", "Successfully connected to Carla server!")

        except Exception as e:
            messagebox.showerror(
                "Connection Error",
                f"Failed to connect to Carla server:\n{str(e)}\n\n"
                "Please make sure the Carla server is running on localhost:2000"
            )

    def _find_vehicles(self):
        """查找已生成的指定类型车辆"""
        try:
            # 获取所有车辆
            all_vehicles = self.world.get_actors().filter('vehicle.*')

            # 根据过滤条件筛选
            filter_text = self.entry_blueprint.get().strip()

            # 清空列表
            self.vehicle_listbox.delete(0, tk.END)
            self.available_vehicles = []

            for vehicle in all_vehicles:
                vehicle_type = vehicle.type_id
                # 如果过滤条件为空或者匹配，则添加到列表
                if not filter_text or filter_text in vehicle_type:
                    self.available_vehicles.append(vehicle)
                    # 获取车辆位置信息
                    location = vehicle.get_location()
                    self.vehicle_listbox.insert(
                        tk.END,
                        f"{vehicle_type} (ID: {vehicle.id}, Pos: ({location.x:.1f}, {location.y:.1f}))"
                    )

            if self.available_vehicles:
                self.btn_select_vehicle.config(state="normal")
                self.vehicle_listbox.selection_set(0)
                messagebox.showinfo(
                    "Success",
                    f"Found {len(self.available_vehicles)} vehicle(s) matching '{filter_text}'"
                )
            else:
                self.btn_select_vehicle.config(state="disabled")
                messagebox.showwarning(
                    "Warning",
                    f"No vehicles found matching '{filter_text}'"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to find vehicles:\n{str(e)}")

    def _select_vehicle(self):
        """选择选中的车辆"""
        selection = self.vehicle_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a vehicle from the list")
            return

        # 如果已存在车辆，先清理相关资源
        if self.vehicle is not None:
            self._cleanup_camera_view()
            if self.camera_sensor is not None:
                self.camera_sensor.stop()
                self.camera_sensor = None

        # 获取选中的车辆
        index = selection[0]
        self.vehicle = self.available_vehicles[index]

        try:
            # 移动观察者视角到车辆位置
            self._move_spectator_to_vehicle()

            # 初始化相机传感器
            self.camera_sensor = CameraSensor(self.vehicle, self.world)

            # 更新界面状态
            self.btn_autopilot.config(state="normal")
            self.btn_camera.config(state="normal")
            self.btn_cam_apply.config(state="normal")

            # 更新车辆类型显示
            self.entry_blueprint.delete(0, tk.END)
            self.entry_blueprint.insert(0, self.vehicle.type_id)

            messagebox.showinfo(
                "Success",
                f"Vehicle selected successfully!\n"
                f"Type: {self.vehicle.type_id}\n"
                f"ID: {self.vehicle.id}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to select vehicle:\n{str(e)}")

    def _move_spectator_to_vehicle(self):
        """移动观察者视角到车辆位置"""
        try:
            spectator = self.world.get_spectator()
            transform = self.vehicle.get_transform()

            # 设置观察者在车辆后上方
            spectator_transform = carla.Transform(
                transform.location + carla.Location(z=15),
                carla.Rotation(pitch=-60, yaw=transform.rotation.yaw)
            )

            spectator.set_transform(spectator_transform)

        except Exception as e:
            print(f"Warning: Failed to move spectator: {e}")

    def _toggle_camera_view(self):
        """切换相机视角显示"""
        if self.camera_window is not None and self.camera_window.is_running():
            # 关闭相机视图
            self._cleanup_camera_view()

        else:
            # 检查是否有车辆
            if self.vehicle is None:
                messagebox.showwarning("Warning", "No vehicle available!")
                return

            # 检查相机传感器
            if self.camera_sensor is None:
                self.camera_sensor = CameraSensor(self.vehicle, self.world)

            # 启动相机传感器
            self.camera_sensor.start()

            # 创建tkinter相机窗口
            self.camera_window = TkCameraWindow(
                self.camera_sensor,
                self.vehicle,
                update_callback=self._on_speed_update
            )

            # 显示窗口
            self.camera_window.show()

            # 更新界面状态
            self.btn_camera.config(text="CLOSE CAMERA VIEW")
            self.lbl_camera_status.config(text="Status: Camera Active", foreground="green")

    def _on_speed_update(self, speed):
        """速度更新回调（可选用于更新主界面）"""
        pass  # 主界面已经在_control_loop中更新

    def _apply_camera_settings(self):
        """应用摄像头位置和角度设置"""
        try:
            # 获取位置和角度值
            x = float(self.spin_cam_x.get())
            y = float(self.spin_cam_y.get())
            z = float(self.spin_cam_z.get())
            pitch = float(self.spin_cam_pitch.get())
            yaw = float(self.spin_cam_yaw.get())
            roll = float(self.spin_cam_roll.get())

            # 检查相机传感器是否存在
            if self.camera_sensor is None:
                if self.vehicle is None:
                    messagebox.showerror("Error", "No vehicle selected!")
                    return
                self.camera_sensor = CameraSensor(self.vehicle, self.world)

            # 记录相机是否正在运行
            camera_was_active = self.camera_sensor.is_active()

            # 如果相机已启动，先关闭窗口
            if camera_was_active and self.camera_window is not None:
                self._cleanup_camera_view()

            # 设置新位置和角度
            self.camera_sensor.set_position(x, y, z, pitch, yaw, roll)

            # 重新启动相机（如果之前是开启状态）
            if camera_was_active:
                self.camera_sensor.start()
                self.camera_window = TkCameraWindow(
                    self.camera_sensor,
                    self.vehicle,
                    update_callback=self._on_speed_update
                )
                self.camera_window.show()
                self.btn_camera.config(text="CLOSE CAMERA VIEW")
                self.lbl_camera_status.config(text="Status: Camera Active", foreground="green")

            messagebox.showinfo(
                "Success",
                f"Camera settings applied:\n"
                f"Position: X={x}m, Y={y}m, Z={z}m\n"
                f"Rotation: Pitch={pitch}°, Yaw={yaw}°, Roll={roll}°"
            )

        except ValueError:
            messagebox.showerror("Error", "Invalid input values!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings:\n{str(e)}")

    def _cleanup_camera_view(self):
        """清理相机视图资源"""
        if self.camera_window is not None:
            self.camera_window.close()
            self.camera_window = None

        if self.camera_sensor is not None:
            self.camera_sensor.stop()

        self.btn_camera.config(text="OPEN CAMERA VIEW")
        self.lbl_camera_status.config(text="Status: Camera Closed", foreground="gray")

    def _toggle_autopilot(self):
        """切换自动驾驶模式 - 使用Carla内置自动驾驶"""
        if not self.vehicle:
            messagebox.showwarning("Warning", "No vehicle available!")
            return

        self.autopilot_enabled = not self.autopilot_enabled

        if self.autopilot_enabled:
            # 获取目标速度（作为最大速度限制）
            target_speed = float(self.spin_speed.get())

            # 设置车辆最大速度（m/s，Carla内部使用m/s）
            # 将km/h转换为m/s
            max_speed_mps = target_speed / 3.6
            self.vehicle.set_max_speed(max_speed_mps)

            # 启用Carla内置自动驾驶
            self.vehicle.set_autopilot(True)

            # 更新按钮状态
            self.btn_autopilot.config(text="STOP AUTOPILOT")
            self.lbl_mode.config(text=f"Mode: Autopilot (Max {target_speed} km/h)", foreground="green")

        else:
            # 禁用自动驾驶
            self.vehicle.set_autopilot(False)

            # 施加刹车使车辆停止
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0)
            )

            # 更新按钮状态
            self.btn_autopilot.config(text="START AUTOPILOT")
            self.lbl_mode.config(text="Mode: Manual", foreground="black")

    def _control_loop(self):
        """
        状态更新循环（后台线程运行）

        用于更新UI显示的车辆速度和转角信息
        Carla内置自动驾驶会自动处理车辆控制
        """
        while self.running:
            try:
                if self.vehicle and self.vehicle.is_alive:
                    # 获取车辆速度
                    velocity = self.vehicle.get_velocity()
                    speed = 3.6 * math.sqrt(
                        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                    )

                    # 获取车辆转角
                    control = self.vehicle.get_control()
                    steer_angle = control.steer * 100  # 转换为角度显示

                    # 更新UI显示
                    self.root.after(
                        0,
                        lambda s=speed, st=steer_angle: [
                            self.lbl_speed.config(text=f"Speed: {s:.1f} km/h"),
                            self.lbl_steering.config(text=f"Steering: {st:+.1f}°")
                        ]
                    )

            except Exception as e:
                print(f"Control loop error: {e}")

            # 更新频率：20Hz
            time.sleep(0.05)

    def _on_close(self):
        """窗口关闭处理"""
        self.running = False

        # 先清理相机资源
        if self.camera_window is not None:
            self.camera_window.close()
            self.camera_window = None

        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor = None

        # 注意：不销毁车辆，因为车辆是已存在的
        # 只清理我们自己创建的资源

        # 关闭窗口
        self.root.destroy()


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主函数"""
    # 检查Carla依赖
    try:
        import carla
    except ImportError:
        print("Error: Carla library not found!")
        print("Please install the Carla client library:")
        print("  pip install carla")
        return

    # 创建Tkinter根窗口
    root = tk.Tk()

    # 设置字体（支持中文）
    try:
        root.tk.call('encoding', 'system')
    except:
        pass

    # 创建应用程序
    app = CarlaAutoPilotApp(root)

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()
