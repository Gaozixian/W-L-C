#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Carla Autonomous Vehicle Control Application

功能说明:
1. 连接到Carla仿真器
2. 生成指定类型的车辆
3. 使用Carla内置自动驾驶功能
4. 实时显示车辆状态和相机视角
5. 安全的资源清理机制

作者: MiniMax Agent
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
    - 支持动态调整相对位置
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

        # 摄像头相对位置（相对于车辆中心）
        # x: 前后位置（正前负后）
        # y: 左右位置（正左负右）
        # z: 高度位置
        self.camera_offset = {'x': 0.5, 'y': 0.0, 'z': 1.5}

    def set_position(self, x=0.5, y=0.0, z=1.5):
        """
        设置摄像头相对于车辆的位置

        Args:
            x: 前后偏移（米），正值为车前，负值为车后
            y: 左右偏移（米），正值为向左，负值为向右
            z: 高度偏移（米），正值为向上
        """
        self.camera_offset = {'x': x, 'y': y, 'z': z}
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
                carla.Rotation(pitch=-5, yaw=0, roll=0)  # 稍微向下倾斜5度
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
    - 显示车辆速度信息
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
        self.window.geometry(f"{self.image_width}x{self.image_height + 60}")
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

        # 创建状态标签
        self.status_label = tk.Label(
            self.window,
            text="Waiting for camera...",
            font=('Arial', 12),
            fg='gray'
        )
        self.status_label.pack(pady=5)

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

                # 获取并显示速度
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * math.sqrt(
                    velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                )
                self.status_label.config(
                    text=f"Speed: {speed:.1f} km/h | Camera Active",
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
            self.window.after_cancel(self.update_id)
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
    - 生成车辆
    - 启动/停止自动驾驶
    - 显示车辆状态
    """

    def __init__(self, root):
        """
        初始化应用程序

        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("Carla Autonomous Vehicle Controller")
        self.root.geometry("680x680")  # 增加高度以显示摄像头位置控件
        self.root.resizable(False, False)

        # Carla相关变量
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None

        # 相机相关变量
        self.camera_sensor = None
        self.pygame_display = None
        self.pygame_thread = None

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

        # 2. 车辆生成区域
        frame_spawn = ttk.LabelFrame(self.root, text="Vehicle Spawn", padding=10)
        frame_spawn.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_spawn, text="Blueprint Filter:").pack(side="left", padx=5)

        self.entry_blueprint = ttk.Entry(frame_spawn, width=25)
        self.entry_blueprint.insert(0, "vehicle.tesla.model3")
        self.entry_blueprint.pack(side="left", padx=5)

        self.btn_spawn = ttk.Button(
            frame_spawn,
            text="Spawn Vehicle",
            command=self._spawn_vehicle,
            state="disabled"
        )
        self.btn_spawn.pack(side="left", padx=5)

        # 3. 自动驾驶控制区域
        frame_control = ttk.LabelFrame(self.root, text="Autopilot Control", padding=10)
        frame_control.pack(fill="x", padx=15, pady=5)

        # 目标速度设置
        ttk.Label(frame_control, text="Target Speed (km/h):").pack(side="left", padx=5)
        # ttk.Label(frame_control, text="Target Task:").pack(side="left", padx=10)

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

        # 3.5 相机视角控制区域
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

        # 摄像头位置调整区域
        frame_camera_pos = ttk.LabelFrame(frame_camera, text="Camera Position", padding=5)
        frame_camera_pos.pack(fill="x", pady=5)

        # X位置（前后）
        ttk.Label(frame_camera_pos, text="X (Front/Back):").grid(row=0, column=0, padx=2, sticky="e")
        self.spin_cam_x = ttk.Spinbox(
            frame_camera_pos,
            from_=-2.0,
            to=5.0,
            increment=0.1,
            width=6
        )
        self.spin_cam_x.set(0.5)
        self.spin_cam_x.grid(row=0, column=1, padx=2)

        # Y位置（左右）
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

        # Z位置（高度）
        ttk.Label(frame_camera_pos, text="Z (Height):").grid(row=1, column=0, padx=2, sticky="e")
        self.spin_cam_z = ttk.Spinbox(
            frame_camera_pos,
            from_=0.5,
            to=5.0,
            increment=0.1,
            width=6
        )
        self.spin_cam_z.set(1.5)
        self.spin_cam_z.grid(row=1, column=1, padx=2)

        # 应用位置按钮
        self.btn_cam_apply = ttk.Button(
            frame_camera,
            text="APPLY POSITION",
            command=self._apply_camera_position,
            state="disabled"
        )
        self.btn_cam_apply.pack(fill="x", pady=5)

        # 4. 状态显示区域
        frame_status = ttk.LabelFrame(self.root, text="Status", padding=10)
        frame_status.pack(fill="x", padx=10, pady=5)

        self.lbl_speed = ttk.Label(
            frame_status,
            text="Speed: 0.0 km/h",
            font=('Microsoft YaHei UI', 14, 'bold')
        )
        self.lbl_speed.pack(pady=5)

        self.lbl_mode = ttk.Label(
            frame_status,
            text="Mode: Manual",
            font=('Microsoft YaHei UI', 12)
        )
        self.lbl_mode.pack(pady=5)

        # 5. 退出按钮
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
            self.btn_spawn.config(state="normal")

            messagebox.showinfo("Success", "Successfully connected to Carla server!")

        except Exception as e:
            messagebox.showerror(
                "Connection Error",
                f"Failed to connect to Carla server:\n{str(e)}\n\n"
                "Please make sure the Carla server is running on localhost:2000"
            )

    def _spawn_vehicle(self):
        """生成车辆"""
        # 如果已存在车辆，先销毁
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
                self.vehicle = None
            except:
                pass

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 获取用户指定的车辆类型
            filter_text = self.entry_blueprint.get().strip()
            blueprints = blueprint_library.filter(filter_text)

            if not blueprints:
                messagebox.showwarning(
                    "Warning",
                    f"No vehicle blueprints found matching: '{filter_text}'\n\n"
                    "Try using 'vehicle.*' or specific vehicle names like:\n"
                    "- vehicle.tesla.model3\n"
                    "- vehicle.nissan.patrol\n"
                    "- vehicle.audi.etron"
                )
                return

            # 选择第一个匹配的蓝图
            blueprint = blueprints[0]

            # 获取随机生成点
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

            # 生成车辆
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

            # 移动观察者视角到车辆位置
            self._move_spectator_to_vehicle()

            # 初始化相机传感器
            self.camera_sensor = CameraSensor(self.vehicle, self.world)

            # 更新界面状态
            self.btn_autopilot.config(state="normal")
            self.btn_camera.config(state="normal")
            self.btn_cam_apply.config(state="normal")
            messagebox.showinfo(
                "Success",
                f"Vehicle spawned successfully!\n"
                f"Blueprint: {blueprint.id}"
            )

        except Exception as e:
            messagebox.showerror("Spawn Error", f"Failed to spawn vehicle:\n{str(e)}")

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
        if self.pygame_display is not None and self.pygame_display.is_running():
            # 关闭相机视图
            self._cleanup_camera_view()

        else:
            # 检查是否有车辆
            if self.vehicle is None:
                messagebox.showwarning("Warning", "No vehicle available!")
                return

            # 启动相机传感器
            if self.camera_sensor is not None:
                self.camera_sensor.start()

                # 创建tkinter相机窗口
                self.pygame_display = TkCameraWindow(
                    self.camera_sensor,
                    self.vehicle,
                    update_callback=self._on_speed_update
                )

                # 显示窗口
                self.pygame_display.show()

                # 更新界面状态
                self.btn_camera.config(text="CLOSE CAMERA VIEW")
                self.lbl_camera_status.config(text="Status: Camera Active", foreground="green")

    def _on_speed_update(self, speed):
        """速度更新回调（可选用于更新主界面）"""
        pass  # 主界面已经在_control_loop中更新

    def _apply_camera_position(self):
        """应用摄像头位置设置"""
        try:
            # 获取位置值
            x = float(self.spin_cam_x.get())
            y = float(self.spin_cam_y.get())
            z = float(self.spin_cam_z.get())

            # 如果相机已启动，先停止
            camera_was_active = (
                    self.camera_sensor is not None and
                    self.camera_sensor.is_active()
            )

            if camera_was_active:
                self._cleanup_camera_view()

            # 设置新位置
            if self.camera_sensor is not None:
                self.camera_sensor.set_position(x, y, z)

            # 重新启动相机（如果之前是开启状态）
            if camera_was_active:
                self.camera_sensor.start()
                self.pygame_display = TkCameraWindow(
                    self.camera_sensor,
                    self.vehicle,
                    update_callback=self._on_speed_update
                )
                self.pygame_display.show()
                self.btn_camera.config(text="CLOSE CAMERA VIEW")
                self.lbl_camera_status.config(text="Status: Camera Active", foreground="green")

            messagebox.showinfo(
                "Success",
                f"Camera position applied:\n"
                f"X: {x}m (Front/Back)\n"
                f"Y: {y}m (Left/Right)\n"
                f"Z: {z}m (Height)"
            )

        except ValueError:
            messagebox.showerror("Error", "Invalid position values!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply position:\n{str(e)}")

    def _cleanup_camera_view(self):
        """清理相机视图资源"""
        if self.pygame_display is not None:
            self.pygame_display.close()
            self.pygame_display = None

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

        用于更新UI显示的车辆速度信息
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

                    # 更新UI显示
                    self.root.after(
                        0,
                        lambda s=speed: self.lbl_speed.config(
                            text=f"Speed: {s:.1f} km/h"
                        )
                    )

            except Exception as e:
                print(f"Control loop error: {e}")

            # 更新频率：20Hz
            time.sleep(0.05)

    def _on_close(self):
        """窗口关闭处理"""
        self.running = False

        # 先清理相机资源
        if self.pygame_display is not None:
            self.pygame_display.stop()
            self.pygame_display = None

        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor = None

        # 销毁车辆
        if self.vehicle:
            try:
                print("Destroying vehicle...")
                self.vehicle.destroy()
            except:
                pass

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