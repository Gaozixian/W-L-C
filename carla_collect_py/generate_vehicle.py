import carla
import random
import time

# 1. 连接Carla服务器，设置超时时间
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
# 获取地图的所有生成点（车辆/行人的预设生成位置）
spawn_points = world.get_map().get_spawn_points()

# 2. 初始化TrafficManager（用于车辆自动驾驶）
tm = client.get_trafficmanager(8000)
# 设置全局交通管理器参数：忽略交通灯、随机车道变更
tm.set_global_distance_to_leading_vehicle(2.0)  # 跟车距离
tm.set_random_device_seed(1)  # 随机种子，保证可复现（可选）

# 3. 定义生成随机车辆的函数
def spawn_random_vehicles(num_vehicles):
    """
    生成指定数量的随机车辆
    :param num_vehicles: 车辆数量
    :return: 生成的车辆actor列表（用于后续销毁）
    """
    vehicle_list = []
    # 过滤蓝图库中的所有车辆
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    # 排除部分特殊车辆（可选，如自行车/摩托车，只保留汽车）
    vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

    for _ in range(num_vehicles):
        try:
            # 随机选择车辆蓝图
            random_bp = random.choice(vehicle_blueprints)
            # 随机选择生成位置（从预设生成点中选）
            if spawn_points:
                random_spawn_point = random.choice(spawn_points)
            else:
                # 若没有预设生成点，自定义随机位置（需根据地图范围调整）
                random_spawn_point = carla.Transform(
                    carla.Location(x=random.uniform(-100, 100), y=random.uniform(-100, 100), z=0.2),
                    carla.Rotation(yaw=random.uniform(0, 360))
                )

            # 生成车辆
            vehicle = world.spawn_actor(random_bp, random_spawn_point)
            vehicle_list.append(vehicle)

            # ========== 设置车辆随机运动状态 ==========
            # 方式1：手动设置随机初始控制（瞬时状态，无持续行为）
            # control = carla.VehicleControl()
            # control.throttle = random.uniform(0.0, 1.0)  # 随机油门
            # control.steer = random.uniform(-1.0, 1.0)    # 随机转向
            # control.brake = random.uniform(0.0, 0.0)     # 刹车设为0
            # vehicle.apply_control(control)

            # 方式2：通过TrafficManager设置自动驾驶（持续随机行为，推荐）
            vehicle.set_autopilot(True, tm.get_port())
            # 设置单辆车的随机参数
            tm.auto_lane_change(vehicle, random.choice([True, False]))  # 随机是否允许变道
            tm.set_desired_speed(vehicle, random.uniform(10, 50))  # 随机期望速度（km/h）
            tm.set_lane_change_probability(vehicle, random.uniform(0.0, 1.0))  # 随机变道概率

            print(f"生成车辆：{random_bp.id}，位置：{random_spawn_point.location}")

        except Exception as e:
            # 生成失败时跳过（如位置冲突）
            print(f"车辆生成失败：{e}")
            continue

    return vehicle_list

# 4. 生成随机车辆（示例：生成20辆）
vehicle_list = spawn_random_vehicles(20)
print(f"成功生成 {len(vehicle_list)} 辆随机车辆")

# 5. 运行仿真，保持车辆状态
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 按下Ctrl+C时销毁所有车辆
    print("\n开始销毁车辆...")
    for vehicle in vehicle_list:
        if vehicle.is_alive:
            vehicle.destroy()
    print("所有车辆已销毁，仿真结束")
