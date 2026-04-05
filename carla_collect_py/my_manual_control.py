import carla
import sys
import time
# 导入官方manual_control的核心类（需添加examples目录到Python路径）
sys.path.append("../")  # 若脚本在examples子目录，需添加上级路径（根据实际目录调整）
from manual_control import KeyboardControl  # 导入官方手动控制类
from carla_data_collector import CarlaDataCollector  # 导入数据采集模块

def find_existing_tesla(world):
    """
    在Carla世界中查找已生成的特斯拉Model3车辆
    :param world: Carla世界对象
    :return: 找到的特斯拉车辆Actor（None表示未找到）
    """
    # 遍历所有已生成的Actor
    for actor in world.get_actors():
        # 判断是否是车辆，且蓝图名称为特斯拉Model3
        if actor.type_id == "vehicle.tesla.cybertruck" and actor.attributes.get("role_name", "") != "autopilot":
            print(f"找到已生成的特斯拉车辆：ID={actor.id}")
            return actor
    return None

def main():
    # 1. 连接Carla服务器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 2. 查找已生成的特斯拉车辆（核心修改：不再生成新车辆）
    vehicle = find_existing_tesla(world)
    if not vehicle:
        print("ERROR：未在Carla服务器中找到已生成的特斯拉Model3！")
        print("请先在Carla中生成特斯拉车辆（可通过官方spawn_npc.py或手动生成）")
        return

    # 3. 初始化数据采集器（传入已找到的车辆）
    collector = CarlaDataCollector(
        client=client,
        vehicle=vehicle,
        save_root="carla_data_collect",
        camera_res=(640, 480),
        sensor_tick=0.1  # 采集频率与控制循环匹配
    )
    collector.spawn_cameras()  # 挂载四方向摄像头
    print("数据采集器初始化完成，已启动图像+车辆状态采集")

    # 4. 初始化官方手动控制类（复用官方键盘控制逻辑）
    # keyboard_control = KeyboardControl(world, vehicle)
    # print("\n===== 官方手动控制按键说明 =====")
    # print("W：前进  S：后退  A：左转  D：右转")
    # print("空格：刹车  Q：退出  P：切换视角")
    # print("↑↓：调节油门灵敏度  ←→：调节转向灵敏度")

    try:
        # 5. 手动控制+数据采集循环（复用官方控制循环逻辑）
        while True:
            # 官方手动控制：处理键盘输入并应用车辆控制
            # if not keyboard_control.parse_events():
            #     break  # 若返回False（如按下Q键），退出循环

            # 数据采集：与控制循环同步执行
            collector.collect_data()

            # 保持循环频率（与采集频率一致）
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n采集被用户中断！")
    finally:
        # 6. 清理资源
        collector.stop_collect()  # 停止数据采集并保存文件
        # 官方手动控制的清理（释放窗口等资源）
        # keyboard_control.cleanup()
        print("资源清理完成，脚本结束！")

if __name__ == "__main__":
    main()
# sensor_tick=0.1 time.sleep(0.05)要进行修改