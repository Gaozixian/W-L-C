import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图依赖


# ========== 原有2D折线图函数（保留不变） ==========
def plot_csv_column(file_path, target_column, fig_size=(12, 6), dpi=100):
    """
    读取CSV目标列，按先后顺序绘制折线图
    :param file_path: CSV文件路径（相对/绝对路径）
    :param target_column: 目标列名（字符串）或列索引（整数，0开始）
    :param fig_size: 图片尺寸（宽, 高），默认(12,6)
    :param dpi: 图片清晰度，默认100
    """
    # ---------------------- 1. 读取CSV数据 ----------------------
    try:
        # 根据列名/列索引读取目标列（只加载目标列，提升效率）
        if isinstance(target_column, str):
            # 按列名读取（推荐，无需关心列位置）
            df = pd.read_csv(
                file_path,
                usecols=[target_column],  # 只加载目标列，减少内存占用
                encoding='utf-8',  # 中文适配，乱码时改为'gbk'
                # errors='ignore'  # 忽略特殊字符错误
            )
        elif isinstance(target_column, int):
            # 按列索引读取（无表头时使用）
            df = pd.read_csv(
                file_path,
                usecols=[target_column],
                header=None,  # 无表头时指定header=None
                encoding='utf-8',
            )
            # 给无表头的列命名（方便后续显示）
            df.columns = [f'列_{target_column}']
            target_column = f'列_{target_column}'  # 更新列名用于后续显示
        else:
            print("错误：target_column必须是列名（字符串）或列索引（整数）")
            return

        # 数据预处理：去除空值、转换为数值类型（避免绘图失败）
        df = df.dropna()  # 删除空值行
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce').dropna()  # 转换为数值型，失败值设为NaN后删除
        if df.empty:
            print(f"错误：目标列「{target_column}」无有效数值数据")
            return

        # ---------------------- 2. 绘制折线图 ----------------------
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示乱码（Windows）
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

        # 创建画布
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        # 准备X轴（行索引=数据先后顺序，从1开始更直观）、Y轴（目标列数据）
        x_data = np.arange(1, len(df) + 1)  # X轴：1,2,3,...n（n为有效数据行数）
        y_data = df[target_column].values  # Y轴：目标列的数值数据

        # 绘制折线图（自定义样式：蓝色实线+圆点标记，线条宽度2）
        ax.plot(
            x_data, y_data,
            color='#2E86AB',  # 线条颜色（可替换为'red'/'green'等）
            linewidth=1.5,  # 线条宽度
            marker='.',  # 数据点标记（'.'=小点，'o'=圆点，None=无标记）
            markersize=3,  # 标记大小
            alpha=0.8  # 透明度（避免点重叠时看不清）
        )

        # ---------------------- 3. 图表美化（提升可读性） ----------------------
        ax.set_title(f'CSV列「{target_column}」数据趋势图（按先后顺序）', fontsize=16, pad=20)
        ax.set_xlabel('数据顺序（第n行）', fontsize=12)
        ax.set_ylabel(target_column, fontsize=12)  # Y轴标签用列名

        # 设置网格（便于读取数值）
        ax.grid(True, alpha=0.3, linestyle='--')

        # 调整坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)

        # 自动调整布局（避免标签被截断）
        plt.tight_layout()

        # ---------------------- 4. 显示/保存图片 ----------------------
        plt.show()  # 显示图片（运行后弹出窗口）
        # 可选：保存图片（路径可自定义，支持png/jpg/pdf格式）
        fig.savefig(f'{target_column}_trend.png', dpi=150, bbox_inches='tight')
        print(f"绘图完成！共处理{len(df)}行有效数据")

    except FileNotFoundError:
        print(f"错误：未找到文件「{file_path}」，请检查路径是否正确")
    except KeyError:
        print(f"错误：CSV文件中不存在列名「{target_column}」")
    except Exception as e:
        print(f"错误：绘图失败 → {str(e)}")


# ========== 新增3D坐标绘图函数 ==========
def plot_3d_coordinates(file_path, x_col='x', y_col='y', z_col='z', fig_size=(12, 8), dpi=100, save_fig=True):
    """
    读取CSV中的x/y/z列，绘制三维坐标轨迹图
    :param file_path: CSV文件路径
    :param x_col: x列名（默认'x'）
    :param y_col: y列名（默认'y'）
    :param z_col: z列名（默认'z'）
    :param fig_size: 图片尺寸
    :param dpi: 清晰度
    :param save_fig: 是否保存图片
    """
    try:
        # 1. 读取x/y/z三列数据（仅加载目标列，提升效率）
        df = pd.read_csv(
            file_path,
            usecols=[x_col, y_col, z_col],
            encoding='utf-8'
        )

        # 2. 数据预处理：去空值、转数值型
        df = df.dropna(subset=[x_col, y_col, z_col])  # 删除任意一列有空值的行
        for col in [x_col, y_col, z_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()  # 删除转换失败的行
        if df.empty:
            print("错误：x/y/z列无有效数值数据")
            return

        # 3. 准备绘图数据
        x_data = df[x_col].values
        y_data = df[y_col].values
        z_data = df[z_col].values
        data_len = len(df)
        print(f"共加载{data_len}行有效3D坐标数据")

        # 4. 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 5. 创建3D画布
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴

        # 6. 绘制3D轨迹（核心）
        # 绘制轨迹线（渐变颜色：从蓝到红，体现数据顺序）
        colors = plt.cm.jet(np.linspace(0, 1, data_len))  # 渐变颜色映射
        ax.plot(
            x_data, y_data, z_data,
            color='#2E86AB',  # 主线颜色
            linewidth=2,  # 线条宽度
            alpha=0.8,  # 透明度
            label='运动轨迹'
        )
        # 绘制散点（可选：突出数据点，颜色渐变体现顺序）
        scatter = ax.scatter(
            x_data, y_data, z_data,
            c=range(data_len),  # 按数据顺序着色
            cmap='jet',  # 配色方案
            s=10,  # 点大小
            alpha=0.9,
            edgecolors='black',  # 点边缘色
            linewidths=0.2
        )

        # 7. 图表美化
        ax.set_title('三维坐标轨迹图（X/Y/Z）', fontsize=18, pad=30)
        ax.set_xlabel(f'{x_col} 轴', fontsize=12, labelpad=15)
        ax.set_ylabel(f'{y_col} 轴', fontsize=12, labelpad=15)
        ax.set_zlabel(f'{z_col} 轴', fontsize=12, labelpad=15)

        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)
        # 添加颜色条（体现数据顺序）
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('数据顺序（第n行）', fontsize=10)

        # 自动调整布局
        plt.tight_layout()

        # 8. 显示/保存
        plt.show()
        if save_fig:
            fig.savefig('3d_coordinates.png', dpi=150, bbox_inches='tight')
            print("3D坐标图已保存为 3d_coordinates.png")

    except FileNotFoundError:
        print(f"错误：未找到文件「{file_path}」")
    except KeyError as e:
        print(f"错误：CSV中不存在列「{e}」，请检查列名是否正确")
    except Exception as e:
        print(f"3D绘图失败 → {str(e)}")


# ------------------- 调用示例（修改这部分！） -------------------
if __name__ == "__main__":
    # 1. 替换为你的CSV文件路径
    csv_file_path = "./driving_log_normalized.csv"

    # ========== 原有2D折线图调用（保留） ==========
    speed_target_col = "speed_kmh"  # 速度列名
    steer_target_col = "steer"  # 转向角列名

    # 绘制速度折线图
    plot_csv_column(
        file_path=csv_file_path,
        target_column=speed_target_col,
        fig_size=(14, 7),
        dpi=120
    )

    # 绘制转向角折线图
    plot_csv_column(
        file_path=csv_file_path,
        target_column=steer_target_col,
        fig_size=(14, 7),
        dpi=120
    )

    # ========== 新增3D坐标图调用（核心） ==========
    # 替换为你CSV中的x/y/z列名（比如：x_col='pos_x', y_col='pos_y', z_col='pos_z'）
    # plot_3d_coordinates(
    #     file_path=csv_file_path,
    #     x_col='global_x',  # 你的x列名
    #     y_col='global_y',  # 你的y列名
    #     z_col='global_z',  # 你的z列名
    #     fig_size=(14, 9),  # 3D图建议更大尺寸
    #     dpi=120,
    #     save_fig=True  # 是否保存图片
    # )