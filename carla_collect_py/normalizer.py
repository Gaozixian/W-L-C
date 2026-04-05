import pandas as pd
import numpy as np
import json
from typing import Dict, Any


class dataNormalizer:
    def __init__(self, file_path):
        """初始化：读取驾驶日志数据并筛选数值列"""
        self.df = pd.read_csv(file_path)
        self.normalize_cols = ['steer', 'throttle', 'brake', 'speed_kmh']  # 需归一化的驾驶特征
        self.normalize_params: Dict[str, Dict[str, Any]] = {}  # 存储归一化参数

    def _convert_numpy_to_python(self, data):
        """辅助函数：将NumPy类型（int64/float64）转换为Python原生类型（避免JSON序列化报错）"""
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def min_max_normalize(self, df_input: pd.DataFrame, feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Min-Max归一化：将特征映射到指定区间（默认[0,1]）
        适用场景：油门（throttle）、刹车（brake）等有明确范围的特征
        """
        df_norm = df_input.copy()
        min_val, max_val = feature_range

        for col in self.normalize_cols:
            # 计算原始特征的最值
            original_min = df_norm[col].min()
            original_max = df_norm[col].max()
            # 转换为Python原生类型并保存参数
            self.normalize_params[col] = {
                **self.normalize_params.get(col, {}),
                'min_max': {
                    'original_min': self._convert_numpy_to_python(original_min),
                    'original_max': self._convert_numpy_to_python(original_max),
                    'target_min': min_val,
                    'target_max': max_val
                }
            }
            # 避免除以0（处理特征值全相同的情况）
            if original_max - original_min < 1e-10:
                df_norm[f'{col}_minmax'] = min_val
            else:
                # Min-Max核心公式
                df_norm[f'{col}_minmax'] = (df_norm[col] - original_min) / (original_max - original_min) * (
                            max_val - min_val) + min_val
        return df_norm

    def zscore_normalize(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Z-Score标准化：将特征转换为均值=0、标准差=1的分布
        适用场景：转向角（steer）、速度（speed_kmh）等无明确范围的特征
        """
        df_norm = df_input.copy()

        for col in self.normalize_cols:
            # 计算原始特征的均值和标准差
            original_mean = df_norm[col].mean()
            original_std = df_norm[col].std()
            # 转换为Python原生类型并保存参数
            self.normalize_params[col] = {
                **self.normalize_params.get(col, {}),
                'zscore': {
                    'original_mean': self._convert_numpy_to_python(original_mean),
                    'original_std': self._convert_numpy_to_python(original_std)
                }
            }
            # 避免除以0（处理特征值全相同的情况）
            if original_std < 1e-10:
                df_norm[f'{col}_zscore'] = 0.0
            else:
                # Z-Score核心公式
                df_norm[f'{col}_zscore'] = (df_norm[col] - original_mean) / original_std
        return df_norm

    def save_results(self, df_norm: pd.DataFrame,
                     data_path: str = 'vehicle_data_normalized.csv',
                     params_path: str = 'normalize_params.json') -> None:
        """保存归一化后的数据和参数（支持后续模型训练和新数据归一化）"""
        # 保存归一化数据（包含原始列和两种归一化列）
        df_norm.to_csv(data_path, index=False, encoding='utf-8')
        # 保存归一化参数（JSON格式，可复用）
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(self.normalize_params, f, indent=4)
        print(f"✅ 归一化数据文件：{data_path}")
        print(f"✅ 归一化参数文件：{params_path}")

    def inverse_min_max(self, normalized_val: float, col: str) -> float:
        """Min-Max逆归一化：将模型输出的归一化值还原为原始物理值（如速度km/h、转向角）"""
        params = self.normalize_params.get(col, {}).get('min_max')
        if not params:
            raise ValueError(f"请先对{col}列执行min_max_normalize，再进行逆归一化")

        original_min = params['original_min']
        original_max = params['original_max']
        target_min = params['target_min']
        target_max = params['target_max']

        # 逆归一化公式
        original_val = (normalized_val - target_min) / (target_max - target_min) * (
                    original_max - original_min) + original_min
        return original_val

class inverseNormalizer:
    def __init__(self, params_path="normalize_params.json"):
        """初始化：加载归一化参数"""
        with open(params_path, "r", encoding="utf-8") as f:
            self.params = json.load(f)

    def inverse_min_max(self, norm_value, col):
        """Min-Max反归一化（指定列名）"""
        p = self.params[col]["min_max"]
        return (norm_value - p["target_min"]) / (p["target_max"] - p["target_min"]) * (p["original_max"] - p["original_min"]) + p["original_min"]

    def inverse_zscore(self, std_value, col):
        """Z-Score反标准化（指定列名）"""
        p = self.params[col]["zscore"]
        return std_value * p["original_std"] + p["original_mean"]


# ------------------- 主执行逻辑（直接运行即可） -------------------
if __name__ == "__main__":
    # 1. 加载数据
    data_path = '../LSTM/driving_log.csv'
    normalizer = dataNormalizer(file_path=data_path)
    print(f"📊 数据加载完成：共{normalizer.df.shape[0]}行数据，{normalizer.df.shape[1]}列")
    print(f"🎯 待归一化的驾驶特征：{normalizer.normalize_cols}")

    # 2. 执行两种归一化（保留原始数据，新增归一化列）
    df_with_minmax = normalizer.min_max_normalize(df_input=normalizer.df)
    df_final = normalizer.zscore_normalize(df_input=df_with_minmax)

    # 3. 展示归一化效果（前3行关键特征对比）
    print("\n" + "=" * 80)
    print("🔍 归一化前后对比（前3行）：")
    display_cols = [
        'timestamp', 'steer', 'steer_minmax', 'steer_zscore',
        'speed_kmh', 'speed_kmh_minmax', 'speed_kmh_zscore'
    ]
    print(df_final[display_cols].head(3).round(6))  # 保留6位小数，便于查看

    # 4. 验证归一化正确性（Z-Score均值≈0，标准差≈1；Min-Max值在[0,1]）
    print("\n" + "=" * 80)
    print("✅ 归一化有效性验证：")
    zscore_cols = [col for col in df_final.columns if 'zscore' in col]
    zscore_stats = df_final[zscore_cols].agg(['mean', 'std']).round(6)
    print("Z-Score特征（均值≈0，标准差≈1）：")
    print(zscore_stats)

    minmax_cols = [col for col in df_final.columns if 'minmax' in col]
    minmax_stats = df_final[minmax_cols].agg(['min', 'max']).round(6)
    print("\nMin-Max特征（值在[0,1]区间）：")
    print(minmax_stats)

    # 5. 保存结果
    normalizer.save_results(df_norm=df_final)

    # 6. 反归一化
    normalizer = inverseNormalizer()
    # 还原转向角
    print(normalizer.inverse_min_max(0.53604, "steer"))  # 输出：0.0