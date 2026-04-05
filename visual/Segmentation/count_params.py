import sys
import torch
import json
import os

sys.path.append(os.getcwd())

# 1. 导入 FPN 和 DeepLab
from model import ResNet50_AdaptiveInteraction_Segmentation
from model import DPAI_DeepLabV3P_Segmentation

# 2. 动态提取 Notebook 中的 DPAI_UNet_Segmentation 并加载
with open('dpai_unet.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_strs = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'class DPAI_UNet_Segmentation' in source:
            code_strs.append(source)

# 动态执行 Notebook 里的代码块
exec(code_strs[0], globals())

# 3. 计算参数量的核心函数
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

try:
    print("Initializing Models...")
    m1 = ResNet50_AdaptiveInteraction_Segmentation(num_classes=19)
    m2 = globals()['DPAI_UNet_Segmentation'](num_classes=19)
    m3 = DPAI_DeepLabV3P_Segmentation(num_classes=19)

    p1 = count_parameters(m1)
    p2 = count_parameters(m2)
    p3 = count_parameters(m3)

    print("================ Parameter Counts ================")
    print(f"DPAI_FPN (级联重叠) Params:        {p1 / 1e6:.2f} M")
    print(f"DPAI_UNet (对称连接) Params:       {p2 / 1e6:.2f} M")
    print(f"DPAI_Deeplabv3p (ASPP宏观) Params: {p3 / 1e6:.2f} M")
    print("==================================================")
except Exception as e:
    print(f"Error during parameter counting: {e}")
