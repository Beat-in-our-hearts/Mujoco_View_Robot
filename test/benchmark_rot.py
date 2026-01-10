import numpy as np
from scipy.spatial.transform import Rotation as R
import time

N = 100_000
q1 = R.from_quat([0, 0, 0, 1])           # 单位四元数
q2_quat = np.random.randn(N, 4)
q2_quat = q2_quat / np.linalg.norm(q2_quat, axis=1, keepdims=True)  # 归一化
q2 = R.from_quat(q2_quat)

# 测试乘法速度
t0 = time.time()
combined = q1 * q2
t1 = time.time()
print(f"组合 {N} 个旋转耗时: {t1 - t0:.4f} 秒")