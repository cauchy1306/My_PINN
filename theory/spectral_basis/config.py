# 超参数
LR = 1e-3                 #步长
HIDDEN_DIM = 10           #隐藏层维数
NUM_HIDDEN_LAYERS = 2     #隐藏层层数
STEPS = 5000              #训练步数
N_SAMPLES = 200           #样本数
X_RANGE = (0.0, 1.0)      #样本范围
# RUN_NAME = "baseline"   #实验名称，模型存在 runs/{RUN_NAME}/ 下
RUN_NAME = "fourier"

# 傅里叶特征映射
USE_FOURIER = True
N_FOURIER_FEATURES = 40    # 映射特征数（输出维度 = 此值 × 2）
FOURIER_SIGMA = 10.0       # 频率采样标准差 σ
