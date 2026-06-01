# Chapter 1: MLP 拟合高频信号

## 目标

用 MLP 拟合手动构造的 1D 高频信号 `f(x) = sin(2πx) + 0.1·sin(50πx)`，直观感受：
- **频谱偏置（spectral bias）**：MLP 天然偏向学习低频分量，高频收敛慢
- **NTK（Neural Tangent Kernel）**：训练初期 NTK 的特征值分布决定了收敛速度

## 项目结构

```
chapter1/
├── config.py    # 超参数集中管理
├── model.py     # MLP 模型定义
├── data.py      # 目标函数 + 采样 + 傅里叶特征映射
├── train.py     # 单次训练 + 保存
├── infer.py     # 加载已有模型 + 推理画图
├── compare.py   # σ 对比实验（遍历多个 σ，一张图对比）
└── runs/        # 实验产物（.gitignore 忽略）
```

## 用法

```bash
python train.py      # 按 config.py 的参数训练
python infer.py      # 加载 runs/{RUN_NAME}/model.pth 推理
python compare.py    # 遍历 σ=[1,5,10,20,50]，训练完后一张图对比
```

## 实验

### 1. Baseline — 原始 MLP
- `RUN_NAME = "baseline"`, `USE_FOURIER = False`
- 输入维度 1，直接学 `x → f(x)`
- 结果：低频（2π）能拟合，高频（50π）严重欠拟合 → 验证频谱偏置

### 2. Fourier Feature Mapping
- 对输入 x 做傅里叶特征映射：`γ(x) = [cos(2πBx), sin(2πBx)]`，B ~ N(0, σ²)
- 输入维度从 1 膨胀到 80（40 个频率 × cos+sin）
- σ 控制频率带宽：σ 越大，基函数包含越高频分量

### 3. σ 对比实验
- `python compare.py` 对比 σ ∈ {1.0, 5.0, 10.0, 20.0, 50.0}
- 预期：σ 过小 → 只能学低频；σ 合适 → 低频高频都能学；σ 过大 → 过拟合/噪声
