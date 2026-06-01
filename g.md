graph TD
      A["3d912a0<br/>Initial commit<br/>建立基础MLP"]
      B["42018ec<br/>傅里叶特征映射"]
      C["13681de<br/>σ对比实验"]
      D["0e8e333<br/>完善说明文档"]
      E["a406dcc<br/>first commit<br/>一维热传导方程"]
      F["d298187<br/>Merge commit"]

      A --> B --> C --> D
      A --> F
      E --> F

      D -..- FF["fourier-feature-mapping 分支指针 ← HEAD 在这里"]
      F -..- MM["main 分支指针"]

      style A fill:#e8e8e8
      style E fill:#e8e8e8
      style F fill:#fff3cd
      style D fill:#d4edda
      style FF fill:none,stroke:none
      style MM fill:none,stroke:none