import matplotlib.pyplot as plt

from train import run_experiment

SIGMAS = [1.0, 5.0, 10.0, 20.0, 50.0]

results = {}

for s in SIGMAS:
    print(f"\n=== sigma = {s} ===")
    x, y, pred = run_experiment(run_name=f"sigma_{s}", sigma=s)
    results[s] = (x, y, pred)

# 一张图对比
plt.figure(figsize=(10, 6))
plt.plot(x, y, "k-", label="target", linewidth=2)
for s in SIGMAS:
    _, _, pred = results[s]
    plt.plot(x, pred, linestyle="--", label=f"σ={s}")

plt.legend()
plt.title("Fourier feature: effect of σ on high-frequency fitting")
plt.show()
