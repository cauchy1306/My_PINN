import json
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import LR, HIDDEN_DIM, NUM_HIDDEN_LAYERS, STEPS, N_SAMPLES, X_RANGE, RUN_NAME
from config import N_FOURIER_FEATURES
from data import sample_data, fourier_feature_map
from model import MLP


def run_experiment(run_name, sigma):
    x, y = sample_data(N_SAMPLES, X_RANGE)

    x_encoded = fourier_feature_map(x, N_FOURIER_FEATURES, sigma)
    input_dim = 2 * N_FOURIER_FEATURES

    model = MLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                num_hidden_layers=NUM_HIDDEN_LAYERS)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        pred = model(x_encoded)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.6f}")

    with torch.no_grad():
        pred = model(x_encoded)

    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{run_dir}/model.pth")
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump({
            "lr": LR, "hidden_dim": HIDDEN_DIM,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "steps": STEPS, "n_samples": N_SAMPLES,
            "x_range": list(X_RANGE), "final_loss": loss.item(),
            "use_fourier": True,
            "n_fourier_features": N_FOURIER_FEATURES,
            "fourier_sigma": sigma,
        }, f, indent=2)

    print(f"  saved to {run_dir}/ (final loss {loss.item():.6f})")

    return x.numpy(), y.numpy(), pred.numpy()


if __name__ == "__main__":
    from config import FOURIER_SIGMA
    x, y, pred = run_experiment(RUN_NAME, FOURIER_SIGMA)

    plt.plot(x, y, label="target", linewidth=2)
    plt.plot(x, pred, label="pred", linestyle="--")
    plt.legend()
    plt.title("MLP fitting high-frequency signal")
    plt.show()
