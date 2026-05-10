import torch
import matplotlib.pyplot as plt

from config import HIDDEN_DIM, NUM_HIDDEN_LAYERS, N_SAMPLES, X_RANGE, RUN_NAME
from data import sample_data
from model import MLP


def main():
    x, y = sample_data(N_SAMPLES, X_RANGE)

    model_path = f"runs/{RUN_NAME}/model.pth"
    model = MLP(HIDDEN_DIM, NUM_HIDDEN_LAYERS)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(x)

    plt.plot(x.numpy(), y.numpy(), label="target", linewidth=2)
    plt.plot(x.numpy(), pred.numpy(), label="pred", linestyle="--")
    plt.legend()
    plt.title("MLP inference (loaded model)")
    plt.show()


if __name__ == "__main__":
    main()
