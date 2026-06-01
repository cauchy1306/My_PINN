import torch
import matplotlib.pyplot as plt

from config import HIDDEN_DIM, NUM_HIDDEN_LAYERS, N_SAMPLES, X_RANGE, RUN_NAME
from config import USE_FOURIER, N_FOURIER_FEATURES, FOURIER_SIGMA
from data import sample_data, fourier_feature_map
from model import MLP


def main():
    x, y = sample_data(N_SAMPLES, X_RANGE)

    if USE_FOURIER:
        x_encoded = fourier_feature_map(x, N_FOURIER_FEATURES, FOURIER_SIGMA)
        input_dim = 2 * N_FOURIER_FEATURES
    else:
        x_encoded = x
        input_dim = 1

    model_path = f"runs/{RUN_NAME}/model.pth"
    model = MLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                num_hidden_layers=NUM_HIDDEN_LAYERS)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(x_encoded)

    plt.plot(x.numpy(), y.numpy(), label="target", linewidth=2)
    plt.plot(x.numpy(), pred.numpy(), label="pred", linestyle="--")
    plt.legend()
    plt.title("MLP inference (loaded model)")
    plt.show()


if __name__ == "__main__":
    main()
