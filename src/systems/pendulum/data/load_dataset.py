import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

chemin_data = Path(__file__).parent.parent / 'data' / 'mon_fichier.npz'
data = np.load(chemin_data)

def load_pendulum(model_type="hnn"):
    chemin_data_train = Path(__file__).parent.parent / 'data' / 'dataset_train.npz'
    chemin_data_test = Path(__file__).parent.parent / 'data' / 'dataset_test.npz'

    data_train = np.load(chemin_data_train)
    data_test = np.load(chemin_data_test)

    if model_type in ["hnn","lnn"]:
        theta_train = data_train["theta"]
        omega_train = data_train["omega"]
        X_train = np.stack([np.sin(theta_train), np.cos(theta_train), omega_train], axis=1)
        dx_train = np.stack([data_train["d_theta"], data_train["d_omega"]], axis=1)  

        theta_test = data_test["theta"]
        omega_test = data_test["omega"]
        X_test = np.stack([np.sin(theta_test), np.cos(theta_test), omega_test], axis=1)
        dx_test = np.stack([data_test["d_theta"], data_test["d_omega"]], axis=1)

    elif model_type in ["mlp", "vector_field"]:
        X_train = np.stack([data_train["theta"], data_train["omega"]], axis=1)
        dx_train = np.stack([data_train["d_theta"], data_train["d_omega"]], axis=1)

        X_test = np.stack([data_test["theta"], data_test["omega"]], axis=1)
        dx_test = np.stack([data_test["d_theta"], data_test["d_omega"]], axis=1)

    else:
        raise ValueError("Unknown model_type")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    dx_train = torch.tensor(dx_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    dx_test = torch.tensor(dx_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, dx_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(X_test, dx_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, test_loader
