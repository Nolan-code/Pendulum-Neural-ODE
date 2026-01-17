import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

def load_double_pendulum(format):
    chemin_data_train = Path(__file__).parent.parent / 'data' / 'double_pendulum_train2.npz'
    chemin_data_test = Path(__file__).parent.parent / 'data' / 'double_pendulum_test2.npz'
    
    data_train = np.load(chemin_data_train)
    data_test = np.load(chemin_data_test)

    if format in ["sincos"]:
        theta1_train = data_train["theta1"]
        theta2_train = data_train["theta2"]
        omega1_train = data_train["omega1"]
        omega2_train = data_train["omega2"]

        #omega1_train = (omega1_train - omega1_train.mean())/omega1_train.std()
        #omega2_train = (omega2_train - omega2_train.mean())/omega2_train.std()
        X_train = np.stack([np.sin(theta1_train), np.cos(theta1_train), np.sin(theta2_train), np.cos(theta2_train), omega1_train, omega2_train], axis=1)
        dx_train = np.stack([data_train["dtheta1"], data_train["dtheta2"], data_train["domega1"], data_train["domega2"]], axis=1)

        theta1_test = data_test["theta1"]
        theta2_test = data_test["theta2"]
        omega1_test = data_test["omega1"]
        omega2_test = data_test["omega2"]

        #omega1_test = (omega1_test - omega1_test.mean())/omega1_test.std()
        #omega2_test = (omega2_test - omega2_test.mean())/omega2_test.std()

        X_test = np.stack([np.sin(theta1_test), np.cos(theta1_test), np.sin(theta2_test), np.cos(theta2_test), omega1_test, omega2_test], axis=1)
        dx_test = np.stack([data_test["dtheta1"], data_test["dtheta2"], data_test["domega1"], data_test["domega2"]], axis=1)

    elif format in ["theta"]:
        X_train = np.stack([data_train["theta1"], data_train["theta2"], data_train["omega1"], data_train["omega2"]], axis=1)
        dx_train = np.stack([data_train["dtheta1"], data_train["dtheta2"], data_train["domega1"], data_train["domega2"]], axis=1)

        X_test = np.stack([data_test["theta1"], data_test["theta2"], data_test["omega1"], data_test["omega2"]], axis=1)
        dx_test = np.stack([data_test["dtheta1"], data_test["dtheta2"], data_test["domega1"], data_test["domega2"]], axis=1)

    else:
        raise ValueError("Unknown format")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    dx_train = torch.tensor(dx_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    dx_test = torch.tensor(dx_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, dx_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = TensorDataset(X_test, dx_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader
