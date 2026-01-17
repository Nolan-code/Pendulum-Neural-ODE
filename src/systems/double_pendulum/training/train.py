import torch
import torch.nn as nn
import argparse
from pathlib import Path

from src.systems.double_pendulum.data.load_double_pendulum import load_double_pendulum
from src.systems.double_pendulum.models.mlp import MLP
from src.systems.double_pendulum.models.hnn import HNN
from src.systems.double_pendulum.models.lnn import LNN

MODEL_REGISTRY = {
    "mlp": MLP,
    "hnn": HNN,
    "lnn": LNN,
}

def build_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]()
    except KeyError:
        raise ValueError(f"Unknown model type: {model_name}")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model type: mlp, hnn, or lnn')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')  
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
args = parser.parse_args()

import numpy as np
import random

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)
#---------
#  Model
#---------

model = build_model(args.model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)


#----------------
# Training loop
#----------------

if args.model in ["mlp","hnn"]:
  format = "sincos"
elif args.model == "lnn":
  format = "theta"

alpha = 0.1

train_loader, test_loader = load_double_pendulum(format)
n_epochs = args.epochs

for epoch in range(n_epochs):
  model.train()
  train_loss = 0.0

  for X_batch, Y_batch in train_loader:

    dx = model(X_batch)
    dtheta1_pred = dx[:,0]
    dtheta2_pred = dx[:,1]
    domega1_pred = dx[:,2]
    domega2_pred = dx[:,3]

    loss = (
        loss_fn(dtheta1_pred, Y_batch[:,0]) +
        loss_fn(dtheta2_pred, Y_batch[:,1]) +
        alpha * loss_fn(domega1_pred, Y_batch[:,2]) +
        alpha * loss_fn(domega2_pred, Y_batch[:,3])
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  train_loss /= len(train_loader)

  model.eval()

  test_loss = 0.0
  for X_batch, Y_batch in test_loader:

    dx = model(X_batch)
    dtheta1_pred = dx[:,0]
    dtheta2_pred = dx[:,1]
    domega1_pred = dx[:,2]
    domega2_pred = dx[:,3]

    loss = (
        loss_fn(dtheta1_pred, Y_batch[:,0]) +
        loss_fn(dtheta2_pred, Y_batch[:,1]) +
        alpha * loss_fn(domega1_pred, Y_batch[:,2]) +
        alpha * loss_fn(domega2_pred, Y_batch[:,3])
        )


    test_loss += loss.item()

  test_loss /= len(test_loader)
  scheduler.step(test_loss)

  if epoch % 10 == 0:
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
      
#-------------
# Save model
#-------------

checkpoint_dir = Path("./src/systems/double_pendulum/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / f"{args.model}.pth"
torch.save({
    "model": args.model,          
    "state_dict": model.state_dict(),
    "epochs": args.epochs,
    "lr": args.lr
    }, checkpoint_path)

print(f"Model saved to {checkpoint_path}")
