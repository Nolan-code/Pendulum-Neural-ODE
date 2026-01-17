import torch
import torch.nn as nn
import argparse
from pathlib import Path

from src.data.load_dataset import load_pendulum
from src.models.mlp_vectorfield import VectorFieldMLP
from src.models.hnn import HNN
from src.models.lnn import LNN

MODEL_REGISTRY = {
    "mlp": VectorFieldMLP,
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
parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')  
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
args = parser.parse_args()

import torch
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

#----------------
# Training loop
#----------------

train_loader, test_loader = load_pendulum(model_type=args.model)
n_epochs = args.epochs

for epoch in range(n_epochs):
  model.train()
  train_loss = 0.0

  for X_batch, Y_batch in train_loader:

    dx = model(X_batch)
    dtheta_pred = dx[:,0]
    domega_pred = dx[:,1]
    loss = (
        loss_fn(dtheta_pred, Y_batch[:,0]) +
        loss_fn(domega_pred, Y_batch[:,1]) 
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
      dtheta_pred = dx[:,0]
      domega_pred = dx[:,1]
      loss = (
        loss_fn(dtheta_pred, Y_batch[:,0]) +
        loss_fn(domega_pred, Y_batch[:,1])
      )
      test_loss += loss.item()

  test_loss /= len(test_loader)
  if epoch % 10 == 0:
      print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")
      
#-------------
# Save model
#-------------

checkpoint_dir = Path("./src/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / f"{args.model}.pth"
torch.save({
    "model": args.model,          
    "state_dict": model.state_dict(),
    "epochs": args.epochs,
    "lr": args.lr
    }, checkpoint_path)

print(f"Model saved to {checkpoint_path}")
