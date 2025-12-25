#-------------
# Load data
#-------------

data = np.load("pendulum_no_control_dataset.npz.npz")
split = np.load("splits_no_control.npz")

X = np.hstack(data["X"])
X = data["X"].reshape(10000, 2)
Y = data["Y"]
Y = data["Y"].reshape(10000, 2)
print(X.shape)

#---------------
# Normalization
#---------------

X_mean = X.mean()
X_std  = X.std()

Y_mean = Y.mean()
Y_std  = Y.std()

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std


X_train = X[split["train_idx"]]
Y_train = Y[split["train_idx"]]

X_test = X[split["test_idx"]]
Y_test = Y[split["test_idx"]]

#-------------------
# Convert to tensor
#-------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

#---------
#  Model
#---------

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

#----------------
# Training loop
#----------------

n_epochs = 500

for epoch in range(n_epochs):
  model.train()
  train_loss = 0.0

  for X_batch, Y_batch in train_loader:
    Y_pred = hamiltonian_dynamics(model, X_batch)
    loss = loss_fn(Y_pred, Y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  train_loss /= len(train_loader)


  model.eval()
  test_loss = 0.0

  for X_batch, Y_batch in test_loader:
      Y_pred = hamiltonian_dynamics(model, X_batch)
      loss = loss_fn(Y_pred, Y_batch)
      test_loss += loss.item()

  test_loss /= len(test_loader)
  print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

#-------------
# Save model
#-------------

torch.save(model.state_dict(), "HNN_rk4.pth")
