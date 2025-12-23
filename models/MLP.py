#-------------
# Load data
#-------------

data = np.load("pendulum_dataset.npz")
split = np.load("splits.npz")

X = np.hstack([data["X"], data["U"]])
Y = data["X_next"]

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

n_epochs = 50

for epoch in range(n_epochs):
  model.train()
  train_loss = 0.0

  for X_batch, Y_batch in train_loader:
    Y_pred = model(X_batch)
    loss = loss_fn(Y_pred, Y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  train_loss /= len(train_loader)

  model.eval()

  with torch.no_grad(): # Compute loss on test dataset without grad
    test_loss = 0.0
    for X_batch, Y_batch in test_loader:
      Y_pred = model(X_batch)
      loss = loss_fn(Y_pred, Y_batch)

      test_loss += loss.item()

    test_loss /= len(test_loader)

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

#-------------
# Save model
#-------------

torch.save(model.state_dict(), "mlp.pth")
