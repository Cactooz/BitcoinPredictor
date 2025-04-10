import torch
import torch.nn as nn

class LSTMPrediction(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
		super(LSTMPrediction, self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
		self.fc1 = nn.Linear(hidden_size*2, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(p=dropout)

		self.layer_norm = nn.LayerNorm(input_size)

	def forward(self, x):
		output = self.layer_norm(x)
		output, _ = self.lstm(output)
		output = output[:, -1, :]
		output = self.fc1(output)
		output = self.activation(output)
		output = self.dropout(output)
		output = self.fc2(output)
		return output

def train(model, train_loader, criterion, optimizer, epochs=100, patience=30):
	train_losses = []
	val_losses = []
	best_val_loss = float("inf")

	for epoch in range(epochs):
		model.train()
		train_loss = 0
		for X_batch, y_batch in train_loader:
			
			# Predict the output
			y_pred = model(X_batch)
			loss = criterion(y_pred, y_batch)
			l1_lambda = 1e-4
			l1_norm = sum(p.abs().sum() for p in model.parameters())
			loss += l1_lambda * l1_norm

			# Reset gradients, calculate new ones and update model parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

		train_loss /= len(train_loader)
		train_losses.append(train_loss)
        # Validation phase
		model.eval()
		val_loss = 0
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch, y_batch = X_batch.to(device), y_batch.to(device)

				y_pred = model(X_batch)
				loss = criterion(y_pred, y_batch)
				val_loss += loss.item()

		val_loss /= len(val_loader)
		val_losses.append(val_loss)

		print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping logic
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
			# Save the best model
			torch.save(model.state_dict(), "../weights/best_model.pth")
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print(f"Early stopping triggered at epoch {epoch+1}")
				break

    # Load the best model before returning
	model.load_state_dict(torch.load("../weights/best_model.pth"))

	return model, train_losses
