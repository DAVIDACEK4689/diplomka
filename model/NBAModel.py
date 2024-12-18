import torch
import torch.nn as nn

from model.TrainableModule import TrainableModule


class NBAModel(TrainableModule):
    def __init__(self, team_features, players_count, players_features):
        super().__init__()

        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, players_features)),
            nn.Tanh()
        )
        self.flatten = nn.Flatten()

        # Dynamic output size for conv layer
        self.conv_output_size = 32 * players_count

        # Dense layers (combining conv output + team input)
        self.fc1 = nn.Linear(self.conv_output_size + team_features, 32)
        self.fc2 = nn.Linear(32, 8)
        self.output = nn.Linear(8, 1)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, team_input, player_input):
        # Reshape player input for Conv2D
        x_conv = player_input.unsqueeze(1)
        x_conv = self.conv(x_conv)
        x_conv = self.flatten(x_conv)

        # Concatenate conv output with team stats
        x = torch.cat([x_conv, team_input], dim=1)

        # Fully connected layers
        x = nn.functional.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.tanh(self.fc2(x))
        x = self.dropout2(x)

        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output(x))
        x = x.squeeze()
        return x



def train_model(model, train_loader, dev_loader, criterion, optimizer, epochs, device="auto"):
    """
    Train the model and evaluate on the validation set.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        dev_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (CPU or GPU).
        epochs: Number of training epochs.
    """
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training loop
        for team_input, player_input, labels in train_loader:
            team_input, player_input, labels = (team_input.to(device), player_input.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(team_input, player_input).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, dev_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to run on (CPU or GPU).

    Returns:
        Average loss over the dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for team_input, player_input, labels in data_loader:
            team_input, player_input, labels = (team_input.to(device), player_input.to(device), labels.to(device))
            outputs = model(team_input, player_input).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def save_model(model, path):
    torch.save(model.state_dict(), path)