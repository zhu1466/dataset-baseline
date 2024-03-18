import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


def mlp_train_test_proc(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.DataFrame,
                        y_test: pd.DataFrame,
                        model_best_configs: dict)->dict:
    
    input_size = 88
    hidden_size1 = 256
    hidden_size2 = 64
    hidden_size3 = 16
    output_size = 1
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3
    device = 'cpu'

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # 模型训练
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:

            # 将数据移动到 GPU 上
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy()

    y_pred = y_pred.flatten()
    auc = roc_auc_score( y_test, y_pred)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率:", accuracy)
    print("AUC:", auc)

    return {'acc':accuracy, 'auc': auc}



    