import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import print_model_results
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

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.zeros_(self.fc3.bias)
        # nn.init.xavier_uniform_(self.fc4.weight)
        # nn.init.zeros_(self.fc4.bias)

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
    
    input_size = len(X_train.columns)
    output_size = 1
    device = 'cuda'

    try:
        hidden_size1 = model_best_configs['MLP']['hidden_size1']
        hidden_size2 = model_best_configs['MLP']['hidden_size2']
        hidden_size3 = model_best_configs['MLP']['hidden_size3']
        learning_rate = model_best_configs['MLP']['learning_rate']
        batch_size = model_best_configs['MLP']['batch_size']
        num_epochs = model_best_configs['MLP']['num_epochs']

    except:
        print('Model params havent be saved as yml files, please run files from model_tuning first!')
        return False

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_record = []
    model.train()
    # 模型训练
    for epoch in range(num_epochs):
        loss_per_epoch = 0.00
        for inputs, targets in train_loader:

            optimizer.zero_grad()

            # 将数据移动到 GPU 上
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_per_epoch += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()
        
        loss_per_epoch = loss_per_epoch/len(train_loader)
        loss_record.append(loss_per_epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_per_epoch:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy()

    y_pred = y_pred.flatten()
    auc = roc_auc_score( y_test, y_pred)


    y_pred_classifier = y_pred.copy()
    y_pred_classifier[y_pred_classifier >= 0.5] = 1
    y_pred_classifier[y_pred_classifier < 0.5] = 0
    acc = accuracy_score(y_test, y_pred_classifier)
    precision = precision_score(y_test, y_pred_classifier)
    recall = recall_score(y_test, y_pred_classifier)
    f1 = f1_score(y_test, y_pred_classifier)
        
    model_save_path = 'machine_learn\\deep_learn_model_save\\MLP.pt' 
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(model_save_path)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1Score': f1,
               'loss_series': loss_record}
    print_model_results(model_name='MLP', model_results = results)
    return results



    