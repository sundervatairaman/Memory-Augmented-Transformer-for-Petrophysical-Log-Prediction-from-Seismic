# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

# -------------------- Custom Memory-Augmented Transformer Modules -------------------- #

class TitansMemoryModule(nn.Module):
    def __init__(self, input_dim, mem_capacity=500, momentum_factor=0.9, lr_memory=0.1, weight_decay=0.01):
        super().__init__()
        self.register_buffer('memory', torch.zeros((mem_capacity, input_dim)))
        self.register_buffer('surprise_scores', torch.zeros(mem_capacity))
        self.register_buffer('momentum_buffer', torch.zeros((mem_capacity, input_dim)))
        self.ptr = 0
        self.capacity = mem_capacity
        self.surprise_threshold = 0.5
        self.forgetting_rate = 0.1
        self.momentum_factor = momentum_factor
        self.lr_memory = lr_memory
        self.weight_decay = weight_decay

    def calculate_surprise(self, x):
        if torch.norm(self.memory) == 0:
            return torch.ones(x.size(0), device=x.device)
        similarities = torch.cosine_similarity(x.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)
        return 1 - torch.max(similarities, dim=1)[0]

    def update_memory(self, x, surprises):
        batch_size = x.size(0)
        for i in range(batch_size):
            if surprises[i] > self.surprise_threshold or self.ptr < self.capacity:
                index = self.ptr % self.capacity
                diff = x[i] - self.memory[index]
                self.momentum_buffer[index] = self.momentum_factor * self.momentum_buffer[index] + \
                                              (1 - self.momentum_factor) * diff
                self.memory[index] += self.lr_memory * self.momentum_buffer[index]
                self.surprise_scores[index] = surprises[i]
                self.ptr += 1

    def adaptive_forgetting(self):
        if self.ptr >= self.capacity:
            avg_score = torch.mean(self.surprise_scores)
            mask = self.surprise_scores < (avg_score * self.forgetting_rate)
            self.memory[mask] *= (1 - self.weight_decay)
            self.surprise_scores[mask] *= (1 - self.weight_decay)
            self.momentum_buffer[mask] *= (1 - self.weight_decay)


class TitansTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, mem_capacity=500):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.memory_module = TitansMemoryModule(d_model, mem_capacity)
        self.activation = nn.ReLU()

    def forward(self, src):
        surprises = self.memory_module.calculate_surprise(src)
        mem_context = self.memory_module.memory
        if mem_context.size(0) > 0:
            mem_context = mem_context.unsqueeze(1).repeat(1, src.size(0), 1)
            enhanced_src = torch.cat([mem_context, src.unsqueeze(0)], dim=0)
        else:
            enhanced_src = src.unsqueeze(0)
        src2, _ = self.self_attn(enhanced_src, enhanced_src, enhanced_src)
        src = src + self.dropout1(src2[-1])
        src = self.norm1(src)
        self.memory_module.update_memory(src.detach(), surprises.detach())
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if np.random.rand() < 0.05:
            self.memory_module.adaptive_forgetting()
        return src


class TitansModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_size, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList([
            TitansTransformerLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                mem_capacity=500
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, src):
        x = self.input_proj(src)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


# -------------------- Data Loading and Training Pipeline -------------------- #

# Load dataset
raw_dataset = pd.read_excel("merged_kj_w1.xlsx", na_values="?", comment='\t')
dataset = raw_dataset.dropna()

# Features and targets
features = ["TWT", "D2", "Quadr", "TraceGrad", "GradMag", "Freq", "Zone1"]
target = ["AI_Log", "RT", "RHOB", "NPHI"]

# Split dataset
train_indices = dataset[dataset["WELL"] < 8].index
test_indices = dataset[dataset["WELL"] >= 8].index

train_data = dataset.loc[train_indices, features]
train_target = dataset.loc[train_indices, target]
test_data = dataset.loc[test_indices, features]
test_target = dataset.loc[test_indices, target]

# Normalize
scaler = PowerTransformer(method='yeo-johnson', standardize=True)
#scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
# No scaling - use raw feature values
#train_data = train_data.values
#test_data = test_data.values


# Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

# Loaders
train_dataset = CustomDataset(train_data, train_target.values)
test_dataset = CustomDataset(test_data, test_target.values)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TitansModel(input_dim=7, output_dim=4, num_heads=4, hidden_size=128, num_layers=8).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
loss_history = []
for epoch in range(100):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    #print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'gr_model.pt')

# Evaluate
model.eval()
with torch.no_grad():
    test_inputs = test_dataset.features.to(device)
    test_preds = model(test_inputs).cpu().numpy()

print("MAE:", mean_absolute_error(test_target, test_preds))
print("MSE:", mean_squared_error(test_target, test_preds))

# Plot loss
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.show()

# Add predictions to test set
test_dataset1 = dataset.loc[test_indices].copy()
test_dataset1["AI_Log_Pred"] = test_preds[:, 0]
test_dataset1["RT_Pred"] = test_preds[:, 1]
test_dataset1["RHOB_Pred"] = test_preds[:, 2]
test_dataset1["NPHI_Pred"] = test_preds[:, 3]

# Plot true vs predicted AI_Log
plt.scatter(test_dataset1["AI_Log"], test_dataset1["AI_Log_Pred"])
plt.plot([test_dataset1["AI_Log"].min(), test_dataset1["AI_Log"].max()], 
         [test_dataset1["AI_Log"].min(), test_dataset1["AI_Log"].max()], 'r--')
plt.title("True vs Predicted AI_Log")
plt.xlabel("True"), plt.ylabel("Predicted")
plt.show()

# Plot the logs against depth time
y = -(dataset['TWT'])
y1 = -(test_dataset1['TWT'])
plt.figure(figsize=(16, 8))
plt.subplot(1, 7, 1)
plt.scatter(test_dataset1["AI_Log_Pred"], y1)
plt.plot(dataset["AI_Log"], y, 'red')
plt.title("Amplitude")
plt.ylabel("Two Way Time(ms)", fontsize=12)

plt.show()
j=1
##################
plt.figure(figsize=(16,12) )
for i in range(8):
     wno=i+1
     well_dataset=dataset[dataset["WELL"] == wno]
     well_data = dataset[dataset["WELL"] == wno][features]
     #well_predictions = rf_model.predict(well_data)
     well_data = scaler.transform(well_data)
     model.eval()
     with torch.no_grad():
        well_inputs = torch.tensor(well_data, dtype=torch.float32).to(device)
        well_predictions = model(well_inputs).cpu().numpy()


     well_dataset["AI_Log_Pred"] = well_predictions[:,0]
     well_dataset["RT_Pred"] = well_predictions[:,1]
     well_dataset["RHOB_Pred"] = well_predictions[:,2]
     well_dataset["NPHI_Pred"] = well_predictions[:,3]
     # Plot the logs against depth time

     y1 = -(well_dataset['TWT'] )



     plt.subplot(1,50,j+wno)
     plt.plot(well_dataset["AI_Log_Pred"], y1, 'blue')
     plt.plot(well_dataset["AI_Log"], y1, 'red')
     plt.title("Amplitude")
     plt.ylabel("Two Way Time(ms)", fontsize = 12)

     plt.subplot(1,50,wno+(1+j))
     plt.plot(well_dataset["RT_Pred"], y1, 'blue')
     plt.plot(well_dataset["RT"], y1, 'red')
     plt.title("Amplitude")
     plt.ylabel("Two Way Time(ms)", fontsize = 12)

     plt.subplot(1,50,wno+(2+j))
     plt.plot(well_dataset["RHOB_Pred"], y1, 'blue')
     plt.plot(well_dataset["RHOB"], y1, 'red')
     plt.title("Amplitude")
     plt.ylabel("Two Way Time(ms)", fontsize = 12)

     plt.subplot(1,50,wno+(3+j))
     plt.plot(well_dataset["NPHI_Pred"], y1, 'blue')
     plt.plot(well_dataset["NPHI"], y1, 'red')
     plt.title("Amplitude")
     plt.ylabel("Two Way Time(ms)", fontsize = 12)


     j=j+5
plt.show()
