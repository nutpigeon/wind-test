#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from WindPy import w
w.start()

def process_future_contract(future_code,exc):
    data_frames = []

    for i in range(12):  # 尝试处理最多12个月份的合约
        contract_code = f"{future_code}{i:02d}.{exc}"
        #{i:02d}: 这是一个占位符，用于插入变量i的值。i是一个整数，而02d是一个格式说明符，意味着：0表示用零填充。2表示总共显示两位数。d表示十进制整数。
        #因此，如果i的值是3，它会被格式化为"03"。如果i的值是10，则保持不变为"10"。
        data = w.wsd(contract_code, "open,high,low,close,volume,oi", "ED-800TD", "2024-03-19", "")
        
        if data.ErrorCode == 0:
            df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
            df['CONTRACT'] = contract_code
            data_frames.append(df)
        else:
            print(f"Error or no data for {contract_code}: {data.ErrorCode}")
            continue  # 如果数据有误或合约不存在，跳过当前合约

    if not data_frames:
        return pd.DataFrame()  # 如果没有获取到任何合约数据，则返回空DataFrame

    # 合并所有合约的DataFrame
    combined_df = pd.concat(data_frames)

    # 计算加权平均和总和
    weighted_df = pd.DataFrame(index=combined_df.index.unique())
    for column in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
        weighted_values = combined_df.groupby(combined_df.index).apply(
            lambda x: (x[column] * x['OI']).sum() / x['OI'].sum()
        )
        weighted_df[column] = weighted_values

    for column in ['VOLUME', 'OI']:
        total_values = combined_df.groupby(combined_df.index)[column].sum()
        weighted_df[column] = total_values

    return weighted_df

future_codes = ["A.DCE", "AG.SHF","AU.SHF","AL.SHF","C.DCE","CU.SHF","HC.SHF","NI.SHF","RB.SHF","SA.CZC","SP.SHF","SR.CZC","TA.CZC","Y.DCE","ZN.SHF","I.DCE","AU.SHF","AP.CZC","EB.DCE","EG.DCE","FG.CZC"]  # 已有的代码列表

for code in future_codes:
    base_code, exc = code.split('.')  # 分割字符串以获取基础代码和交易所代码
    weighted_df = process_future_contract(base_code, exc)
    # 将加权平均数据保存到文件以便以后使用
    weighted_df.to_csv(f"{code}_weighted.csv")


# In[5]:


import pandas as pd
future_codes = ["A.DCE", "AG.SHF","AL.SHF","C.DCE","CU.SHF","HC.SHF","NI.SHF","RB.SHF","SA.CZC","SP.SHF","SR.CZC","TA.CZC","Y.DCE","ZN.SHF","I.DCE"]
def label_data(df):
    # 确保数据按日期升序排序
    df = df.sort_index(ascending=True)

    # 创建两个新列，一个用于存储标签，另一个用于存储变动率
    df['label'] = 'do nothing'
    df['change_percentage'] = 0.0  # 初始化变动率列

    # 遍历DataFrame，但留出最后5行
    for i in range(len(df) - 5):
        current_close = df.iloc[i]['CLOSE']
        future_close = df.iloc[i + 5]['CLOSE']

        # 计算价格变化百分比
        change_percentage = (future_close - current_close) / current_close * 100
        df.at[df.index[i], 'change_percentage'] = change_percentage  # 添加变动率到DataFrame

        # 根据变化百分比打标签
        if change_percentage > 2:
            df.at[df.index[i], 'label'] = 'buy'
        elif change_percentage < -2:
            df.at[df.index[i], 'label'] = 'sell'

    return df



# In[6]:


for code in future_codes:
    file_path = f"{code}_weighted.csv"
    weighted_df = pd.read_csv(file_path, index_col=0)
    labeled_df = label_data(weighted_df)
    labeled_df.to_csv(f"{code}_labeled.csv")  # 保存带标签的数据


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# 初始化一个空字典来存储每个合约的标签分布
label_distributions = {}

# 遍历每个合约代码，加载数据，计算标签分布
for code in future_codes:
    file_path = f"{code}_labeled.csv"  # 假设标签数据保存在这些文件中
    data = pd.read_csv(file_path)
    label_distribution = data['label'].value_counts(normalize=True)  # 计算标签的分布比例
    label_distributions[code] = label_distribution

# 创建一个新的DataFrame来存储所有合约的标签分布数据
df_distributions = pd.DataFrame(label_distributions).T.fillna(0)

# 绘制条形图
df_distributions.plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title('Label Distribution Across Different Contracts')
plt.ylabel('Proportion')
plt.xlabel('Contracts')
plt.legend(title='Labels')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 初始化存储容器
scalers = {}  # 用于存储每个品种的scaler
all_X_train = []
all_X_test = []
all_y_train = []
all_y_test = []

for code in future_codes:
    file_path = f"{code}_labeled.csv"
    data = pd.read_csv(file_path)

    # 转换第一列为日期格式并按日期排序
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    data = data.sort_values(data.columns[0])

    # 添加品种代码作为特征
    data['Variety'] = code

    lookback_days = 200
    features = []
    labels = []

    # 选择除了标签、变化百分比和日期之外的所有列作为特征
    feature_columns = data.columns[1:-3]  # 排除了日期、标签和变化百分比列

    # 创建输入-输出对
    for i in range(lookback_days, len(data)):
        window = data.iloc[i - lookback_days:i][feature_columns].values
        label = data.iloc[i, -3]  # 假设标签在倒数第二列
        features.append(window)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    # 初始化并应用StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    scalers[code] = scaler  # 保存scaler
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    all_X_train.append(X_train)
    all_X_test.append(X_test)
    all_y_train.append(y_train)
    all_y_test.append(y_test)

# 使用np.concatenate来合并数据
X_train_combined = np.concatenate(all_X_train, axis=0)
X_test_combined = np.concatenate(all_X_test, axis=0)
y_train_combined = np.concatenate(all_y_train, axis=0)
y_test_combined = np.concatenate(all_y_test, axis=0)


# In[ ]:


# Check the shapes of the resulting arrays
X_train_combined.shape, X_test_combined.shape, y_train_combined.shape, y_test_combined.shape


# In[ ]:


# 显示经过预处理和合并后的训练数据集中的一个样本
X_train_combined_sample = y_train_combined[0]

X_train_combined_sample


# 数据已经被标准化并划分为训练集和测试集了。训练集包含3920个样本，测试集包含1694个样本。每个样本都是一个200x6的矩阵，对应于过去200天的6个特征值。

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit label encoder and return encoded labels
y_train_encoded = label_encoder.fit_transform(y_train_combined)
y_test_encoded = label_encoder.transform(y_test_combined)
                                        

# Convert the encoded training and test labels into PyTorch tensors
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)


# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# In[ ]:


print(f"Training samples: {len(train_dataset)}, Batches: {len(train_loader)}")
print(f"Testing samples: {len(test_dataset)}, Batches: {len(test_loader)}")


# In[ ]:


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=2, padding=2)#想实现一些一阶差分的处理所以这里kernel选了2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.fc1 = nn.Linear(896, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Determine the number of features and classes
num_features = X_train.shape[2]
num_classes = len(np.unique(y))


# Initialize the CNN
model = CNN(num_features=X_train.shape[2], num_classes=len(np.unique(y_train_encoded)))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model, train_loader, test_loader


# In[ ]:


num_epochs = 64  # 定义要训练的epoch数量

# 设置模型为训练模式
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data, targets in train_loader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(data)  # 前向传播
        if outputs.shape[0] != targets.shape[0]:
            print(f"Mismatch! Output batch size: {outputs.shape[0]}, Target batch size: {targets.shape[0]}")
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        
        epoch_loss += loss.item() * data.size(0)  # 累加损失

    # 计算平均损失
    epoch_loss /= len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 评估模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():  # 在评估阶段不计算梯度
    for data, targets in test_loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy:.4f}')


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# 确保模型处于评估模式
model.eval()

# 收集所有预测和标签
all_preds = []
all_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_targets, all_preds)
#  `label_encoder` 已经拟合了所有原始的类别标签
original_classes = label_encoder.inverse_transform([0, 1, 2])

# 创建一个映射整数标签到原始标签的字典
label_mapping = {i: original_classes[i] for i in range(len(original_classes))}

# 更新混淆矩阵的轴标签
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




