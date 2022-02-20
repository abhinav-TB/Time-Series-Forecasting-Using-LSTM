import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from utils import StockDataset,train_loop,test_loop,calculate_metrics
from model import Lstm_model

# model parameters
batch_size = 64
input_dim = 1 
hidden_size = 50
num_layers = 3
epochs = 100

df = pd.read_csv('AAPL.csv')
df1 = df.reset_index()['close']

# data preprocessing

# normalising the dataset to values bewtween 0 and 1
scalar = MinMaxScaler(feature_range=(0,1))
df1 = scalar.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split for time series data
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

#creating datasets
train_dataset = StockDataset(train_data) 
test_dataset = StockDataset(test_data) 

#creating dataloaders
train_dataloader = DataLoader(train_dataset,batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size , drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

#initialising model
model = Lstm_model(input_dim , hidden_size , num_layers,batch_size).to(device)

#initialising optimizer and loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#training the model
for epoch in range(epochs):
    print(f"epoch {epoch} ")
    train_loop(train_dataloader,model,device,loss_fn,optimizer,batch_size)
    test_loop(test_dataloader,model,device,loss_fn,batch_size)

# calculating final loss metrics
print(f"train mse loss {calculate_metrics(train_dataloader,model,device,loss_fn,batch_size,scalar)}")
print(f"test mse loss {calculate_metrics(test_dataloader,model,device,loss_fn,batch_size,scalar)}")