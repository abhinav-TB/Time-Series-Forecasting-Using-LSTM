import math
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
class StockDataset(Dataset):
    def __init__(self,data,seq_len = 100):
        self.data = data
        self.data = torch.from_numpy(data).float().view(-1)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)-self.seq_len-1

    def __getitem__(self, index) :
        return  self.data[index : index+self.seq_len] , self.data[index+self.seq_len]


def train_loop(dataloader,model,device,loss_fn,optimizer,batch_size):
    hn , cn = model.init()
    hn , cn = hn.to(device) , cn.to(device)
    size = len(dataloader.dataset)
    model.train()
    for batch , item in enumerate(dataloader):
        x , y = item
        x = x.to(device)
        y = y.to(device)
        out , hn , cn = model(x.reshape(100,batch_size,1),hn,cn)
        loss = loss_fn(out.reshape(batch_size) , y)
        hn = hn.detach()
        cn = cn.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch == 10:
            loss, current = (loss.item()/y.sum().item())*100, batch 
            print(f"train loss: {loss:>7f} ")

def test_loop(dataloader,model,device,loss_fn,batch_size):
    hn , cn = model.init()
    hn , cn = hn.to(device) , cn.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    for batch , item in enumerate(dataloader):
        x , y = item
        x = x.to(device)
        y = y.to(device)
        out , hn , cn = model(x.reshape(100,batch_size,1),hn,cn)
        loss = loss_fn(out.reshape(batch_size) , y)
       
        if batch == 4:
            loss, current = (loss.item()/y.sum().item())*100, batch 
            print(f"test loss: {loss:>7f} ")

def calculate_metrics(data_loader,model,device,loss_fn,batch_size,scalar):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        hn , cn = model.init()
        hn , cn = hn.to(device) , cn.to(device)
        for batch , item in enumerate(data_loader):
            x , y = item
            x , y = x.to(device) , y.to(device)
            x = x.view(100,64,1)
            pred = model(x,hn,cn)[0]
            pred = scalar.inverse_transform(pred.detach().cpu().numpy()).reshape(-1)
            y = scalar.inverse_transform(y.detach().cpu().numpy().reshape(1,-1)).reshape(-1)
            # print(pr
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)
        return math.sqrt(mean_squared_error(y_arr,pred_arr))