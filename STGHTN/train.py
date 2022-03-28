import os
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from Dataloader import *
from utils import metric, huber_loss
from mould.STGHTN import STGHTN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS03", help="the datasets name")
parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=12, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=12, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=16, help="Number of training batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--in_dim', type=float, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=float, default=32, help="Embed_size")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
args = parser.parse_args()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, adj = load_data(args.dataset)
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    data1 =np.mat(data,dtype=np.float32)
    trainX, trainY, valX, valY, testX, testY, mean, std = preprocess_data(data1, time_len, args.train_rate,
                                                                          args.seq_len, args.pre_len)

    Sadj = torch.tensor(np.load(""), dtype=torch.float32).to(device)
    Tadj = torch.tensor(np.load(""), dtype=torch.float32).to(device) - torch.eye(num_nodes).to(device)
    STadj = [Sadj,Tadj]
    train_data = TensorDataset(trainX, trainY)
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_data = TensorDataset(valX, valY)
    val_dataloader = DataLoader(val_data, batch_size=args.batchsize)

    model = STGHTN(STadj,args.embed_size,num_nodes,[args.embed_size,args.embed_size,args.embed_size,args.embed_size,args.embed_size],args.heads)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mae = 100
    best_rmse = None
    best_mape = None
    for epoch in range(100):
      model.train()
      for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        pre = model(x)
        loss = huber_loss(pre*std+mean,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      model.eval()
      P = []
      L = []
      for x, y in val_dataloader:
          x = x.to(device)
          pre = model(x) * std + mean
          P.append(pre.cpu().detach())
          L.append(y)
      pre = torch.cat(P, 0)
      label = torch.cat(L, 0)

      pre = pre.reshape(-1, adj.shape[0])
      label = label.reshape(-1, adj.shape[0])

      mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
      print("rmse,mae,mape,wape:", epoch, rmse, mae, mape, wape)
      DATANAME = args.dataset
      t = datetime.datetime.now()
      dir = str(t)[0:10]
      file = str(t)[10:-7].replace(":", "-")
      if not os.path.exists(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir))):
          os.makedirs(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir)))
      torch.save(model.state_dict(), "Model/PEMS/{}/{}/epoch+{}+time{}.pkl".format(DATANAME, dir, epoch, file))

      with open("result/{}min_{}_mean_std.txt".format(args.pre_len * 5, DATANAME), "a+") as f:
          f.write("{}>>>>{} min pre：rmse:{},mae:{},mape:{}".format(str(t), args.pre_len * 5, rmse, mae,
                                                                   mape) + "\n")

      if mae < best_mae:
          best_rmse = rmse
          best_mae = mae
          best_mape = mape
          print("now best mae:>>>>>>>>>>>>>", best_rmse, best_mae, best_mape)
          with open("result/{}min_{}_mean_std.txt".format(args.pre_len * 5, DATANAME), "a+") as f:
              f.write(
                  "{}>>>>{} min pre：rmse:{},mae:{},mape:{}".format(str(t), args.pre_len * 5, best_rmse, best_mae,
                                                                   best_mape) + "\n")