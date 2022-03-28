import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
warnings.filterwarnings("ignore")
from Dataloader import *
from mould.STGHTN import STGHTN
from utils import metric
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS08", help="the datasets name")
parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=12, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=12, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=8, help="Number of training batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--in_dim', type=float, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=float, default=64, help="Embed_size")
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

    test_data = TensorDataset(testX,testY)
    test_dataloader = DataLoader(test_data,batch_size=args.batchsize)
    model = STGHTN(STadj,args.embed_size,num_nodes,[args.embed_size,args.embed_size,args.embed_size,args.embed_size,args.embed_size],args.heads)
    model.load_state_dict(torch.load(""))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mae = 100
    best_rmse = None
    best_mape = None
    model.eval()
    P = []
    L = []
    for x, y in test_dataloader:
        x = x.to(device)
        pre = model(x) * std + mean
        P.append(pre.cpu().detach())
        L.append(y)
    pre = torch.cat(P, 0)
    label = torch.cat(L, 0)

    pre = pre.reshape(-1, adj.shape[0])
    label = label.reshape(-1, adj.shape[0])

    mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
    print("rmse,mae,mape,wape:", rmse, mae, mape, wape)