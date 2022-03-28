import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order



    def forward(self,x,support):
        x = x.permute(0, 3, 1, 2)
        out = [x]
        #support = [support]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h.permute(0,2,3,1)



class STBlk(nn.Module):
    def __init__(self,STadj,embedsize,nodes,num_channels,heads,dropout=0):
        super(STBlk, self).__init__()
        self.nodes = nodes
        self.tcn1 = TemporalConvNet(num_inputs=embedsize, num_channels=num_channels, kernel_size=3,dropout=dropout)
        self.tcn2 = TemporalConvNet(num_inputs=embedsize, num_channels=num_channels, kernel_size=2, dropout=dropout)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, nodes, 12, embedsize))

        self.STadj = STadj
        self.gcn1 = gcn(embedsize,embedsize,dropout,support_len=2)
        self.gcn2 = gcn(embedsize, embedsize, dropout, support_len=2)
        self.gcn3 = gcn(embedsize, embedsize, dropout, support_len=1)


        self.SAT = SMultiHeadAttention(embedsize, heads)

        self.TAT = TMultiHeadAttention(embedsize, heads)


        self.relu = nn.ReLU()

        self.feed_forward = nn.Sequential(
            nn.Linear(embedsize, 4 * embedsize),
            nn.ReLU(),
            nn.Linear(4 * embedsize, embedsize),
        )

        self.feed_forward1 = nn.Sequential(
            nn.Linear(embedsize, 4 * embedsize),
            nn.ReLU(),
            nn.Linear(4 * embedsize, embedsize),
        )





        self.norm1 = nn.LayerNorm(embedsize)
        self.norm2 = nn.LayerNorm(embedsize)

        self.norm3 = nn.LayerNorm(embedsize)
        self.norm4 = nn.LayerNorm(embedsize)

        self.norm5 = nn.LayerNorm(embedsize)
        self.norm6 = nn.LayerNorm(embedsize)

        self.tcngate = nn.Sequential(
            nn.Linear(embedsize, embedsize),
            nn.Sigmoid()
        )
        self.gcngate = nn.Sequential(
            nn.Linear(embedsize, embedsize),
            nn.Sigmoid()
        )


        self.dropout = nn.Dropout(dropout)


        self.fc1 = nn.Sequential(
            nn.Linear(embedsize,embedsize),
            nn.Linear(embedsize, embedsize),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embedsize,embedsize),
            nn.Linear(embedsize, embedsize),
        )

    def forward(self,x_T,x_S,Adj):

        res_T = x_T
        res_S = x_S
        x_tcn = x_T
        x_gcn = x_S


        B,N,T,H = x_T.shape

        x_T_Q = x_S
        x_T_K = x_T
        x_T_V = x_T


        x_S_Q = x_T
        x_S_K = x_S
        x_S_V = x_S

        ########################时间

        x_T = self.TAT(x_T_Q,x_T_K,x_T_V)
        x_tcn = x_tcn.reshape(B,N*T,H).permute(0,2,1)
        x_tcn =torch.tanh(self.tcn1(x_tcn).permute(0,2,1).reshape(B,N,T,H))*torch.sigmoid(self.tcn2(x_tcn).permute(0,2,1).reshape(B,N,T,H))
        x_T = x_T*self.tcngate(x_tcn)
        x_T = self.dropout(self.norm1(x_T+res_T))
        x_feed = self.feed_forward(x_T)
        x_T = self.dropout(self.norm2(x_feed+x_T))


        ########################空间



        x_S = self.SAT(x_S_Q,x_S_K,x_S_V)

        x_gcn_F = torch.tanh(self.gcn1(x_gcn, Adj[0:2])) * torch.sigmoid(self.gcn2(x_gcn, Adj[0:2]))
        x_gcn_D = self.gcn3(x_gcn, Adj[2:3])
        g = torch.sigmoid(self.fc1(x_gcn_F) + self.fc2(x_gcn_D))
        x_gcn = g * x_gcn_F + (1 - g) * x_gcn_D

        x_S = x_S*self.gcngate(x_gcn)
        x_S= self.dropout(self.norm3(x_S+res_S))
        x_feed = self.feed_forward1(x_S)
        x_S = self.dropout(self.norm4(x_feed+x_S))


        return x_T,x_S


class STGHTN(nn.Module):
    def __init__(self,STadj,embedsize,nodes,num_channels,heads,dropout=0):
        super(STGHTN, self).__init__()
        self.nodes = nodes
        self.STadj = STadj
        self.ST1 = STBlk(STadj, embedsize, nodes, num_channels, heads, dropout)
        self.ST2 = STBlk(STadj, embedsize, nodes, num_channels, heads, dropout)
        self.ST3 = STBlk(STadj, embedsize, nodes, num_channels, heads, dropout)
        self.ST4 = STBlk(STadj, embedsize, nodes, num_channels, heads, dropout)

        self.conv1 = nn.Conv2d(1, embedsize, 1)
        self.conv2 = nn.Conv2d(12, 12, 1)
        self.conv3 = nn.Conv2d(embedsize, 1, 1)
        self.norm1 = nn.LayerNorm(embedsize)
        self.norm2 = nn.LayerNorm(embedsize)
        self.norm3 = nn.LayerNorm(embedsize)
        # self.norm2 = nn.LayerNorm(12)
        self.relu = nn.ReLU()

        self.fs = nn.Linear(embedsize, embedsize)
        self.fg = nn.Linear(embedsize, embedsize)

        self.fs1 = nn.Linear(embedsize, embedsize)
        self.fg1 = nn.Linear(embedsize, embedsize)


        self.pos_embed = nn.Parameter(torch.zeros(1, nodes, 12, embedsize), requires_grad=True)
        self.Linear1 = nn.Sequential(
            nn.Linear(embedsize,embedsize),
            nn.Linear(embedsize,1),
            nn.LeakyReLU(),
        )
        self.Linear2 = nn.Sequential(
        nn.Linear(embedsize, embedsize),
        nn.Linear(embedsize, embedsize),
        nn.ReLU(),
        )
        self.Linear3 = nn.Sequential(
        nn.Linear(embedsize, embedsize//2),
        nn.Linear(embedsize//2, 1),
        nn.ReLU(),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedsize, 4 * embedsize),
            nn.ReLU(),
            nn.Linear(4 * embedsize, embedsize),
        )

        self.feed_forward1 = nn.Sequential(
            nn.Linear(embedsize, 4 * embedsize),
            nn.ReLU(),
            nn.Linear(4 * embedsize, embedsize),
        )
        m, p, n = torch.svd(STadj[1])
        initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
    def forward(self, x):  #B T N
        x = x.permute(0, 2, 1).unsqueeze(1)
        input = self.conv1(x)
        input = input.permute(0, 2, 3, 1)
        input = input+self.pos_embed
        T_SKIP = 0
        S_SKIP = 0
        B,N,T,H = input.shape

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)*(self.STadj[0]+self.STadj[1])
        #print(adp)
        Adj = self.STadj+[adp]
        #Adj = self.STadj
        new_Adj = []
        for i in range(len(Adj)):
            if i<2:
                A_H = Adj[i]+torch.eye(self.nodes).to(device)
            else:
                A_H = Adj[i]
            D = (A_H.sum(-1)**-0.5)
            D[torch.isinf(D)]=0.
            D = torch.diag_embed(D)
            A = torch.matmul(torch.matmul(D, A_H), D)
            new_Adj.append(A)
        #print(new_Adj[-1])
        ###############################################
        x_T, x_S = self.ST1(input, input,new_Adj)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST2(x_T, x_S,new_Adj)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST3(x_T, x_S,new_Adj)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST4(x_T, x_S,new_Adj)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP


        feed_T = self.feed_forward(T_SKIP)
        T_SKIP = self.norm1(feed_T+T_SKIP)

        feed_S = self.feed_forward1(S_SKIP)
        S_SKIP = self.norm2(feed_S+S_SKIP)
        x = T_SKIP + S_SKIP
        #################################################
        out = x.permute(0, 2, 1, 3)
        out = self.relu(self.conv2(out))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.squeeze(1)
        # print(out)
        return out.permute(0, 2, 1)





