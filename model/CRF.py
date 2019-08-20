from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import copy

def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim
    Returns:
        size=(batch_size, hidden_dim)
    """
    # print(vec)
    _, idx = torch.max(vec, 1)  # B * 1 * M
    # print(idx)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)



class LinearCRF(nn.Module):
    def __init__(self):
        super(LinearCRF, self).__init__()
        self.label_size = 2

       #T[i,j] for j to i, not i to j
        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))


    # feats: batch * sent_l * label_size



    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).
        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)
        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)

        init_alphas = torch.Tensor(seq_len + 1, batch_size, self.label_size).fill_(0).cuda()
        forward_var = Variable(init_alphas)

        mask = mask.transpose(1, 0).contiguous() # mask: size=(seq_len,batch_size)

        feats = feats.transpose(1, 0).contiguous()
        forward_var[1] = feats[0]
        
        scores = feats.view(seq_len, batch_size, 1, self.label_size).expand(seq_len, batch_size, self.label_size, self.label_size)
        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__() #(batch_size, tag_size, tag_size)
        except:
            _, inivalues = seq_iter.next()

        partition = feats[0].clone()#开始状态是start，第一个单词之后是对应状态的概率
        # i,j 从i到j的概率
        for idx, cur_values in seq_iter:
            # cur_values 两行，每行相同 [[0.1,0.2],[0.1,0.2]]
            # partition 两列，每列相同
            cur_values = cur_values + partition.contiguous().view(
                batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + self.transitions.view(
                1, self.label_size, self.label_size).expand(batch_size, self.label_size, self.label_size)#(batch_size,tag_size,tag_size) 全概率公式累乘得到的每个位置的概率
            cur_partition = log_sum_exp(cur_values, self.label_size) #(batch_size,tag_size) 每个位置的概率
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, self.label_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)

            if masked_cur_partition.size()[0] != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, self.label_size)
                partition.masked_scatter_(mask_idx, masked_cur_partition) #根据mask更新partition，如果当前词被mask，那么partition不更新
            
            forward_var[idx+1] = partition
        terminal_var = forward_var[-1].view(
                batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size)
        alpha = log_sum_exp(terminal_var, self.label_size)
        return alpha, forward_var[1:]




    def _backward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).
        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)
        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        init_betas = torch.Tensor(seq_len + 1, batch_size, self.label_size).fill_(0).cuda()
        backward_var = Variable(init_betas)
        mask = mask.transpose(1, 0).contiguous() # mask: size=(seq_len,batch_size)

        feats = feats.transpose(1, 0).contiguous()
        last_score = feats[-1].clone()
        scores = feats.view(seq_len, batch_size, 1, self.label_size).expand(seq_len, batch_size, self.label_size, self.label_size)
        partition = torch.Tensor(batch_size, self.label_size).fill_(0).cuda()
        mask_idx = mask[seq_len-1, :].view(batch_size, 1).expand(batch_size, self.label_size)
        masked_partition = last_score.masked_select(mask_idx)
        mask_idx = mask_idx.contiguous().view(batch_size, self.label_size)
        partition = partition.masked_scatter_(mask_idx, masked_partition)
        backward_var[seq_len-1] = partition  
       

        for idx in reversed(range(seq_len-1)):
            trans = self.transitions.t().view(
                1, self.label_size, self.label_size).expand(batch_size, self.label_size, self.label_size)
            
            cur_values = scores[idx] + partition.contiguous().view(
                batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + trans#(batch_size,tag_size,tag_size) 全概率公式累乘得到的每个位置的概率
            cur_partition = log_sum_exp(cur_values, self.label_size) #(batch_size,tag_size) 每个位置的概率
            
            neg_mask_idx = mask_idx==0
            masked_prev_value = feats[idx].masked_select(neg_mask_idx)
            if masked_prev_value.size()[0] != 0:
                cur_partition.masked_scatter_(neg_mask_idx, masked_prev_value)
            
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, self.label_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.size()[0] != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, self.label_size)
                partition.masked_scatter_(mask_idx, masked_cur_partition) #根据mask更新partition，如果当前词被mask，那么partition不更新
            backward_var[idx] = partition
        terminal_var = backward_var[0].view(
                batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size)
        beta = log_sum_exp(terminal_var, self.label_size)
        return beta, backward_var[:-1]



    def reset_transition(self):
        # self.transitions.data[0,0] = 0.5
        # self.transitions.data[1,1] = 1
        # self.transitions.data[0,1] = -0.5
        # self.transitions.data[1,0] = -0.5
        pass


    def forward(self, feats, mask):
        batch_size, sent_len, feat_dim = feats.size()
        Z1, forward_mat = self._forward_alg(feats, mask) 
        Z2, backward_mat = self._backward_alg(feats, mask)
        forward_v = forward_mat.transpose(1, 0).contiguous()
        backward_v = backward_mat.transpose(1, 0).contiguous()

        message_v = forward_v + backward_v - feats
        Z = Z1.view(batch_size,1,feat_dim).contiguous().expand(batch_size, sent_len, feat_dim)
        marginal_v = torch.exp(message_v - Z)

        return marginal_v
        
