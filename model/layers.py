"""
Additional layers.
"""
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from utils import constant, torch_utils
from torch.nn.parameter import Parameter
from model.CRF import LinearCRF
from model.BatchLinearChainCRF import LinearChainCrf

class LSTMLayer(nn.Module):
    """ A wrapper for LSTM with sequence packing. """

    def __init__(self, emb_dim, hidden_dim, num_layers, dropout, use_cuda):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.use_cuda = use_cuda

    def forward(self, x, x_mask, init_state):
        """
        x: batch_size * feature_size * seq_len
        x_mask : batch_size * seq_len
        """
        x_lens = x_mask.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lens = list(x_lens[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        
        # sort by seq lens
        x = x.index_select(0, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        rnn_output, (ht, ct) = self.rnn(rnn_input, init_state)
        rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)[0]
        
        # unsort
        rnn_output = rnn_output.index_select(0, idx_unsort)
        ht = ht.index_select(0, idx_unsort)
        ct = ct.index_select(0, idx_unsort)
        return rnn_output, (ht, ct)

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    
    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.epoch = 0
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
    
    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs, weights








class PositionCRFBatch(nn.Module):
    def __init__(self, opt):
        '''
        LSTM+Aspect
        '''
        super(PositionCRFBatch, self).__init__()

        self.C1 = opt['C1']
        self.C2 = opt['C2']
        self.inter_crf = LinearChainCrf(2+2)



        self.input_size = opt['hidden_dim']
        
        
        self.feat2tri = nn.Linear(self.input_size, 2+2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.feat2tri.weight)
        nn.init.constant(self.feat2tri.bias, 0.0)

        

    def compute_scores(self, context, masks, lens, pos_relevance, is_training=True):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        
        batch_size, max_len, hidden_dim = context.size()


        
        ###neural features
        feats = self.feat2tri(context) #Batch_size*sent_len*2

        
        # word_mask = torch.full((batch_size, max_len), 0)
        # for i in range(batch_size):
        #     word_mask[i, :lens[i]] = 1.0

        marginals = self.inter_crf.compute_marginal(feats, masks.type_as(feats))
        #print(word_mask.sum(1))
        # print(marginals[0].shape)
        # .mul(pos_relevance[i, :lens[i]])
             
        select_polarities = [marginal[:, 1]  for i,marginal in enumerate(marginals)]
        # feature = torch.zeros_like(context)
        # for i, sp in enumerate(select_polarities):
        #     #gamma = sp.sum()/2
        #     feature[i, :lens[i], :] = (sp).unsqueeze(1).repeat(1,self.input_size)
        # x = context.unsqueeze(1)
        # x = [F.tanh(conv(x)).squeeze(3) for conv in self.convs]
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # sent_vs = torch.cat(x, 1)
        gammas = [sp.sum()/2 for sp in select_polarities]
        sent_vs = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities)]
        sent_vs = torch.cat([sv/gamma for sv, gamma in zip(sent_vs, gammas)],0)#normalization
        best_latent_seqs = self.inter_crf.decode(feats, masks.type_as(feats))
        #print(best_latent_seqs)
        for i, sp in enumerate(select_polarities):
            select_polarities[i] = select_polarities[i]/gammas[i]
        if is_training:
            return sent_vs, select_polarities
        else:
            best_latent_seqs = self.inter_crf.decode(feats, word_mask.type_as(feats))
            return sent_vs, best_latent_seqs



    
    def forward(self, hidden, hidden_mask, subj_position, obj_position, seq_lens, pe_inputs, pe_binary):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentencesï¼Œ batch_size*len*emb_dim
        mask: a list of mask for each sentence, batch_size*len
        label: a list labels
        '''

        #scores: batch_size*label_size
        #s_prob:batch_size*sent_len

        # if self.config.if_reset:  self.cat_layer.reset_binary()

        subj_position = torch.abs(subj_position.float())
        obj_position = torch.abs(obj_position.float())
        hidden_mask_0 = hidden_mask.eq(0).float()

        subj_relevance = torch.clamp(1-subj_position.float()/5, min=0)
        obj_relevance = torch.clamp(1-obj_position.float()/5,min=0)
        relevance = torch.max(subj_relevance,obj_relevance).mul(hidden_mask_0)
        # relevance = relevance.unsqueeze(2).repeat(1,1,self.input_size)

        feats = hidden

        #feats = torch.cat([hidden,pe_inputs],2)


        mask = (hidden_mask==0)
        sent_vs, s_prob  = self.compute_scores(feats, mask, seq_lens, relevance)
        
        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
            F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
        norm_pen = self.C1 * pena + self.C2 * s_prob_norm 
        
        #print('Transition Penalty:', pena)
        #print('Marginal Penalty:', s_prob_norm)
        


        return sent_vs, norm_pen, s_prob 

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks)
        scores, best_seqs = self.compute_scores(sents, masks, sent_lens, False)
        _, pred_label = scores.max(1)    
        
        #Modified by Richard Sun
        return pred_label, best_seqs


        