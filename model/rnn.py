"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers

from utils.misc import cellModule, split_rnn_outputs, compute_budget_loss

#from rnn_cells.custom_cells import CSkipLSTMCell, CMultiSkipLSTMCell÷




class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion2 = torch.nn.BCELoss(size_average=True)
        self.criterion2 = nn.CrossEntropyLoss(weight=torch.Tensor([1.0,1.0]).cuda())
        self.criterion3 = nn.NLLLoss()
        self.mse = nn.MSELoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.att_w = torch.eye(len(constant.LABEL_TO_ID)).cuda()
        # self.att_w[0][0] = 1
        self.epoch = 0
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [Variable(b.cuda()) for b in batch[:7]]
            all_labels = Variable(batch[7].cuda())
            true_label = Variable(batch[8].cuda())
            current_idx = Variable(batch[9].cuda())
            nona_idx = Variable(batch[10].cuda())
            na_idxs = Variable(batch[11].cuda())
            one_hot_label = Variable(batch[12].cuda()).float()
        else:
            inputs = [Variable(b) for b in batch[:7]]
            all_labels = Variable(batch[7])
            true_label = Variable(batch[8])
            current_idx = Variable(batch[9])
            nona_idx = Variable(batch[10])
            na_idxs = Variable(batch[11])
            one_hot_label = Variable(batch[12]).float()

        na_labels = torch.zeros_like(all_labels).cuda()

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        
            
        if self.opt['crf']:
            pred_logits, final_hidden, outputs, hidden,pen,s_prob = self.model(inputs,current_idx,all_labels)
            loss = self.criterion(pred_logits, true_label)
            loss += pen
                    
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_train = loss.data.item()
        return loss_train

    def predict(self, batch, test):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [Variable(b.cuda()) for b in batch[:7]]
            all_labels = Variable(batch[7].cuda())
            true_label = Variable(batch[8].cuda())
            current_idx = Variable(batch[9].cuda())
            nona_idx = Variable(batch[10].cuda())
            na_idxs = Variable(batch[11].cuda())
            one_hot_label = Variable(batch[12].cuda()).float()
            sens = batch[13]
            ids = batch[14]

        else:
            inputs = [Variable(b) for b in batch[:7]]
            all_labels = Variable(batch[7])
            true_label = Variable(batch[8])
            current_idx = Variable(batch[9])
            nona_idx = Variable(batch[10])
            na_idxs = Variable(batch[11])
            one_hot_label = Variable(batch[12]).float()


        orig_idx = batch[-1]
        # forward
        self.model.eval()    
        if self.opt['crf']:
            logits, _, o, h1, pen,s_prob = self.model(inputs,current_idx,all_labels)
            probs = F.softmax(logits).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                predictions, probs)))]
        return predictions, probs

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def merge_prediction(self, prob_42, prob_2):
        probs = []
        for batch_id in range(prob_42.shape[0]):
            prob = [prob_2[batch_id][1]*prob_42[batch_id][0]]
            for i in range(1,prob_42.shape[1]):
                prob.append(prob_42[batch_id][i]*prob_2[batch_id][0])
            probs.append(prob)
        return np.array(probs)

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.opt = opt
        self.drop = nn.Dropout(opt['dropout'])
        self.drop2 = nn.Dropout(0.2)
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.new_emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] +opt['pe_dim']

        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])
        
        if opt['attn']:
            if self.opt['crf']:
                self.attn_layer = layers.PositionCRFBatch(opt)   
                self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])
                input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] +opt['pe_dim']
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])
        
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.mul_linear = nn.Linear(opt['emb_dim'], 50)
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        
        self.linear.bias.data.fill_(0)
        init.xavier_uniform(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        self.new_emb.weight.data.copy_(self.emb.weight.data)
        self.mul_linear.bias.data.fill_(0)
        init.xavier_uniform(self.mul_linear.weight, gain=1) # initialize linear layer

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")
    def zero_state(self, batch_size): 
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    
    def forward(self, inputs, current_idx, relation_id):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        entity = words[:,:2]
        words = words[:,2:]
        entity_pos = pos[:,:2]
        pos = pos[:,2:]
        entity_ner = ner[:,:2]
        ner = ner[:,2:]        
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]




        subj_pos_binary = (subj_pos==0)
        obj_pos_binary = (obj_pos==0)

        pe_binary = obj_pos_binary+subj_pos_binary
        #print(pe_binary)
        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)

        if self.opt['crf']:
            pe_inputs = self.pe_emb(pe_binary.long())
            inputs += [pe_inputs]

        inputs = torch.cat(inputs, dim=2) # add dropout to input
        inputs = self.drop(inputs)
        input_size = inputs.size(2)
        
        #else:
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)
        
        # attention
        if self.opt['attn']:
            if self.opt['crf']:
                final_hidden, pen, s_prob = self.attn_layer(outputs, masks, subj_pos, obj_pos, seq_lens,pe_inputs,pe_binary)                 
        logits = self.linear(final_hidden)
        if self.opt['crf']:
            return logits, final_hidden, outputs, hidden,pen,s_prob

    

