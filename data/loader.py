"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab
from model.tree import head_to_tree, tree_to_seq, get_shortest_hops,get_sdp

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens_idxs = map_to_ids(tokens, vocab.word2id)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            
            relation = constant.LABEL_TO_ID[d['relation']]
            stp_tokens_idxs = tree_to_seq(head_to_tree(np.array(head), np.array(tokens), l, 0, np.array(subj_positions), np.array(obj_positions)), tokens)
            hop1_tokens_idxs = tree_to_seq(head_to_tree(np.array(head), np.array(tokens), l, 1, np.array(subj_positions), np.array(obj_positions)), tokens)
                
            stp_tokens_idxs, stp_pos, stp_ner, stp_deprel, stp_subj_positions, stp_obj_positions = get_path_input(tokens,pos,ner,deprel,stp_tokens_idxs,'SUBJ-'+d['subj_type'],'OBJ-'+d['obj_type'],vocab)
            hop1_tokens_idxs, hop1_pos, hop1_ner, hop1_deprel, hop1_subj_positions, hop1_obj_positions = get_path_input(tokens,pos,ner,deprel,hop1_tokens_idxs,'SUBJ-'+d['subj_type'],'OBJ-'+d['obj_type'],vocab)
            
            processed += [(tokens_idxs, pos, ner, deprel, subj_positions, obj_positions, relation,
                    stp_tokens_idxs, stp_pos, stp_ner, stp_deprel, stp_subj_positions, stp_obj_positions, relation,
                    hop1_tokens_idxs, hop1_pos, hop1_ner, hop1_deprel, hop1_subj_positions, hop1_obj_positions, relation)]
       
        return processed




    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = list(zip(*batch))
        batch_full = batch[:7]
        batch_stp = batch[7:14]
        batch_hop1 = batch[14:21]
        true_relation = torch.LongTensor(batch_full[-1])
        for i in range(len(batch_full)):
            batch_full[i] = batch_full[i]+batch_stp[i]+batch_hop1[i]
        batch = batch_full
        batch_size = len(batch[0])


        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        
        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        sen_list = map_to_tokens(words,self.vocab)
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words[:,2:], 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)

        # rels.fill_(1)


        current_idx = map_current_idx_to_orig(orig_idx)


        return (words, masks, pos, ner, deprel, subj_positions, obj_positions, true_relation, current_idx, sen_list, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_tokens(ids, vocab):
    sens_list = []
    for id_list in ids:
        sens_list.append(vocab.unmap(id_list[2:]))
    return sens_list



def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens




    

def get_index(tokens, target_word):
    idxs = [i for i, word in enumerate(tokens) if word == target_word]
    return idxs[0], idxs[-1]

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

def get_path_input(tokens, pos, ner, deprel, path_tokens_idxs, subj, obj, vocab):

    path_tokens = [tokens[i] for i in path_tokens_idxs]
    path_pos = [pos[i] for i in path_tokens_idxs]
    path_ner = [ner[i] for i in path_tokens_idxs]
    path_deprel = [deprel[i] for i in path_tokens_idxs]  
    path_subj_positions = get_positions(*get_index(path_tokens, subj), len(path_tokens))
    path_obj_positions = get_positions(*get_index(path_tokens, obj), len(path_tokens)) 
    subj_pos,_ = get_index(path_tokens, subj)
    obj_pos,_ = get_index(path_tokens, obj)
    path_tokens.insert(0,subj)
    path_tokens.insert(1,obj)
    path_pos.insert(0,pos[subj_pos])
    path_pos.insert(1,pos[obj_pos])
    path_ner.insert(0,ner[subj_pos])
    path_ner.insert(1,ner[obj_pos])    
    path_tokens_ids = map_to_ids(path_tokens, vocab.word2id)
    return path_tokens_ids, path_pos, path_ner, path_deprel, path_subj_positions, path_obj_positions
    

def map_current_idx_to_orig(orig_idx):
    
    assert len(orig_idx) % 3 == 0
    total = int(len(orig_idx)/3)
    current_idx = []
    for i in range(total):
        current_idx.append(orig_idx.index(i))
        current_idx.append(orig_idx.index(i+total))
        current_idx.append(orig_idx.index(i+2*total))
    return torch.LongTensor(current_idx)