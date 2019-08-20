"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict
import copy
class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4) # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret

def tree_to_seq(tree, tokens):
    """
    Convert a tree object to a sequence.
    """

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        queue += t.children

    idx = sorted(idx)
    pruned_tokens = [tokens[i] for i in idx]
    return idx

def get_path_to_root(head, pos):
    cas = None
    ancestors = set(pos)
    for s in pos:
        h = head[s]
        tmp = [s]
        while h > 0:
            tmp += [h-1]
            ancestors.add(h-1)
            h = head[h-1]
        if cas is None:
            cas = set(tmp)
        else:
            cas.intersection_update(tmp)
    return ancestors, cas

def get_lca(cas, head):
    # find lowest common ancestor
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k:0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1

        # the LCA has no child in the CA set
        for ca in cas:
            if child_count[ca] == 0:
                lca = ca
                break
    return lca


def get_shortest_hops(head, subj_pos, obj_pos, token_idxs):

    subj_pos = [i for i in range(len(subj_pos)) if subj_pos[i] == 0]
    obj_pos = [i for i in range(len(obj_pos)) if obj_pos[i] == 0]
    


    subj_ancestors, subj_cas = get_path_to_root(head, subj_pos)
    obj_ancestors, obj_cas = get_path_to_root(head, obj_pos)
    # print(subj_cas)
    # print(subj_ancestors)
    # print(obj_cas)

    subj_hops = []; obj_hops = [];

    for token_idx in token_idxs:
        local_subj_cas = copy.deepcopy(subj_cas)
        local_obj_cas = copy.deepcopy(obj_cas)
        token_ancestors, token_cas = get_path_to_root(head, [token_idx])

        local_subj_cas.intersection_update(token_cas)
        token_subj_lca = get_lca(local_subj_cas,head)
        local_obj_cas.intersection_update(token_cas)
        # if token_idx ==6:
        #     print(local_subj_cas)
        token_obj_lca = get_lca(local_obj_cas,head)
        subj_path_nodes = subj_ancestors.union(token_ancestors).difference(local_subj_cas)
        subj_path_nodes.add(token_subj_lca)
        subj_path_nodes = subj_path_nodes.difference(set(subj_pos))
        # if token_idx ==6:
        #     print(token_cas)
        #     print(token_ancestors)
        #     print(token_subj_lca)
        #     print(subj_path_nodes)
        obj_path_nodes = obj_ancestors.union(token_ancestors).difference(local_obj_cas)
        obj_path_nodes.add(token_obj_lca)
        obj_path_nodes = obj_path_nodes.difference(set(obj_pos))
        if subj_path_nodes:
            subj_hops.append(len(subj_path_nodes)) 
        else:
            subj_hops.append(0)

        if obj_path_nodes:
            obj_hops.append(len(obj_path_nodes)) 
        else:
            obj_hops.append(0)
                
    return subj_hops, obj_hops

def head_to_tree_v2(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)
        # path_nodes =

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4) # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def get_sdp(head, tokens, len_, subj_pos, obj_pos):
    stp = head_to_tree(head, np.array(tokens), len_, 0, subj_pos, obj_pos)
    stp_seq = tree_to_seq(stp,tokens)
    child_node = head_to_tree_v2(head, np.array(tokens), len_, 1, subj_pos, obj_pos)
    child_seq = tree_to_seq(child_node,tokens)

    # stp_la_idx = head[stp.idx]
    # if stp_la_idx > 0:
    #     if stp_la_idx not in stp_seq:
    #         stp_seq.append(stp_la_idx-1)
    stp_seq = set(stp_seq).union(set(child_seq))
    sdp_seq_idx = sorted(stp_seq)
    pruned_tokens = [tokens[i] for i in sdp_seq_idx]
    # print(pruned_tokens)
    return sdp_seq_idx








    # path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    # path_nodes.add(lca)
    # return len(path_nodes)


if __name__ == "__main__":
    head = [7, 3, 7, 7, 7, 7, 26, 10, 10, 7, 14, 13, 14, 26, 14, 26, 26, 26, 22, 22, 22, 26, 26, 26, 26, 0, 26, 26]
    tokens = ['While', 'his', 'brother', 'was', 'the', 'famous', 'SUBJ-PERSON', 'at', 'that', 'time', ',', 'the', 'OBJ-ORGANIZATION', 'said', ',', '``', 'nowadays', ',', '-LRB-', 'Dominick', '-RRB-', 'Dunne', 'is', 'far', 'better', 'known', '.', "''"]
    # print(tokens[head[16]-1])
    subj_positions = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    obj_positions = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # print(tokens.index('in'))
    # subj_positions,obj_positions  = get_shortest_hops(np.array(head), np.array(subj_positions), np.array(obj_positions), tokens_idxs)
    print(get_sdp(np.array(head), tokens, len(tokens), np.array(subj_positions), np.array(obj_positions)))
    # print(subj_positions)
    # print(obj_positions)