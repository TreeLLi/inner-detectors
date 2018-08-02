'''
Data structure helper

To provide auxiliary functions easily handling data structures

'''


import numpy as np
from operator import itemgetter
from collections import Iterator


'''
Arithmatic

'''

def dictMean(dic, key):
    if isinstance(dic, list):
        return dic[key]
    elif not isinstance(dic, dict):
        return dic

    means = []
    for k, v in dic.items():
        _mean = dictMean(v, key)
        means.append(_mean)
    return np.mean(means)


'''
Sorting, Filter

'''

# result is returned as a list of tuples (key, val)
def sortDict(dic, by_key=False, indices=[], descending=True):
    # sorted by dictionary values
    _indices = [0] if by_key else [1]
    _indices += indices
    sort = sorted(dic.items(), key=getter(_indices), reverse=descending)
    return sort

class getter:
    def __init__(self, indices):
        self.indices = indices
        self.sequen_types = [list, dict, tuple]

    def __call__(self, sequen, _indices=None):
        _indices = _indices if _indices else self.indices

        if type(sequen) not in self.sequen_types:
            return sequen
        else:
            sequen = sequen[_indices[0]]
            return self.__call__(sequen, _indices[1:])


'''
Reverse

'''

def reverseDict(dic):
    rever = {}
    for key_1, key_2, val in nested(dic):
        if key_2 not in rever:
            rever[key_2] = {key_1 : val}
        else:
            rever[key_2][key_1] = val
    
    return rever


'''
Split

'''

def splitDict(dic, num):
    distri = splitNumber(len(dic), num)
    split = []
    keys = list(dic.keys())
    for size in distri:
        sub_keys = keys[:size]
        keys = keys[size:]
        split.append({key : dic[key] for key in sub_keys})
    return split

def splitList(lis, num):
    distri = splitNumber(len(lis), num)
    split = []
    for size in distri:
        split.append(lis[:size])
        lis = lis[size:]
    return split

def splitNumber(num, amount):
    base = num // amount
    left = num % amount
    split = [base + 1 if x < left else base
             for x in range(amount)]
    return split


'''
Iteration

'''

def nested(nest, depth=None):
    return NestedIterator(nest, depth)

class NestedIterator(Iterator):
    def __init__(self, nest, depth=None):
        self.nest = nest
        # 0 for dictionary, 1 for list
        depth = self.depth() if not depth else depth
        self.indices = [0 for x in range(depth)]
        
    def __iter__(self):
        return self
        
    def get(self, indices, nest=None):
        nest = nest if nest else self.nest
        
        if not indices:
            return [nest]

        idx = indices[0]
        if isinstance(nest, dict):
            idx = list(nest.keys())[idx]

        nest = nest[idx]
        return [idx] + self.get(indices[1:], nest)

    def depth(self, nest=None):
        nest = nest if nest else self.nest

        if isinstance(nest, list):
            nest = nest[0]
        elif isinstance(nest, dict):
            key = list(nest.keys())[0]
            nest = nest[key]
        else:
            return 0
        
        return self.depth(nest) + 1
            
    
    def __next__(self):
        if not self.indices:
            raise StopIteration

        values = self.get(self.indices)
        self.increaseIndices()
        return values

    def increaseIndices(self, indices=None):
        if indices is None:
            indices = self.indices
        elif indices == []:
            self.indices = None
            return None
        else:
            indices = indices
        
        curr_idx = indices[-1]
        curr_range = self.get(indices[:-1])[-1]
        if len(curr_range) > curr_idx+1:
            self.indices[len(indices)-1] = curr_idx + 1
        else:
            self.indices[len(indices)-1] = 0
            self.increaseIndices(indices[:-1])
