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

def nested(nest):
    return NestedIterator(nest)

class NestedIterator(Iterator):
    def __init__(self, nest):
        self.nest = nest
        # 0 for dictionary, 1 for list
        self.mode = 0 if isinstance(nest, dict) else 1
        self.indices = [None if self.mode==0 else 0 for x in range(2)]
        self.increaseIndices()
        
    def __iter__(self):
        return self
        
    def get(self, nest, indices):
        _nest = nest
        for i in indices:
            _nest = _nest[i]
        return _nest
        
    def __next__(self):
        if not self.indices:
            raise StopIteration
        
        if self.mode == 0:
            val = self.get(self.nest, self.indices)
            values = self.indices + [val]
            self.increaseIndices()
            return values
        elif self.mode == 1:
            self.increaseIndices()
        else:
            raise Exception("Error: NestedIterator invalid mode")

    def increaseIndices(self):
        # TODO - shit codes, must be refactored
        if self.mode == 0:
            if not self.indices[0]:
                self.indices[0] = sorted(self.nest.keys())[0]
                self.indices[1] = sorted(self.nest[self.indices[0]].keys())[0]
                return None
            dic = self.nest[self.indices[0]]
            keys = sorted(dic.keys())
            if self.indices[-1] in keys:
                idx = keys.index(self.indices[-1])
                if idx < len(keys)-1:
                    self.indices[-1] = keys[idx+1]
                else:
                    keys = sorted(self.nest.keys())
                    idx = keys.index(self.indices[0])
                    if idx < len(keys)-1:
                        self.indices[0] = keys[idx+1]
                        self.increaseIndices()
                    else:
                        self.indices = None
            else:
                self.indices[-1] = keys[0]
