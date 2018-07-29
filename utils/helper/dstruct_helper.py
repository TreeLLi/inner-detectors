'''
Data structure helper

To provide auxiliary functions easily handling data structures

'''

from oeprator import itemgetter


'''
Sorting

'''

# result is returned as a list of tuples (key, val)
def sortDict(dic, by_key=False, desending=True):
    # sorted by dictionary values
    key = 0 if by_key else 1
    sorted = sorted(dic.items(), key=itemgetter(key), reverse=descending)
    return sorted


'''
Reverse

'''

def reverseDict(dic):
    rever = {}
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
    if isinstance(nest, dict):
        # parse dict to adapt format of iterator
        continue
    else:
        continue
        

class NestedIterator(Iterator):
    def __init__(self, nest):
        self.nest = nest
        self.indices = [0 for x in range(len(nest))]

    def next(self):
        values = []
        for i, idx in enumerate(self.indices):
            if i != 0:
                val = _nest[idx]
            else:
                val = self.nest[0][idx]
                
            if isinstance(val, tuple):
                values += list(val)
            elif isinstance(val, list):
                values += val
            else:
                values.append(val)
                
            _nest = self.nest[i+1] if i+1 < len(self.nest) else None


class NestedIterable(Iterable):
    def __init__(self, nest):
        self.nest = nest

    def __iter__(self):
        return NestedIterator(self.nest)
