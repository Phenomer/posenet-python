#!/usr/bin/python3
#-*- coding: utf-8 -*-

import collections
import statistics

class DictStat():
    def __init__(self, size):
        self.deque = collections.deque([], size)
    
    def append(self, dic):
        self.deque.append(dic)
    
    def process(self, func):
        dlist = {}
        dic_res = {}
        for dic in list(self.deque):
            for k, v in dic.items():
                if not k in dlist:
                    dlist[k] = []
                dlist[k].append(v)
        
        for k, v in dlist.items():
            dic_res[k] = func(v)
        return dic_res


if __name__ == "__main__":
    ds = DictStat(5)
    ds.append({'mizuki': 1, 'miku': 2})
    ds.append({'mizuki': 3, 'miku': 4})
    ds.append({'mizuki': 1, 'miku': 9})
    ds.append({'mizuki': 3, 'miku': 3})
    ds.append({'mizuki': 3, 'miku': 2})
    ds.append({'mizuki': 6, 'miku': 9})
    ds.append({'mizuki': 5, 'miku': 2})
    ds.append({'mizuki': 4, 'miku': 3})
    ds.append({'mizuki': 2, 'miku': 2})
    ds.append({'mizuki': 3, 'miku': 4})
    print(ds.process(statistics.mean))
    print(ds.process(statistics.harmonic_mean))
    print(ds.process(statistics.median))
    print(ds.process(statistics.median_grouped))
    #print(ds.process(statistics.mode))
