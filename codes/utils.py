# -*- coding: utf-8 -*-

from itertools import chain
import os

def flatten(y):
    return list(chain.from_iterable(y))

class Reader:
    
    def __init__(self):
        pass
    
    def read(dataset_folder_path):
        text_arr = None
        tags_arr = None
        
        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.read().splitlines()
        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.read().splitlines()
            
        assert len(text_arr) == len(tags_arr)
        return text_arr, tags_arr
