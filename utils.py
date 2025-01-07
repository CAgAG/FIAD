# @Author: CAG
# @Github: CAgAG
# @Encode: UTF-8
# @FileName: utils.py

import sys

import numpy as np


class PrintToFile(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.file = open(self.file_path, 'a', encoding="utf-8")

    def __enter__(self):
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        self.file.close()


def set_top_k_to_one(arr, k):
    top_k_indices = np.argpartition(arr, -k)[-k:]
    result = np.zeros_like(arr)
    result[top_k_indices] = 1
    return result
