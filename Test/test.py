"""测试功能"""
import os
import sys
sys.path.append('.')

from PatcheNLP.config import *
from PatcheNLP.cut_hmm import HMM as CUTHMM
from PatcheNLP.hmm import HMM
from PatcheNLP.TF import get_TF_from_file


if __name__ == '__main__':
    x = HMM()
    res = x.cut('落霞与孤鹜齐飞')
    assert len(res) >= 1 and len(res) <= 7
    print(res)

    x = CUTHMM()
    res = x.cut('落霞与孤鹜齐飞')
    assert len(res) >= 1 and len(res) <= 7
    print(res)

    result = get_TF_from_file(file_path=os.path.join(cut_data_dir, "CUT@People's Daily199801"))
    print(result)
