"""测试功能"""
import sys
sys.path.append('.')

from PatcheNLP.hmm import HMM


if __name__ == '__main__':
    x = HMM()
    res = x.cut('落霞与孤鹜齐飞')
    assert len(res) >= 1 and len(res) <= 7
    print(res)
