"""获取文章高频词 TF(Term Frequency)

后期会有画图功能吧,包装画图库,然后保存到指定路径.
"""

import os
from collections import Counter

from .config import cut_data_dir, stop_words_data_dir


def stop_words(file_path):
    with open(file_path, 'rt', encoding='utf-8') as f:
        return {w.strip() for w in f}


def get_TF_from_file(file_path,
                     stop_words=stop_words(
                         os.path.join(stop_words_data_dir,
                                      'Chinese stop words')),
                     topK=None):
    with open(file_path, 'rt', encoding='utf-8') as f:
        text = f.read().strip()
    words_list = text.split()
    return get_TF(words_list, stop_words=stop_words, topK=topK)


def get_TF(words_list,
           stop_words=stop_words(
               os.path.join(stop_words_data_dir, 'Chinese stop words')),
           topK=None):
    """获取高频词的API接口函数"""
    counter = Counter(words_list)
    # 去除停用词典中的词
    for words in stop_words:
        counter.pop(words, None)
    if topK is None:
        data = counter.most_common()
    else:
        data = counter.most_common(n=topK)
    return data


if __name__ == '__main__':
    result = get_TF_from_file(
        os.path.join(cut_data_dir, "CUT@People's Daily199801"), topK=30)
    print(result)
