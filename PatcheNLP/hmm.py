# -*- coding: utf-8 -*-
"""隐含马尔可夫模型
出现符号 '@@@' ,表明该处未搞懂,存在疑惑等.
"""

import os
import pickle
from .config import hmm_model_dir, cut_data_dir


class HMM(object):
    def __init__(self, model_file_path=os.path.join(hmm_model_dir, 'hmm_model.pkl'), data_file_path=os.path.join(cut_data_dir, "CUT@People's Daily199801")):
        # 存取算法的中间结果的二进制文件(pickle序列化)
        self.model_file_path = model_file_path
        self.data_file_path = data_file_path
        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']  # 词首,词中,词尾,单独成词
        # 是否重新加载model_file
        # 通过该参数,使只有第一次调用cut函数时,会通过模型加载数据
        self.load_para = False
    
    def try_load_model(self, trained):
        """加载中间结果
        Args:
            trained: True: 加载数据 False: 重置概率矩阵
        """
        if trained:
            with open(self.model_file_path, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率(状态->状态)
            self.A_dic = {}
            # 发射概率(状态->词语)
            self.B_dic = {}
            # 初始概率
            self.Pi_dic = {}
            self.load_para = False
    
    def train(self, cut_data_path):
        """通过给定的分词语料进行训练
        计算初始概率,转移概率,发射概率
        """
        self.try_load_model(False)  # 重置概率矩阵
        # 统计状态出现次数,用于计算发射概率和转移概率
        Count_dic = {}

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}  # 转移概率矩阵(嵌套字典)
                self.Pi_dic[state] = 0.0  # 初始概率矩阵
                self.B_dic[state] = {}  # 发射概率矩阵(嵌套矩阵)
                Count_dic[state] = 0

        def makeLabel(text):
            """为每个在语料库中的切分好的词逐字加标签"""
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text
        
        init_parameters()
        line_num = -1  # 初始行号
    
        words = set()
        with open(cut_data_path, encoding='utf-8') as f:
            for line in f:
                line_num += 1  # @@@个人:放在这里,不合适,空行不应该被算入..而且初始-1不如0吧,如果只有一行,不是出bug了么
                line = line.strip()
                if not line:  # 空行
                    continue
                word_list = [i for i in line if i != ' ']
                words |= set(word_list)  # 并集
            
                linelist = line.split()  # 按空切分,不同于split(' ')
                line_state = []
                for w in linelist:
                    line_state.extend(makeLabel(w))

                assert len(word_list) == len(line_state)  # 保证字和状态数相同

                for index, state in enumerate(line_state):
                    Count_dic[state] += 1
                    if index == 0:
                        # 每个句子的第一个字的状态,用于计算初始状态概率
                        self.Pi_dic[state] += 1
                    else:
                        self.A_dic[line_state[index - 1]][state] += 1  # 统计o(k)与o(k-1)的出现次数
                        self.B_dic[line_state[index]][word_list[index]] = (
                            self.B_dic[line_state[index]].get(word_list[index], 0) + 1.0  # 统计在在词性确定下,字符出现的次数
                        )
        
        # 每行开头出现该种状态的概率(P(o1))
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}  # P(o(k)|P(o(k-1))
        # @@@加1平滑(猜想:因为v1过小吗,如果是0,会导致永远不可能出现这种情况,但实际上可能因为语料中没有,而出现巨大错误???)
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dic.items()}  # P(λ(k)|o(k))
        
        with open(self.model_file_path, 'wb') as f:  # 覆盖以前的数据
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        """Viterbi算法(动态规划思想)
        Args:
            start_p: 初始概率,一句话第一个字的状态概率,即P(o1)
            trans_p: 转移概率
            emit_p: 发射概率
        Returns:
            eg:    (float, ['B', 'M', 'M', 'E', 'S', ...])
        """
        V = [{}]  # 存储每一步的路径概率
        path = {}
        for state_item in states:
            V[0][state_item] = start_p[state_item] * emit_p[state_item].get(text[0], 0)  # 可能会出现第一个字不在预料中的情况
            path[state_item] = [state_item]
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            
            # 检验发射概率
            cur_char = text[t]
            nerver_seen: bool = (  # 确认该字不在语料库中
                cur_char not in emit_p['S'] and
                cur_char not in emit_p['M'] and
                cur_char not in emit_p['E'] and
                cur_char not in emit_p['B']
            )
            for state_item in states:
                emitP = emit_p[state_item].get(cur_char, 0) if not nerver_seen else 1.0  # @@@设置未知字单独成词
                (prob, state) = max(
                    [(V[t - 1][state_item_0] * trans_p[state_item_0].get(state_item, 0) * emitP, state_item_0) 
                    for state_item_0 in states if V[t - 1][state_item_0] > 0]  # 等于0,没必要计算了
                )
                V[t][state_item] = prob
                new_path[state_item] = path[state] + [state_item]
            path = new_path
        
        # @@@ 这个if-else语句,没看懂...
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            # 最后一个字为词中的概率 大于 为独立成词的概率
            (prob, state) = max([(V[-1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[-1][y], y) for y in states])

        return (prob, path[state])

    def cut(self, text):
        if not self.load_para:
            if os.path.exists(self.model_file_path):
                self.try_load_model(True)
            else:
                self.train(self.data_file_path)
        result = []
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next_ = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            # @@@ 这几个判断语句,我认为有bug,如果语料错误数据过多..出现 'E', 'B',或者'B', 'S', 'E'这种,都会报错
            # 不改这几条语句,而是在加载或计算数据之后,根据规则将转移概率某些数据置0,(估计可以通过改上面的算法,1.在运算时即置零,可能效果高点? 2.在整体计算之后再置零.)
            if pos == 'B':
                begin = i
            elif pos == 'E':
                result.append(text[begin:i + 1])
                next_ = i + 1
            elif pos == 'S':
                result.append(char)
                next_ = i + 1
        if next_ < len(text):
            result.append(text[next_:])
        return result


if __name__ == '__main__':
    hmm = HMM()

    text = '周郎妙计安天下，赔了夫人又折兵。'
    res = hmm.cut(text)
    print(text)
    print(str(list(res)))

    # with open('./data/hmm_model.pkl', 'rb') as f:
    #     A_dic = pickle.load(f)
    #     B_dic = pickle.load(f)
    #     Pi_dic = pickle.load(f)
    # # from pprint import pprint
    # # pprint(A_dic)
    # # # pprint(B_dic)
    # # pprint(Pi_dic)
    # with open('temp', 'wt', encoding='utf-8') as f:
    #     print(B_dic, file=f)
