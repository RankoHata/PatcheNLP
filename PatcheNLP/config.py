# -*- coding: utf-8 -*-
"""工程配置文件"""

import os

# 主目录
base_dir = os.path.dirname(__file__)

# 数据目录
data_dir = os.path.join(base_dir, 'data')

cut_data_dir = os.path.join(data_dir, 'cut')
pos_tagging_data_dir = os.path.join(data_dir, 'pos_tagging')

# 模型目录
model_dir = os.path.join(base_dir, 'model')

hmm_model_dir = os.path.join(model_dir, 'hmm')
