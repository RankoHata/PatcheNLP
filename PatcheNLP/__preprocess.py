import os
import re
from config import hmm_model_dir, pos_tagging_data_dir, cut_data_dir

original_file_path = os.path.join(pos_tagging_data_dir,
                                  "POS tagging@People's Daily199801")


def preprocess_text(file_path=original_file_path):
    with open(file_path, 'rt', encoding='utf-8') as f:
        with open(os.path.join(cut_data_dir, "CUT@People's Daily199801"), 'wt', encoding='utf-8') as q:
            for line in f:
                if line == '\n':
                    continue
                _, text  = line.split(maxsplit=1)
                text = re.sub(r'/\w*', '', text)
                q.write(text)


if __name__ == '__main__':
    # preprocess_text()
    pass
