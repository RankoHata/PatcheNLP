                text = re.sub(r'\[|/\w*|]\w*', '',
                              text)  # 这一步可以把 复合词性 和 普通词性 全部去掉
                q.write(text + '\n')


if __name__ == '__main__':
    pos_file_path = os.path.join(pos_tagging_data_dir,
                                 "POS tagging@People's Daily199801")
    cut_file_path = os.path.join(cut_data_dir, "CUT@People's Daily199801")
    pos2cut_people_daily(pos_file_path, cut_file_path)
    pass
