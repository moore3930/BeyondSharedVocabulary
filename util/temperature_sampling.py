import argparse
import random
import numpy as np
import os


def get_temperature(langs_sent_cnt, temperature):
    langs_sent_cnt = np.array(langs_sent_cnt, dtype=float)
    corpus_cnt = langs_sent_cnt.sum()
    temperature_list = (langs_sent_cnt / (2 * corpus_cnt)) ** (1./temperature)
    min_ratio = temperature_list.min()
    min_cnt = langs_sent_cnt.min()
    times = temperature_list / min_ratio
    samples_nums = times * min_cnt
    downsample_ratio = samples_nums / langs_sent_cnt
    return temperature_list, downsample_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-langs', dest='langs', default='en,ar,he')
    parser.add_argument('-pivot', dest='pivot', default='en')
    parser.add_argument('-train_dir', dest='train_dir', default='./train_data')
    parser.add_argument('-bpe_dir', dest='bpe_dir', default='./alignment_data')
    parser.add_argument('-temperature', dest='temperature', default='2')
    args = parser.parse_args()

    # get langs
    langs = args.langs.strip().split(',')
    pivot = args.pivot.strip()
    if pivot in langs:
        langs.remove(pivot)

    # get langs_sent_cnt
    langs_sent_cnt = []
    train_dir = args.train_dir
    for lang in langs:
        lang_pair = "{}-{}".format(pivot, lang)
        src_file = os.path.join(train_dir, "train.{}.{}".format(lang_pair, pivot))
        tgt_file = os.path.join(train_dir, "train.{}.{}".format(lang_pair, lang))
        cnt = 0
        with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                cnt += 1
        langs_sent_cnt.append(cnt)

    # calculate temperature
    temperatures, downsample_ratio = get_temperature(langs_sent_cnt, 2.0)
    print(downsample_ratio)

    # dump
    bpe_dir = args.bpe_dir
    for lang, ratio in zip(langs, downsample_ratio):
        lang_pair = "{}-{}".format(pivot, lang)
        src_file = os.path.join(train_dir, "train.{}.{}".format(lang_pair, pivot))
        tgt_file = os.path.join(train_dir, "train.{}.{}".format(lang_pair, lang))
        src_output = os.path.join(bpe_dir, "bpe.{}.{}".format(lang_pair, pivot))
        tgt_output = os.path.join(bpe_dir, "bpe.{}.{}".format(lang_pair, lang))

        with open(src_file) as src_fin, open(tgt_file) as tgt_fin, \
                open(src_output, 'w') as src_fout, open(tgt_output, 'w') as tgt_fout:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                if random.uniform(0, 1) < ratio:
                    src_fout.write(src_sent)
                    tgt_fout.write(tgt_sent)





