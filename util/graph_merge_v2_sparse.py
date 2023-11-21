import os.path

import numpy as np
import argparse
import copy
import math
import os
from os import listdir
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz


# This version distribute self-link prob based on alignments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-langs', dest='langs', default="en,ar,he,de,nl")
    parser.add_argument('-lang_pairs', dest='lang_pairs', default="en-ar,en-he,en-de,en-nl")
    parser.add_argument('-dict_file', dest='dict_file', default="./OPUS/dict.txt")
    parser.add_argument('-sentence_prefix', dest='sentence_prefix', default="./OPUS/fast_text")
    parser.add_argument('-alignment_prefix', dest='alignment_prefix', default="./OPUS/final")
    parser.add_argument('-alpha', dest='alpha', type=float, default=1.0)
    parser.add_argument('-output', dest='output', default="./OPUS/data_bin/alignment_matrix.npz")
    args = parser.parse_args()

    # initialize graph
    token_list = []
    token_dict = {}
    idx = 0
    with open(args.dict_file) as fin:
        for line in fin:
            token, _ = line.strip().split()
            token_list.append(token)
            token_dict[token] = idx
            idx += 1

    token_num = len(token_list)
    R = np.zeros([token_num, token_num])
    I = np.identity(token_num)

    # set alignment dict
    pair_list = args.lang_pairs.strip().split(',')
    alignment_dict = {}

    # English -> X Alignment Prob
    for pair in pair_list:
        print("Calculate {} alignment prob...".format(pair))
        sentence_file = "{}.{}".format(args.sentence_prefix, pair)
        alignment_file = "{}.{}.align".format(args.alignment_prefix, pair)
        alignment_dict[pair] = {}
        cur_dict = alignment_dict[pair]

        with open(sentence_file) as fin_1:
            with open(alignment_file) as fin_2:
                for sent_pair, alignments in zip(fin_1, fin_2):
                    temp = sent_pair.strip().split(" ||| ")
                    if len(temp) != 2:
                        continue
                    src_sent, tgt_sent = temp
                    src_token_list = src_sent.strip().split(" ")
                    tgt_token_list = tgt_sent.strip().split(" ")
                    alignment_list = alignments.strip().split(" ")
                    for alignment in alignment_list:
                        if len(alignment.strip()) == 0:
                            continue
                        src_idx, tgt_idx = alignment.strip().split("-")
                        src_token = src_token_list[int(src_idx)]
                        tgt_token = tgt_token_list[int(tgt_idx)]
                        if src_token not in token_dict or tgt_token not in token_dict:
                            continue
                        if src_token not in cur_dict:
                            temp = {tgt_token: 1.0}
                            cur_dict[src_token] = temp
                        else:
                            temp = cur_dict[src_token]
                            if tgt_token not in temp:
                                temp[tgt_token] = 1.0
                            else:
                                temp[tgt_token] += 1.0

        # normalize
        for tok in cur_dict:
            temp_dict = cur_dict[tok]
            temp_dict = {key: val for key, val in temp_dict.items() if val >= 10}
            if len(temp_dict) == 0:
                continue
            cnt = sum(temp_dict.values())
            temp_dict = {key: value / cnt for key, value in temp_dict.items()}
            cur_dict[tok] = temp_dict

    # X -> English Alignment Prob: using EN-X data reversely
    for pair in pair_list:
        sentence_file = "{}.{}".format(args.sentence_prefix, pair)
        alignment_file = "{}.{}.align".format(args.alignment_prefix, pair)

        src, tgt = pair.strip().split("-")
        pair_rs = tgt + "-" + src
        alignment_dict[pair_rs] = {}
        cur_dict = alignment_dict[pair_rs]
        print("Calculate {} alignment prob...".format(pair_rs))

        with open(sentence_file) as fin_1:
            with open(alignment_file) as fin_2:
                for sent_pair, alignments in zip(fin_1, fin_2):
                    temp = sent_pair.strip().split(" ||| ")
                    if len(temp) != 2:
                        continue
                    tgt_sent, src_sent = temp                                       # reverse
                    src_token_list = src_sent.strip().split(" ")
                    tgt_token_list = tgt_sent.strip().split(" ")
                    alignment_list = alignments.strip().split(" ")
                    for alignment in alignment_list:
                        if len(alignment.strip()) == 0:
                            continue
                        tgt_idx, src_idx = alignment.strip().split("-")             # reverse
                        src_token = src_token_list[int(src_idx)]                    # X
                        tgt_token = tgt_token_list[int(tgt_idx)]                    # EN
                        if src_token not in token_dict or tgt_token not in token_dict:
                            continue
                        if src_token not in cur_dict:
                            temp = {tgt_token: 1.0}
                            cur_dict[src_token] = temp
                        else:
                            temp = cur_dict[src_token]
                            if tgt_token not in temp:
                                temp[tgt_token] = 1.0
                            else:
                                temp[tgt_token] += 1.0

        # normalize
        for tok in cur_dict:
            temp_dict = cur_dict[tok]
            temp_dict = {key: val for key, val in temp_dict.items() if val >= 10}
            if len(temp_dict) == 0:
                continue
            cnt = sum(temp_dict.values())
            temp_dict = {key: value / cnt for key, value in temp_dict.items()}
            cur_dict[tok] = temp_dict

    # build graph
    print("build graph ...")
    alpha = args.alpha

    def merge_dict(dict_list):
        key_set = set()
        for temp_dict in dict_list:
            key_set.update(temp_dict.keys())

        out_dict = {}
        for key in key_set:
            prob_sum = 0.0
            for cur_dict in dict_list:
                if key in cur_dict:
                    prob_sum += cur_dict[key]
            out_dict[key] = prob_sum

        return out_dict

    for src_tok in token_dict.keys():
        src_tok_idx = token_dict[src_tok]
        tgt_dict_list = []
        for pair in alignment_dict:
            cur_dict = alignment_dict[pair]
            if src_tok in cur_dict:
                tgt_dict = copy.deepcopy(cur_dict[src_tok])
                tgt_dict_list.append(tgt_dict)
        merged_dict = merge_dict(tgt_dict_list)

        for tok in merged_dict:
            tgt_tok_idx = token_dict[tok]
            R[src_tok_idx][tgt_tok_idx] = merged_dict[tok]

    # normalize graph
    row_sum = np.expand_dims(R.sum(-1), -1)         # [N, 1]
    R = R / row_sum
    R = np.nan_to_num(R)                            # in case divide zero
    R = R * alpha

    # add self alignment
    A = (row_sum == 0) * 1.
    B = (row_sum != 0) * (1. - alpha)
    self_align_weight = A + B
    I = I * self_align_weight
    R = R + I

    # add special tokens & lang tokens
    langs = args.langs.strip().split(',')
    special_token_num = 4
    lang_token_num = len(langs)
    total_dim = R.shape[0] + special_token_num + lang_token_num
    R_expand = np.zeros([total_dim, total_dim])
    R_expand[special_token_num:-lang_token_num, special_token_num:-lang_token_num] = R
    for idx in range(special_token_num):
        R_expand[idx][idx] = 1.0
    for idx in range(lang_token_num):
        R_expand[special_token_num + token_num + idx][special_token_num + token_num + idx] = 1.0

    print("### shape of graph matrix: {}".format(str(R_expand.shape[0])))
    print(R_expand)
    print(np.sum(R_expand))

    # dump matrix
    R_expand = np.float32(R_expand)
    export_sparse_np = coo_matrix(R_expand)
    save_npz(args.output, export_sparse_np)
    load_sparse_np = load_npz(args.output + ".npz")

    if np.array_equal(export_sparse_np.row, load_sparse_np.row) \
            and np.array_equal(export_sparse_np.col, load_sparse_np.col) \
            and np.array_equal(export_sparse_np.data, load_sparse_np.data):
        print("The sparse matrices are equal.")
    else:
        print("The sparse matrices are not equal.")
