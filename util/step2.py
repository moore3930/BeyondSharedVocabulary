import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_sentences', dest='src_sentences', default='')
    parser.add_argument('-tgt_sentences', dest='tgt_sentences', default='')
    parser.add_argument('-fast_text', dest='fast_text', default='')
    args = parser.parse_args()

    with open(args.src_sentences) as fin_src:
        with open(args.tgt_sentences) as fin_tgt:
            with open(args.fast_text, 'w') as fout:
                for src, tgt in zip(fin_src, fin_tgt):
                    src = src.strip()
                    tgt = tgt.strip()
                    fout.write("{} ||| {}\n".format(src, tgt))




