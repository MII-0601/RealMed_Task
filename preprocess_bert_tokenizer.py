import sys
import mojimoji
from transformers import BertJapaneseTokenizer, BertTokenizer


def split_tag(tag):
    if tag == 'O':
        return 'O', None
    else:
        t, l = tag.split('-')
        return t, l


def merge_tag(tag, label):
    if tag == "O":
        return "O"
    else:
        return tag + '-' + label


def expand_tag(tag, length):
    t, l = split_tag(tag)
    if t == 'B':
        ex_tag = merge_tag('I', l)
    else:
        ex_tag = merge_tag(t, l)

    return [tag] + [ex_tag] * (length - 1)


def split_subword(token, tag, tokenizer):
    sub_tokens = tokenizer.tokenize(token)
    sub_tags = expand_tag(tag, len(sub_tokens))
    return list(zip(sub_tokens, sub_tags))


def print_many_tag(tag_list):
    """
    input:  [('私', 'O'), ...]
    output: 私\tO\n...
    """
    for t, l in tag_list:
        print(t + '\t' + l)

if __name__ == '__main__':
    # data path
    data = sys.argv[1]

    # tokenizer の指定
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char')

    # 東大BERT
    #tokenizer = BertTokenizer('/home/is/ujiie/models/UTH_BERT_BASE_MC_BPE_V25000_10M/vocab.txt', do_basic_tokenize=False)

    max_len = 511
    subword_len_counter = 0

    with open(data, 'r') as f:
        for line in f:
            line = line.rstrip()

            if not line:
                print("")
                subword_len_counter = 0
                continue

            token, tag = line.split('\t')

            sub_iob = split_subword(token, tag, tokenizer)
            current_subwords_len = len(sub_iob)

            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                print("")
                print_many_tag(sub_iob)
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            print_many_tag(sub_iob)

