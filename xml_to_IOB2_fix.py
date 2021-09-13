import sys
from xml.etree.ElementTree import iterparse
import xml.etree.ElementTree as ET
import MeCab


class Tokenizer(object):
    def __init__(self, mecab):
        self.tokenizer = mecab


    def tokenize(self, sent):
        return self.tokenizer.parse(sent)[:-2].split(' ')


def split_tag(tag):
    if tag == "O":
        return tag, None
    else:
        t, l = tag.split('-')
        return t, l


def convert_xml_to_taglist(sent, tag_list=None, attr=None):
    """
    input:  "私は<C>宇宙人</C>だ．"
    output: "私は宇宙人だ", [(2, 4, "C")]
    """

    text = '<sent>' + sent + '</sent>'
    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed(text)

    ne_type = "O"
    ne_prefix = ""
    res = ""
    label = []
    tag_set = set()
    s_pos = -1
    idx = 0

    for event, elem in parser.read_events():
        isuse = (tag_list is None
                or (tag_list is not None and elem.tag in tag_list))

        if event == 'start':
            assert len(tag_set) < 2, "タグが入れ子になっています\n{}".format(sent)
            s_pos = idx

            if elem.attrib:
                attr_list = ''.join([v for k, v in elem.attrib.items() if k in attr])
            else:
                attr_list = ''

            word = elem.text if elem.text is not None else ""
            res += word
            idx += len(word)

            if elem.tag != 'sent' and isuse:
                tag_set.add(elem.tag)
                label.append((s_pos, idx-1, elem.tag + attr_list, word))

        if event == 'end':
            if elem.tag != 'sent' and isuse:
                tag_set.remove(elem.tag)
            word = elem.tail if elem.tail is not None else ""
            res += word
            idx += len(word)

    return res, label


def convert_taglist_to_iob(sent, label, tokenizer=list):
    """
    input: "私は宇宙人だ", [(2, 4, "C")], list
    output: [('私', 'O'), ('は', 'O'), ('宇', 'C') ...]

    parameters:
        sent:   "私は宇宙人だ"
        label:  [(2, 4, 'C')]
        tokenizer:  fn(str) -> list
    """

    tokens = tokenizer(sent)
    results = []

    idx = 0
    i = 0
    j = 0

    nebegin = True

    while j < len(sent) and idx < len(label):
        k = j + len(tokens[i]) - 1
        if k < label[idx][0]:
            results.append((tokens[i], 'O'))
        elif label[idx][0] <= k and nebegin:
            results.append((tokens[i], 'B-' + label[idx][2]))
            nebegin = False
        else:
            results.append((tokens[i], 'I-' + label[idx][2]))

        j += len(tokens[i])
        i += 1

        while idx < len(label) and label[idx][1] < j:
            idx += 1
            nebegin = True

    while i < len(tokens):
        results.append((tokens[i], 'O'))
        i += 1


    return results


def convert_xml_to_iob(sent, tag_list=None, attr=None, tokenizer=list):
    res, label = convert_xml_to_taglist(sent, tag_list=tag_list, attr=attr)
    return convert_taglist_to_iob(res, label, tokenizer=tokenizer)


def print_iob(iob):
    for t, l in iob:
        print(t + '\t' + l)


if __name__ == "__main__":
    # data path
    data = sys.argv[1]

    # 属性リスト
    attr_list = ['MOD'] # 有効なタグリスト
    valid_tag = ['d', 'a', 'timex3', 't-test', 't-key', 't-val', 'm-key', 'm-val']
    #valid_tag = None

    # tokenizer
    # 文字単位だと
    tokenizer = list
    """
    mecab = MeCab.Tagger('-Owakati')
    tokenizer = Tokenizer(mecab).tokenize

    # 東大BERT
    mecab = MeCab.Tagger('-Owakati -d /opt/mecab/lib/mecab/dic/mecab-ipadic-neologd -u /opt/mecab/lib/mecab/dic/MANBYO_201907_Dic-utf8.dic')
    tokenizer = Tokenizer(mecab).tokenize
    """
    
    all_tag_list = []
    all_tags = []
    with open(data, 'r') as f:
        
        for i, line in enumerate(f):
            line = line.rstrip()

            if not line:
                continue

            res, tags = convert_xml_to_taglist(line, valid_tag, attr=attr_list)
            #tag_list = convert_taglist_to_iob(res, tags)
            tag_list = convert_xml_to_iob(line, valid_tag, attr_list)
            print(tags)
            print_iob(tag_list)
            all_tag_list.append(tag_list)
            all_tags.append(tags)
            print("")
        
file_path = data+'.txt'
f = open(file_path,'w+')
for i in all_tag_list:
    for j,k in i:
            f.writelines(j)
            f.writelines('\t')
            f.writelines(k)
            f.writelines('\n')
f.close()

g = open('tag.txt','w+')
tag_list = []
for i in all_tags:
    for j in i:
        if j[2] not in tag_list:
            tag1 = 'B-'+j[2]
            tag2 = 'I-'+j[2]
            g.writelines(tag1)
            g.writelines('\n')
            g.writelines(tag2)
            g.writelines('\n')
        tag_list.append(j[2])
g.writelines('O')
g.close()