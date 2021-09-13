from transformers import BertJapaneseTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset
from allennlp.modules import conditional_random_field
from sklearn.model_selection import train_test_split
import argparse
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import mlflow
import numpy as np
from tqdm import tqdm
import mojimoji
import json
from model import BertCrf
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

import data_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_dataset, val_dataset, max_epoch=20, batch_size=16, outputdir=None):
    data = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_utils.my_collate_fn)

    val_data = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_utils.my_collate_fn)
    val_loss = [float('inf'), float('inf')]

    bert_parameter = list(model.bert.parameters())
    other_parameter = (list(model.hidden_to_output.parameters()) +
        list(model.crf.parameters()))
    optimizer_grouped_parameters = [
        {'params': bert_parameter, 'lr':5e-5},
        {'params': other_parameter, 'lr':0.001}
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters)

    losses = []
    model.to(device)
    for epoch in tqdm(range(max_epoch)):
        model.train()
        all_loss = 0
        step = 0

        for sent, label in data:
            #print(len(label[0]))
            input_x = pad_sequence([torch.tensor(x)
                for x in sent], batch_first=True).to(device)
            input_y = pad_sequence([torch.tensor(y)
                for y in label], batch_first=True).to(device)
            print(input_x.shape)
            print(input_y.shape)
            mask = [[float(i>0) for i in ii] for ii in input_x]
            print(len(mask[0]))
            mask = torch.tensor(mask).to(device)

            loss = model(input_x, input_y, mask) * (-1.0)
            all_loss += loss.item()

            loss.backward()
            optimizer.step()
            #scheduler.step()
            model.zero_grad()

            step += 1

        losses.append(all_loss / step)
        mlflow.log_metric("loss", losses[-1], step=epoch)

        model.eval()
        all_loss = 0
        step = 0

        for sent, label in val_data:
            input_x = pad_sequence([torch.tensor(x)
                for x in sent], batch_first=True).to(device)
            input_y = pad_sequence([torch.tensor(y)
                for y in label], batch_first=True).to(device)
            mask = [[float(i>0) for i in ii] for ii in input_x]
            mask = torch.tensor(mask).to(device)

            loss = model(input_x, input_y, mask) * (-1.0)
            all_loss += loss.item()

            step += 1
        val_loss.append(all_loss / step)
        mlflow.log_metric("val_loss", val_loss[-1], step=epoch)
        output_path = outputdir + '/checkpoint{}.model'.format(len(val_loss)-1)
        torch.save(model.state_dict(), output_path)

        if val_loss[-1] > val_loss[-2] and val_loss[-2] > val_loss[-3]:
            break

    #print(val_loss)
    if val is not None:
        min_epoch = np.argmin(val_loss)
        print(min_epoch)
        model_path = outputdir + '/checkpoint{}.model'.format(min_epoch)
        model.load_state_dict(torch.load(model_path))

    torch.save(model.state_dict(), outputdir+'/final.model')

def evaluate(model, x):
    data = data_utils.Batch(x, x, batch_size=8, sort=False)

    model.to(device)
    output = []
    model.eval()
    for sent, label in data:
        input_x = pad_sequence([torch.tensor(x)
            for x in sent], batch_first=True).to(device)
        input_y = pad_sequence([torch.tensor(y)
            for y in label], batch_first=True).to(device)
        mask = [[float(i>0) for i in ii] for ii in input_x]
        mask = torch.tensor(mask).to(device)

        tags = [m[0] for m in model.decode(input_x, mask)]
        output += tags

    return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--train_path', type=str, help='data path')
    parser.add_argument('--val_path', type=str, help='data path')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_dir', type=str, help='batch size')
    parser.add_argument('--output_path', type=str, help='batch size')
    parser.add_argument('--labels', type=str, help='batch size')
    args = parser.parse_args()

    mlflow.start_run()

    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

    label_vocab = data_utils.create_label_vocab_from_file(args.labels)
    itol = {i:w for w, i in label_vocab.items()}

    constraints = conditional_random_field.allowed_transitions("BIO", itol)
    train_dataset = data_utils.IobDataset(args.train_path, tokenizer, label_vocab)
    if args.val_path:
        val_dataset = data_utils.IobDataset(args.val_path, tokenizer, label_vocab)
    else:
        # validation setの指定が無い場合、1/10をvalidationにする
        train_size = len(train_dataset)
        val_index = random.sample([i for i in range(train_size)], train_size//10)
        val_dataset = Subset(train_dataset, val_index)

        val_index = set(val_index)
        train_index = [i for i in range(train_size) if i not in val_index]
        train_dataset = Subset(train_dataset, train_index)

    bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-char')
    model = BertCrf(bert, len(label_vocab), constraints)

    train(model, train_dataset, val_dataset, outputdir=args.output_dir, batch_size=args.batch_size)
    tags = evaluate(model, input_x_test)


    mlflow.end_run()

    """
    # test script
    labels = [[itol[t] for t in tag] for tag in tags]
    input_x_test = [tokenizer.convert_ids_to_tokens(t)[1:] for t in input_x_test]
    input_y_test = [[itol[i] for i in t] for t in input_y_test]

    output = []
    for x, t, y in zip(input_x_test, labels, input_y_test):
        output.append('\n'.join([x1 + '\t' + str(x2) + '\t' + str(x3) for x1, x2, x3 in zip(x, y, t)]))

    with open(args.output_path, 'w') as f:
        f.write('\n\n'.join(output))
    """

