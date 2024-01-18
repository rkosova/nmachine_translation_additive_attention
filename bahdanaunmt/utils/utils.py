# Some of this code comes from or is modified from 
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#loading-data-files

from __future__ import unicode_literals, print_function, division
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re

import numpy as np

import random

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from bahdanaunmt.models import Encoder, Decoder

class Lang:
    def __init__(self, name) -> None:
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def prepareLangs(in_lang, out_lang, path):
    with open(path) as f:
        lines = f.readlines()

        pairs = []

        for l in lines:
            l = l.split('\t')
            pair = []
            skip = False
            for s in l:
                if s == '':
                    skip = True
                    break
                pair.append(normalizeString(s))
            if not skip:
                pairs.append(pair) 

        lang1 = Lang(in_lang)
        lang2 = Lang(out_lang)


    return pairs, lang1, lang2


def populateLangs(in_lang, out_lang, path, max_s_len = 20):
    pairs, lang1, lang2 = prepareLangs(in_lang, out_lang, path)
    pairs_r = []

    for p in pairs:
        p1_l = len(p[0].split(' '))
        p2_l= len(p[1].split(' '))

        if p1_l >= max_s_len or p2_l >= max_s_len or len(p) != 2:
            pass
        else:
            pairs_r.append(p)
            lang1.addSentence(p[0])
            lang2.addSentence(p[1])


    print(f"Read {len(pairs_r)} sentences with less than {max_s_len} words.")
    print(f"Added {lang1.n_words} to {lang1.name} corpus.")
    print(f"Added {lang2.n_words} to {lang2.name} corpus.")

    return pairs_r, lang1, lang2
        

def sentence2index(s, lang:Lang):
    return [lang.word2index[w] for w in s.split(' ')]


def generateDataloader(batch, in_lang, out_lang, path, max_s_len=20, device='cuda'):
    pairs, lang1, lang2 = populateLangs(in_lang, out_lang, path, max_s_len)

    n = len(pairs)
    input_t = np.zeros((n, max_s_len), dtype=np.int32)
    output_t = np.zeros((n, max_s_len), dtype=np.int32)

    for idx, (s1, s2) in enumerate(pairs):
        s1_ids = sentence2index(s1, lang1)
        s2_ids = sentence2index(s2, lang2)
        s1_ids.append(1)
        s2_ids.append(1)
        input_t[idx, :len(s1_ids)] = s1_ids
        output_t[idx, :len(s2_ids)] =s2_ids

    data = TensorDataset(torch.LongTensor(input_t).to(device), 
                         torch.LongTensor(output_t).to(device))
    
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, batch, sampler=sampler)


    return dataloader, pairs, lang1, lang2


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (~ %s left)' % (asMinutes(s), asMinutes(rs))


def training_step(dataloader, encoder, decoder, loss_fn, encoder_optimizer, decoder_optimizer):
    epoch_loss = 0

    for idx, data in enumerate(dataloader):
        input_tensor, output_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, attention = decoder(encoder_outputs, encoder_hidden, output_tensor) # train with teacher forcing


        # print(output_tensor.shape, decoder_outputs.shape)
        loss = loss_fn(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            output_tensor.view(-1)
        )

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(dataloader)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, device='cuda'):
    with torch.no_grad():
        input_tokens = sentence2index(sentence, input_lang)
        input_tokens.append(1)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).view(1, -1)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == 1:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def showAttention(input_sentence, output_words, attentions):
    # attentions = attentions.view(len(output_words), 10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(f'attention{random.random()}.png')


def evaluateAndShowAttention(encoder, decoder, input_lang, output_lang, input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5, device='cuda'):
    encoder.eval()
    decoder.eval()
    for i in range(n):
        pair = random.choice(pairs)
        print('input:', pair[0])
        print('expected:', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = ' '.join(output_words)
        print('ai:', output_sentence)
        print('')
    return pair


def train(in_lang, out_lang, path, max_s_len=20, batch=32, n_epochs=50, learning_rate=1e-3, hidden_size=128, dropout_p=0.1, device='cuda'):
    dataloader, pairs, lang1, lang2 = generateDataloader(batch, in_lang, out_lang, path, max_s_len, device)

    start = time.time()

    encoder = Encoder(lang1.n_words, hidden_size, dropout_p).to(device)
    decoder = Decoder(hidden_size, lang2.n_words, dropout_p, device, max_s_len).to(device)
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    total = 0

    for e in range(1, n_epochs + 1):
        loss = training_step(dataloader, encoder, decoder, loss_fn, encoder_optimizer, decoder_optimizer)
        total += loss
        print(f"{timeSince(start, e / n_epochs)} | Avg. Epoch Loss: {loss:.5f} | Avg. Loss So Far: {total/e:.5f} [{e}/{n_epochs} {'=' * round(20 * ((e)/n_epochs))}{'-' * (20 - round(20 * (e/n_epochs)))}]\r", end='\r')
    
    print()

    print("Random evaluation:")
    lp = evaluateRandomly(encoder, decoder, pairs, lang1, lang2)
    evaluateAndShowAttention(encoder, decoder, lang1, lang2, lp[0])

    return encoder, decoder
