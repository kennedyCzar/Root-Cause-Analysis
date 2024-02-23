#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:20:03 2023

@author: kenneth

Source: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        
        
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        self.lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(self.lstm_feats)
        return score, tag_seq
    

#%% Import data
import os
import re
from os.path import join
from itertools import chain
import pandas as pd
wdir = os.getcwd()
fdir = join(wdir, 'Root_causes_Example_N°2')

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = '<PAD>'
TAGS = [START_TAG, STOP_TAG, PAD]


#no of obs
n = 800
def dataset(fdir, n = None):
    #n is the number of subset to select if given
    if n == None:
        data = pd.read_csv(join(fdir, 'Seqviolations_ii.csv'))
    else:
        data = pd.read_csv(join(fdir, 'Seqviolations_ii.csv'))[::-n]
    return data

data = dataset(wdir, n = None)
# d = [(data.iloc[i,:][2:] == 0).all() for i in range(data.shape[0])]
# zero, one = len([i for i in d if i==True]), len([i for i in d if i==False])
# plt.bar(['All zeros', 'Violations'], [zero, one])
# plt.ylabel('Frequency', fontsize = 16)
train_x = data['Event']
train_x = [re.split('(\W)', x) for x in train_x] #use this if you dont want '-' [rx.split('-') for x in train_x]
train_y = data.iloc[:, 2:]
train_y = train_y.astype(str).apply(lambda x: ' '.join(x).split(' '), axis = 1)
training_data = [(x, y) for (x, y) in zip(train_x, [x for x in train_y.values])]
x_train, x_test = train_test_split(training_data, test_size = 0.33, random_state = 42) #tr/ts split
adata_x = sorted(list(set(list(chain(*train_x)))))
adata_y = sorted(list(set(list(chain(*train_y))))) + TAGS
word_to_ix = {} #convert strings to integer type
for word in adata_x:
    if word != '-':
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    else:
        word_to_ix[word] = len(word_to_ix)

tag_to_ix = {k:v for (k, v) in zip(adata_y, range(len(adata_y)))} #map tags tointegers also

#%%

EMBEDDING_DIM = 20
HIDDEN_DIM = 4

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

lls = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
global_step, tr_loss = 0, 0.
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in x_train:
        # print(len(sentence), len(tags))
        tags = tags + [PAD]*(len(sentence)-len(tags)) if sentence != tags else tags
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        global_step += 1
        tr_loss += loss.item()
        tr_loss /= global_step
    print(f'epoch {epoch}: {tr_loss:.3f}')
    lls.append(tr_loss)
        
#%%
# Check predictions after training
with torch.no_grad():
    print(f'Training data: {training_data[0][0]} Training Label: {training_data[0][1]}\n')
    precheck_sent = prepare_sequence(x_train[0][0], word_to_ix)
    print(f'Testing data: {training_data[0][0]} Predicted Label: {[str(x) for x in model(precheck_sent)[1] if x != list(tag_to_ix.values())[-1]]}')

#%% Prediction
from sklearn.metrics import precision_recall_fscore_support

x_targ = [x_test[i][1] for i in range(len(x_test))]

predictions, lstm_features = [], []
for i in range(len(x_test)):
    precheck_sent = prepare_sequence(x_test[i][0], word_to_ix)
    s, p = model(precheck_sent)
    lstm_features.append(model.lstm_feats.detach().numpy()[:, :1])
    predictions.append((s.item(), [str(x) for x in p if x != list(tag_to_ix.values())[-1]]))

y_targ = [predictions[i][1] for i in range(len(predictions))]

prec, rec, f1 = [], [], []

for i in range(len(predictions)):
    try:
        t_pr, t_re, t_f, _ = precision_recall_fscore_support(x_targ[i], y_targ[i], average='macro', zero_division = 0)
        prec.append(t_pr); rec.append(t_re); f1.append(t_f)
    except:
        pass

prec, rec, f1 = np.mean([prec]), np.mean([rec]), np.mean([f1])
print(f'Precision: {prec}\nRecall: {rec}\nF1-score: {f1}')

#%% In exceptional cases: some predictions may have sizes less than ground truth

x_new = [(x, y) for (x, y) in zip(x_targ, y_targ) if len(y) == 6]
x_neww = [x[0] for x in x_new]
y_neww = [x[1] for x in x_new]
print(f'Shape of x_new: {len(x_new)}\nPrint shape of y_new: {len(x_new)}\n------')
prec, rec, f1 = [], [], []

for i in range(len(y_neww)):
    t_pr, t_re, t_f, _ = precision_recall_fscore_support(x_neww[i], y_neww[i], average='macro', zero_division = 0)
    prec.append(t_pr)
    rec.append(t_re)
    beta = 0.5 #trying to find the best beta that penalizes the weight of the majority class...not sufficient
    t_f = lambda x,y: ((1+beta**2)*x*y)/((beta**2 * x) + y)
    f1.append(t_f(t_pr, t_re))

prec, rec, f1 = np.mean([prec]), np.mean([rec]), np.mean([f1])
print(f'Precision: {prec}\nRecall: {rec}\nF1-score: {f1}')


#%% Plot matrix of features
import matplotlib.pyplot as plt

ind = 27
plt.imshow(lstm_features[ind].T)
for ((j,i),label), alph  in zip(np.ndenumerate(lstm_features[ind].T), train_x[ind]):
    val = round(label, 0)
    plt.text(i,j,f'{label:.2f}', horizontalalignment='center', verticalalignment='top', fontsize = 15, c = 'r')
    plt.xticks(range(len(train_x[ind])), train_x[ind])
    plt.title(f'Best sequence score: {predictions[ind][0]:.2f}')
plt.show()
#%% Plot loss
import matplotlib.pyplot as plt
plt.plot(range(len(lls)), lls)
plt.xlabel('Training steps')
plt.ylabel('Negtative log loss (CFR)')


#%% Example results...
'''Example from Toy dataset

After Training for 100-epochs
----------------------------------
Training data: ['14', '-', '10', '-', '11', '-', '12', '-', '9', '-', '17', '-', '15', '-', '13', '-', '16'] Training Label: ['0', '0', '0', '0', '0', '0', '0', '1', '0']

Testing data: ['14', '-', '10', '-', '11', '-', '12', '-', '9', '-', '17', '-', '15', '-', '13', '-', '16'] Predicted Label: ['1', '0', '0', '0', '0', '0', '0', '0', '0']

After Training for 300-epochs
-----------------------------------
Training data: ['14', '-', '10', '-', '11', '-', '12', '-', '9', '-', '17', '-', '15', '-', '13', '-', '16'] Training Label: ['0', '0', '0', '0', '0', '0', '0', '1', '0']

Testing data: ['14', '-', '10', '-', '11', '-', '12', '-', '9', '-', '17', '-', '15', '-', '13', '-', '16'] Predicted Label: ['1', '0', '0', '0', '0', '0', '0', '1', '0']

'''

#%% Data preprocessing...Log

# log_x = pd.read_csv(join(fdir, 'Logs_parcours_complets.csv'), sep = ';')
# log_x = log_x[['ID', 'Event']]
# log_x.set_index('ID', inplace = True)
# log_xg = log_x.astype(str).groupby(['ID']).agg({'Event': lambda x: '-'.join(x)})

# #%% Violations

# viol_x = pd.read_csv(join(fdir, 'Violations.csv'), sep = ';')
# viol_x.set_index('ID', inplace = True)

# #%% Concate datasets

# data = pd.merge(log_xg, viol_x, how='inner', on=['ID'])
# data.to_csv(join(fdir, 'Seqviolations_ii.csv'))
# print(f'Shape of predictors: {log_xg.shape}\nShape of Violations: {viol_x.shape}')


#%% just checking


# data = pd.read_csv(join(fdir, 'Seqviolations.csv'))

# data_xfv = data[data['15_13'] == 1]





