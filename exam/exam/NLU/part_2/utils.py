import json
from pprint import pprint
# import random
# import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from conll import evaluate
from sklearn.metrics import classification_report


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))

pprint(tmp_train_raw[0])



# First we get the 10% of the training set, then we compute the percentage of these examples 
portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])
# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

# Intent distributions
# print('Train:')
# pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
# print('Dev:'), 
# pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
# print('Test:') 
# pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
# print('='*89)
# # Dataset size
# print('TRAIN size:', len(train_raw))
# print('DEV size:', len(dev_raw))
# print('TEST size:', len(test_raw))


w2id = {'pad':PAD_TOKEN, 'unk': 1}
slot2id = {'pad':PAD_TOKEN}
intent2id = {}
# Map the words only from the train set
# Map slot and intent labels of train, dev and test set. 'unk' is not needed.
for example in train_raw:
    for w in example['utterance'].split():
        if w not in w2id:
            w2id[w] = len(w2id)   
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)
        
for example in dev_raw:
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)
        
for example in test_raw:
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)

sent = 'I wanna a flight from Toronto to Kuala Lumpur'
mapping = [w2id[w] if w in w2id else w2id['unk'] for w in sent.split()]

# print('# Vocab:', len(w2id)-2) # we remove pad and unk from the count
# print('# Slots:', len(slot2id)-1)
# print('# Intent:', len(intent2id))

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res



train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:
      optimizer.zero_grad()

      input_ids = sample['utterances']
      attention_mask = (input_ids != PAD_TOKEN).long()
      intent_labels = sample['intents']
      slot_labels = sample['y_slots']

      _,_, loss = model(input_ids = input_ids, attention_mask = attention_mask, intent_labels = intent_labels, slot_labels = slot_labels)
      loss_array.append(loss)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
      for sample in data:
        input_ids = sample['utterances']
        attention_mask = (input_ids != PAD_TOKEN).long()
        intent_labels = sample['intents']
        slot_labels = sample['y_slots']

        intent_logits, slot_logits, loss = model(input_ids = input_ids, attention_mask = attention_mask, intent_labels = intent_labels, slot_labels = slot_labels)
        loss_array.append(loss)

        out_intents = [lang.id2intent[x] 
                       for x in torch.argmax(intent_logits, dim=1).tolist()]
        gt_intents = [lang.id2intent[x] for x in intent_labels.tolist()]
        ref_intents.extend(gt_intents)
        hyp_intents.extend(out_intents)
        output_slots = torch.argmax(slot_logits, dim=2)
        for id_seq, seq in enumerate(output_slots):
          length = sample['slots_len'].tolist()[id_seq]
          utt_ids = input_ids[id_seq][:length].tolist()
          gt_ids = slot_labels[id_seq].tolist()
          gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
          utterance = [lang.id2word[elem] for elem in utt_ids]
          to_decode = seq[:length].tolist()
          ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
          tmp_seq = []
          for id_el, elem in enumerate(to_decode):
            tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
          hyp_slots.append(tmp_seq)

    try:
      results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
          print("Warning:", ex)
          ref_s = set([x[1] for x in ref_slots])
          hyp_s = set([x[1] for x in hyp_slots])
          print(hyp_s.difference(ref_s))
          results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

    return results, report_intent, loss_array