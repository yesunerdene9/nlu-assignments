import torch
import torch.nn as nn
import torch.optim as optim

import copy
import math
import numpy as np
from tqdm import tqdm

from utils import *
from functions import *

from model import LM_RNN

if __name__ == "__main__":
    # Defining hyperparameters 
    lr = 0.9
    clip = 5
    n_layers = 2

    hid_size = 300
    emb_size = 300

    dropout = 0.8
    
    momentum = 0.9
    weight_decay = 0.00001
    vocab_len = len(lang.word2id)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    # optimizer SGD
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Applying Non-Monotonic Triggered AvSGD optimizer
    # optimizer = NonMonotonicTriggeredAvSGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    patience = 5
    n_epochs = 100
    losses_dev = []
    losses_train = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3

                # To save the model
                # path = 'bin/lm_model_part1.pt'
                # torch.save(model.state_dict(), path)
    
            else:
                patience -= 1

            if patience <= 0:
                break

    best_model.to(device)

    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    print('\n\nparams:')
    print('learning rate ', lr)
    print('dropout prob ', dropout)
    print('hidden size ', hid_size)
    print('embedded size ', emb_size)
    print('momentum ', momentum)
    print('weight decay ', weight_decay)
    print('number of layers ', n_layers)
    
    print('\n')