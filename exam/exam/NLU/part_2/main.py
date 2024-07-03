from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from utils import *
from model import *
from functions import *

PAD_TOKEN = 0

if __name__ == "__main__":
    # Defining hyperparameters
    hid_size = 200
    emb_size = 300

    lr = 0.00005 #5e-05
    clip = 5 # Clip the gradient
    weight_decay = 0.01
    n_layers = 2
    dropout = 0.1

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # uses bert-base-uncased Bert model
    model = ModelBertIAS('bert-base-uncased', out_slot, out_int, dropout=dropout, use_crf=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    
    n_epochs = 40
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1_accuracy = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, model, clip=clip)
        if x % 1 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, model, lang)

            # For decreasing the patience, using average between slot f1 and intent accuracy
            f1 = results_dev['total']['f']
            accuracy = intent_res['accuracy']
            f1_accuracy = f1 + accuracy / 2

            print('dev F1: ', f1)
            print('dev Accuracy:', accuracy)
            if f1_accuracy > best_f1_accuracy:
                best_f1_accuracy = f1_accuracy

                PATH = os.path.join("bin", "bertone.pt")
                saving_object = {"epoch": x,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "w2id": w2id,
                                "slot2id": slot2id,
                                "intent2id": intent2id}
                torch.save(saving_object, PATH)
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, model, lang)

    print("learning rate: ", lr)
    print("weight decay: ", weight_decay)
    print("hid_size: ", hid_size)
    print("emb_size: ", emb_size)
    print("dropout: ", dropout)
    print("n_layers: ", n_layers)
    print("\n")
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    # Counldn't plot the losses