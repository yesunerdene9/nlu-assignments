from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import *
from model import *
from functions import *

if __name__ == "__main__":

    # Defining hyperparameters

    clip = 5
    lr = 0.001
    n_layers = 2

    hid_size = 200
    emb_size = 300

    dropout = 0.1
    weight_decay = 0.01

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layers, pad_index=PAD_TOKEN, dropout=dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr,  weight_decay=weight_decay)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    patience = 3
    n_epochs = 200
    losses_dev = []
    losses_train = []
    sampled_epochs = []
    best_f1_accuracy = 0

    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots,
                        criterion_intents, model, clip=clip)

        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            # For decreasing the patience using average between slot f1 and intent accuracy
            f1 = results_dev['total']['f']
            accuracy = intent_res['accuracy']
            f1_accuracy = f1 + accuracy / 2

            
            if f1_accuracy > best_f1_accuracy:
                best_f1_accuracy = f1_accuracy

                # Save the model
                PATH = os.path.join("bin", "lstm_model_part1.pt")
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

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                            criterion_intents, model, lang)

    print("learning rate: ", lr)
    print("weight decay: ", weight_decay)
    print("hid_size: ", hid_size)
    print("emb_size: ", emb_size)
    print("dropout: ", dropout)
    print("n_layers: ", n_layers)
    print("\n")
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])


    # Plot the losses
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()