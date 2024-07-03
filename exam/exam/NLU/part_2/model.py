import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel

PAD_TOKEN = 0

##### For adapting the Bert model,
# the github repository JointBERT was used (https://github.com/monologg/JointBERT/blob/master)
# which is the one implementation of the paper BERT for Joint (https://arxiv.org/pdf/1902.10909),
# for getting the primary indication/idea on
# how to adapt pre-trained language model and
# fine-tune it on specific tasks on actual code.

class ModelBertIAS(nn.Module):
    def __init__(self, bert_model_name, out_slot, out_int, dropout=0.1, use_crf=True):
        super(ModelBertIAS, self).__init__()
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)

        self.out_int = out_int
        self.out_slot = out_slot

        # Adapting the pre-trained Bert model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Using the size of the hidden state of Bert
        hidden_size = self.bert.config.hidden_size

        self.slot_out = nn.Linear(hidden_size, out_slot)
        self.intent_out = nn.Linear(hidden_size, out_int)

        # Dropout layers - created separate dropout layers although the porbability is same
        self.dropout_slot = nn.Dropout(dropout)
        self.dropout_int = nn.Dropout(dropout)
        self.use_crf = use_crf

        if self.use_crf:
            self.crf = CRF(num_tags=out_slot, batch_first = True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Sequence of hidden-states at the output of the last layer of the model
        sequence_output = outputs[0] # last_hidden_stat

        # Last layer hidden-state of the first token of the sequence (classification token)
        pooled_output = outputs[1] # pooler_output  

        # Applying dropout to each output
        sequence_output = self.dropout_slot(sequence_output)
        pooled_output = self.dropout_int(pooled_output)

        slots = self.slot_out(sequence_output)
        intents = self.intent_out(pooled_output)

        total_loss = 0

        if intent_labels is not None :
            if self.out_int == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intents.view(-1), intent_labels.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intents.view(-1, self.out_int), intent_labels.view(-1))

            total_loss = intent_loss + total_loss
            
        if slot_labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)

            if self.use_crf:
                slot_loss = -self.crf(slots, slot_labels, mask = attention_mask.byte(), reduction = 'mean')
            else:
                slot_loss = slot_loss_fct(slots.view(-1, self.out_slot), slot_labels.view(-1))
                
            total_loss = intent_loss + slot_loss

        return intents, slots, total_loss