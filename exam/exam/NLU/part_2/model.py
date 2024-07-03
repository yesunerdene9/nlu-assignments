import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchcrf import CRF
from transformers import BertModel, BertTokenizer

PAD_TOKEN = 0

class ModelBertIAS(nn.Module):
    def __init__(self, bert_model_name, out_slot, out_int, dropout=0.1, use_crf=True):
        super(ModelBertIAS, self).__init__()
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)

        ###
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        self.slot_out = nn.Linear(hidden_size, out_slot)
        self.intent_out = nn.Linear(hidden_size, out_int)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.use_crf = use_crf

        if self.use_crf:
            self.crf = CRF(num_tags=out_slot, batch_first = True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0] #outputs.last_hidden_state  # Sequence of hidden-states at the output of the last layer of the model
        pooled_output = outputs[1] #outputs.pooler_output  # Last layer hidden-state of the first token of the sequence (classification token)

        # Apply dropout to each output
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_out(sequence_output)
        intent_logits = self.intent_out(pooled_output)


        total_loss = 0

        if intent_labels is not None and slot_labels is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
            intent_loss = intent_loss_fct(intent_logits.view(-1, out_int), intent_labels.view(-1))

            if self.use_crf:
                slot_loss = -self.crf(slot_logits, slot_labels, mask = attention_mask.byte(), reduction = 'mean')

            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, out_slot), slot_labels.view(-1))
                
            total_loss = intent_loss + slot_loss

        return intent_logits, slot_logits, total_loss

        # # Slot size: batch_size, seq_len, classes
        # slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # # Slot size: batch_size, classes, seq_len
        # return slots, intent