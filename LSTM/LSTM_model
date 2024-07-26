import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class DKT_Model(torch.nn.Module):
    def __init__(self, emb_size, skill_num, question_num, hidden_size, num_layer, batch_size, dropout_rate, time_model=None):
        super(DKT_Model, self).__init__()
        self.question_num=question_num
        self.skill_num=skill_num
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.batch_size=batch_size
        self.dropout_rate=dropout_rate
        self.time_model=time_model

 
       # define the layers
        self.question_embeddings = torch.nn.Embedding(self.question_num * 2, self.emb_size) # embedding layer is quest_num*2 (which is feature length) by emb dim
        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size) # embedding layer is skill_num*2 (which is feature length) by emb dim

        input_size=self.emb_size+1 if self.time_model else self.emb_size
        self.rnn = torch.nn.LSTM(
            input_size=input_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layer
        )
        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')
    
        # initialize weight matrices with Glorot (w_ih is input matrices for input, forget and output; w_hh is for hidden
        # DECIDE IF WE NEED THIS, or whether uniform default is enough
        # for name, param in self.rnn.named_parameters():
        #     if 'weight_ih' in name:
        #         torch.nn.init.xavier_uniform_(param.data)
        #     elif 'weight_hh' in name:
        #         torch.nn.init.xavier_uniform_(param.data)
        #     elif 'bias' in name:
        #         param.data.fill_(0)
  
    # forward pass
    # 1) get embeddings, 2) pack the sequences together, 3) then pass through lstm layer, 4) unpack, 5) then linear output layer
    def forward(self, feed_dict):
        lengths = feed_dict['seq_length'].cpu()
        user_id = feed_dict['user_id']
        seq_sorted = feed_dict['question']
        labels_sorted = feed_dict['correct']
        orig_idx = feed_dict['orig_idx']
        seconds_between_attempts = feed_dict['seconds_between_attempts']
            
        embed_history_i = self.question_embeddings(seq_sorted + labels_sorted * self.skill_num)  
        if self.time_model:
            embed_history_i = torch.cat([embed_history_i, seconds_between_attempts.unsqueeze(-1)],axis=-1)
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True)
        # note None in rnn() is set for tuple (h, c), this sets hidden to 0 each batch (forgets the state). This is in place of init_hidden()
        output, hidden = self.rnn(embed_history_i_packed, None) 
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) 
        if self.training:
            output = self.dropout(output)
        # we need to solve for pred_vector being seq_length-1. One way is to expand forward to include if self.training: length-1 else length
        # this will create two equivalent chunks with packing, lstm and unpacking, but one with shortened length, and one full
        pred_vector = self.out(output) 

        target_item = seq_sorted[:, 1:] 
        label = labels_sorted[:, 1:] 

        # here predictions and labels are sorted by seq length descending
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1) 
        prediction_sorted = torch.sigmoid(prediction_sorted)
        # resort them into original order and create out_dict
        orig_idx_tensor = torch.tensor(orig_idx)
        rearanged_pred = prediction_sorted.index_select(0, orig_idx_tensor)
        rearanged_label = label.index_select(0,orig_idx_tensor).double()
        out_dict = {'prediction': rearanged_pred, 'label': rearanged_label,
                    'prediction_sorted': prediction_sorted, 'label_sorted':label, 
                    'pred_lengths': lengths-1, 'full_output': torch.sigmoid(pred_vector)}
        return out_dict, pred_vector
   
      
#     def dkt_loss_function(self, out_dict, feed_dict):
#         lengths = feed_dict['seq_length'] - 1
#         # Sort lengths in descending order, and get the indices
#         _, idx_sort = torch.sort(lengths, dim=0, descending=True)
#         # Use the indices to sort predictions, labels, and lengths
#         out_dict_sorted = {k : v.index_select(0, idx_sort) for k, v in out_dict.items()}
#         labels = out_dict_sorted['label']
#         predictions = out_dict_sorted['prediction']
#         labels = torch.nn.utils.rnn.pack_padded_sequence(labels, lengths.cpu().detach(), batch_first=True).data
#         predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, lengths.cpu().detach(), batch_first=True).data
#         dkt_loss = self.loss_function(predictions.float(),labels.float()) 
#         return dkt_loss, labels, predictions
 

    def dkt_loss_function(self, out_dict, feed_dict):
        labels = out_dict['label_sorted']
        predictions = out_dict['prediction_sorted']
        pred_lengths = out_dict['pred_lengths']
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, pred_lengths.cpu().detach(), batch_first=True).data
        predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, pred_lengths.cpu().detach(), batch_first=True).data
        dkt_loss = self.loss_function(predictions.float(),labels.float()) 
        return dkt_loss, labels, predictions
 
