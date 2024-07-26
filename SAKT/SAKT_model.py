# Full model with padding mask, attn mask and positional enc.
#
# To cross the training/serving boundary for pytorch models, we need to save the model as an artifact in GCS.
# we do this simply by saving the model module to GCS and then declaring it as an artifact with wandb
# during a training run.
#
# We'll want to be careful that any declarations of model architecture do not include relative imports
# (eg: from khan_recommender_training.models import SAKT) as this will not work when we import the model
# in a serving context.

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))


class SAKT(torch.nn.Module):
    def __init__(self, num_layer, num_head, skill_num, question_num, emb_size, dropout, max_seq_length, skills_as_keys=True, time_model=False):
        """
        Initialize a SAKT model with random weights.
        Default "skills_as_keys=True" means that keys and values are skill-based (vs. question-based)
        """
        super().__init__()
        self.skill_num = skill_num
        self.emb_size = emb_size
        self.d_model = emb_size+1 if time_model else emb_size
        self.num_head = num_head
        self.dropout = dropout
        self.question_num=question_num
        self.max_seq_length=max_seq_length
        self.skills_as_keys = skills_as_keys
        self.time_model=time_model

        # check that the model dimension is a multiple of num_head
        err_msg = f"embedding size {'+1'*int(time_model)} must be a multiple of num_head"
        assert self.d_model/self.num_head == self.d_model//self.num_head, err_msg
        
        # initialize weights for each step of the network
        if self.skills_as_keys:
            self.inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size) # skill based attention
        else:
            self.inter_embeddings = nn.Embedding(self.question_num * 2, self.emb_size) # question based attention

        self.embd_pos = torch.nn.Embedding(self.max_seq_length , self.d_model) # positional emb layer

        self.attn_blocks = nn.ModuleList([
            TransformerLayer(d_model=self.d_model, d_feature=self.d_model//self.num_head, d_ff=self.d_model,
                             dropout=self.dropout, n_heads=self.num_head)
            for _ in range(num_layer)
        ])

        self.question_embeddings = nn.Embedding(self.question_num, self.emb_size) # predict questions
        self.out = nn.Linear(self.d_model, self.question_num) # output dim is BS X max_seq_len X question_num
        self.loss_function = nn.BCELoss(reduction='mean')

    def forward(self, feed_dict):
        skill_seq = feed_dict['inter_seq']          # batch size by max seq len; these are skill_id seqs
        question_seq = feed_dict['quest_seq']       # same dim as above,  problem_id/item_id seqs
        labels = feed_dict['label_seq']                  # batch size by max seq len; correct/incorrect
        seconds_between_attempts = feed_dict['seconds_between_attempts']

        mask_labels = labels * (labels > -1).long()

        if self.skills_as_keys:
            seq_data = self.inter_embeddings(skill_seq + mask_labels * self.skill_num) # keys and values input for skills
        else:
            seq_data = self.inter_embeddings(question_seq + mask_labels * self.question_num) # keys and values input for questions

        q_data = self.question_embeddings(question_seq) # query input for question based model

        if self.time_model:
            seq_data = torch.cat([seq_data, seconds_between_attempts.unsqueeze(-1)],axis=-1)
            q_data = torch.cat([q_data, seconds_between_attempts.unsqueeze(-1)],axis=-1)
        
        # here we shift question embedding by 1,
        # so that in attention layer we compute similarity between previous skill interaction and next question
        zero_data = torch.zeros_like(q_data)[:,:1, :].to(device)
        q_data = torch.cat([q_data[:, 1:, :], zero_data], dim=1)

        # positional encoding is for keys and values, not for queries 
        pos = self.embd_pos(torch.arange(labels.shape[1]).to(device))
        seq_data = seq_data + pos
        y = seq_data

        # padding mask for attention needs to be defined here
        pad_token = -1
        pad_mask = (labels != pad_token)
        pad_mask = pad_mask.to(device)

        for block in self.attn_blocks:
            y = block(mask_val=1, query=q_data, key=y, values=y, padding_mask=pad_mask)

        # output below has two pieces: 1-full model output, 2-tr_preds for loss
        # 1 - full model output, dims bs X max_seq_len X question_num
        full_model_output = self.out(y).squeeze(-1).sigmoid()


        # 2 - for model training we need to extract the response to the correct item at each seq step
        # next question seen (modeling how will a student do on next question)
        target_item = (question_seq[:, 1:]).unsqueeze(dim=-1)
        target_labels = labels[:, 1:].double()
        tr_preds = torch.gather(full_model_output, dim=-1, index=target_item).squeeze(dim=-1)
        
        # Note the shifting of predictions and labels here, to predict the next item
        out_dict = {'prediction': tr_preds, 'label': target_labels, 'full_output':full_model_output}

        return out_dict

    def loss(self, feed_dict, outdict):
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        mask = label > -1
        loss = self.loss_function(prediction[mask], label[mask])
        return loss

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout):
        super().__init__()
        """Transformer block - multihead attention, residual connection, layer norm, feedforward and dropout."""
        
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask_val, query, key, values, padding_mask): # First time I add padding mask is here
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask_val).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0)
        src_mask = src_mask.to(device)

        query2 = self.masked_attn_head(query, key, values, padding_mask=padding_mask, attn_mask=src_mask)
        query2 = query + self.dropout1(query2)  # residual connection
        query2 = self.layer_norm1(query2)

        query3 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
        query3 = query2 + self.dropout2(query3)  # residual connection
        query3 = self.layer_norm2(query3)
        return query3

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
        super().__init__()
        """
        Projection layer keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, attn_mask, padding_mask):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, attn_mask, padding_mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).reshape(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output

    def attention(self, q, k, v, d_k, attn_mask, padding_mask, dropout):
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # BS, head, seqlen, seqlen

        # apply padding and then attention mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        scores.masked_fill_(padding_mask == 0, -np.inf)
        scores.masked_fill_(attn_mask == 0, -np.inf)

        # softmax
        scores = F.softmax(scores, dim=-1)  # BS,head,seqlen,seqlen
        scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
