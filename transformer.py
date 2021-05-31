import numpy as np
import torch
import torch.nn as nn

common_type = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, dmodel=512, dk=64, dv=64):
        super().__init__()
        self.h = h
        self.dk = dk
        self.dv = dv
        self.wo = nn.Linear(h * dv, dmodel)
        self.wqs = nn.ModuleList()
        self.wks = nn.ModuleList()
        self.wvs = nn.ModuleList()
        for i in range(h):
            self.wqs.append(nn.Linear(dmodel, dk))
            self.wks.append(nn.Linear(dmodel, dk))
            self.wvs.append(nn.Linear(dmodel, dv))
        
    def forward(self, queries, keys, values, has_mask=False):
        # .shape - (batch_size, length, dmodel)
        if has_mask:
            length = values.size()[1]
            mask = np.zeros((length, length))
            for i in range(length):
                mask[i, i+1:] = -np.inf
            mask = mask.reshape(1, length, length)
            mask = torch.tensor(mask, dtype=common_type).to(device)
        
        multihead = []
        for i in range(self.h):
            query = self.wqs[i](queries) # query.shape - (batch_size, length, dk)
            key = self.wks[i](keys) # key.shape - (batch_size, length, dk)
            weight = torch.matmul(query, torch.transpose(key, 1, 2)) # weight.shape - (batch_size, length, length)
            weight = weight / np.sqrt(self.dk)
            
            # mask.shape - (1, length, length)
            if has_mask:
                weight = weight + mask
                
            weight = nn.Softmax(dim=2)(weight) 
            value = self.wvs[i](values) # value.shape - (batch_size, lenght, dv)
            attention = torch.matmul(weight, value) # attention.shape - (batch_size, length, dv)
            multihead.append(attention)
        multihead = torch.cat(multihead, 2) # multihead.shape - (batch_size, length, h*dv)
        answer = self.wo(multihead) # answer.shape - (batch_size, length, dmodel)
        return answer    
    
    
class FeedForward(nn.Module):
    def __init__(self, dmodel=512, dff=2048):
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff)
        self.linear2 = nn.Linear(dff, dmodel)
    
    def forward(self, x):
        # x.shape - (batch_size, length, dmodel)
        l1 = self.linear1(x)
        l2 = self.linear2(nn.ReLU()(l1))
        return l2      
    
    
class EncoderBlock(nn.Module):
    def __init__(self, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        # LayerNorm (Learnable Parameters)?
        self.pdropout = pdropout
        self.multiheadattention = MultiHeadAttention(h, dmodel, dk, dv)
        self.feedforward = FeedForward(dmodel, dff)
        
    def forward(self, x):
        x_normed = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        x_multihead = self.multiheadattention(x_normed, x_normed, x_normed)
        x_dropout = nn.Dropout(p=self.pdropout)(x_multihead)
        current = x + x_dropout
        
        current_normed = nn.LayerNorm(current.size()[1:], elementwise_affine=False)(current)
        current_feedforward = self.feedforward(current_normed)
        current_dropout = nn.Dropout(p=self.pdropout)(current_feedforward)
        result = current + current_dropout
        return result
    
    
class Encoder(nn.Module):
    def __init__(self, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.enc_block1 = EncoderBlock(h, dmodel, dk, dv, dff, pdropout)
        self.enc_block2 = EncoderBlock(h, dmodel, dk, dv, dff, pdropout)
        self.enc_block3 = EncoderBlock(h, dmodel, dk, dv, dff, pdropout)
            
    def forward(self, x):
        result = self.enc_block3(self.enc_block2(self.enc_block1(x)))
        return result  
    
class DecoderBlock(nn.Module):
    def __init__(self, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.pdropout = pdropout
        self.maskedmultiheadattention = MultiHeadAttention(h, dmodel, dk, dv)
        self.multiheadattention = MultiHeadAttention(h, dmodel, dk, dv)
        self.feedforward = FeedForward(dmodel, dff)
        
    def forward(self, x_enc, x_dec):
        x_dec_normed = nn.LayerNorm(x_dec.size()[1:], elementwise_affine=False)(x_dec)
        x_maskedmultihead = self.maskedmultiheadattention(x_dec_normed, x_dec_normed, x_dec_normed, has_mask=True)
        x_dropout = nn.Dropout(p=self.pdropout)(x_maskedmultihead)
        current = x_dec + x_dropout
    
        # queries, keys, values
        current_normed = nn.LayerNorm(current.size()[1:], elementwise_affine=False)(current)
        x_enc_normed = nn.LayerNorm(x_enc.size()[1:], elementwise_affine=False)(x_enc)
        current_multihead = self.multiheadattention(current_normed, x_enc_normed, x_enc_normed)
        current_dropout = nn.Dropout(p=self.pdropout)(current_multihead)
        result = current_dropout + current
        
        result_normed = nn.LayerNorm(result.size()[1:], elementwise_affine=False)(result)
        result_feedforward = self.feedforward(result_normed)
        result_dropout = nn.Dropout(p=self.pdropout)(result_feedforward)
        result_total = result + result_dropout
        return result_total
    
    
class Decoder(nn.Module):
    def __init__(self, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.dec_block1 = DecoderBlock(h, dmodel, dk, dv, dff, pdropout)
        self.dec_block2 = DecoderBlock(h, dmodel, dk, dv, dff, pdropout)
        self.dec_block3 = DecoderBlock(h, dmodel, dk, dv, dff, pdropout)
            
    def forward(self, x_enc, x_dec):
        result1 = self.dec_block1(x_enc, x_dec)
        result2 = self.dec_block2(x_enc, result1)
        result = self.dec_block3(x_enc, result2)
        return result  
    

class PositionalEncoding(nn.Module):
    def __init__(self, length, dmodel):
        super().__init__()
        pos_part = np.tile(np.arange(length), dmodel).reshape(dmodel, length).T
        i_part = np.tile(np.tile(np.arange(int(dmodel/2)) * 2, (2, 1)).T.ravel(), length).reshape(length, dmodel)
        mask = pos_part * 10**(4 * i_part/dmodel)
        mask[:, ::2] = np.sin(mask[:, ::2])
        mask[:, 1::2] = np.cos(mask[:, 1::2])
        mask = mask.reshape(1, length, dmodel)
        self.mask = torch.tensor(mask, dtype=common_type).to(device)
        
    def forward(self, x):
        result = x + self.mask
        return result    
    