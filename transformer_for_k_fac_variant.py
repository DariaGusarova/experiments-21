import numpy as np
import torch
import torch.nn as nn

common_type = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# реализация nn.Linear с сохранением forward и backward статистик
class MyLinear(nn.Module):
    def __init__(self, d_in, d_out, exp_eps, bias=True):
        super().__init__()
        self.bias = bias
        self.d_out = d_out
        self.exp_eps = exp_eps
        self.k = 1
        self.first, self.even = True, False
        self.linear = nn.Linear(d_in, d_out, bias)
        if bias:
            self.forward_st = nn.Parameter(torch.zeros(d_in + d_out, d_in + d_out), requires_grad=False)
        else:
            self.forward_st = nn.Parameter(torch.zeros(d_in, d_in), requires_grad=False)  
        self.backward_st = nn.Parameter(torch.zeros(d_out, d_out), requires_grad=False)      

    def forward(self, x):
        batch_size, length, d = x.size()
        x_ = x
        if self.bias:
            ones = torch.ones(batch_size, length, self.d_out).to(device)
            x_ = torch.cat([x, ones], 2)
            d += self.d_out    
        st = torch.mean(x_.reshape((batch_size, length, d, 1)) * x_.reshape((batch_size, length, 1, d)), (0, 1))
        eps = min(1. - 1./self.k, self.exp_eps)
        self.k += 1
        self.forward_st.mul_(eps).add_(1. - eps, st.detach())   
        return self.linear(x)         


# реализация nn.Linear внутри Attention с частичным сохранением forward и backward статистик 
class QKVLinear(nn.Module):
    def __init__(self, dmodel, dk, dv, exp_eps, kv=True, q=False, kp_forward=False):
        super().__init__()
        self.q = q
        self.kv = kv
        self.exp_eps = exp_eps
        self.k = 1.
        self.kp_forward = kp_forward
        self.first, self.even = True, False
        d_out = 0
        if kv:
            self.klinear = nn.Linear(dmodel, dk, False)
            self.vlinear = nn.Linear(dmodel, dv, False)
            d_out += dk + dv
        if q:
            self.qlinear = nn.Linear(dmodel, dk, False)    
            d_out += dk
        if kp_forward:
            self.forward_st = nn.Parameter(torch.zeros(dmodel, dmodel), requires_grad=False)  
        self.backward_st = nn.Parameter(torch.zeros(d_out, d_out), requires_grad=False)    

    def forward(self, x):
        if self.kp_forward:
            batch_size, length, d = x.size()
            st = torch.mean(x.reshape((batch_size, length, d, 1)) * x.reshape((batch_size, length, 1, d)), (0, 1))
            eps = min(1. - 1./self.k, self.exp_eps)
            self.k += 1
            self.forward_st.mul_(eps).add_(1. - eps, st.detach())
        if self.q and self.kv:          
            return torch.cat([self.qlinear(x), self.klinear(x), self.vlinear(x)], 2)
        if self.kv:
            return torch.cat([self.klinear(x), self.vlinear(x)], 2) 
        if self.q:
            return self.qlinear(x)              
        
        
# для обновления статистик при обратном проходе
def hook_fn(self, grad_input, grad_output):
    x = grad_output[0]
    batch_size, length, d = x.size()
    st = torch.mean(x.reshape((batch_size, length, d, 1)) * x.reshape((batch_size, length, 1, d)), (0, 1))
    if self.even:
        if self.first:
            self.first = False
            self.backward_st.mul_(0.).add_(1., st.detach())
        else:
            self.backward_st.mul_(self.exp_eps).add_(1. - self.exp_eps, st.detach()) 
        self.even = False
    else:
        self.even = True           
    return   


class MultiHeadAttention(nn.Module):
    def __init__(self, exp_eps, h=8, dmodel=512, dk=64, dv=64, qkv=True):
        super().__init__()
        self.h = h
        self.dk = dk
        self.dv = dv
        self.qkv = qkv
        self.wo = MyLinear(h * dv, dmodel, exp_eps, bias=False)
        self.wo.register_backward_hook(hook_fn)
        self.ws = nn.ModuleList()
        if qkv == False:
            self.wqs = nn.ModuleList()
        first = True
        for i in range(h):
            if qkv:
                self.ws.append(QKVLinear(dmodel, dk, dv, exp_eps, q=True, kv=True, kp_forward=first))            
                self.ws[-1].register_backward_hook(hook_fn)
            else:
                self.ws.append(QKVLinear(dmodel, dk, dv, exp_eps, q=False, kv=True, kp_forward=first))
                self.ws[-1].register_backward_hook(hook_fn)
                self.wqs.append(QKVLinear(dmodel, dk, dv, exp_eps, q=True, kv=False, kp_forward=first))
                self.wqs[-1].register_backward_hook(hook_fn)  
            first = False     
        
    def forward(self, x, queries=None, has_mask=False):
        # .shape - (batch_size, length, dmodel)
        if has_mask:
            length = x.size()[1]
            mask = np.zeros((length, length))
            for i in range(length):
                mask[i, i+1:] = -np.inf
            mask = mask.reshape(1, length, length)
            mask = torch.tensor(mask, dtype=common_type).to(device)
        
        multihead = []
        for i in range(self.h):
            if self.qkv:
                linear_part = self.ws[i](x)
                query, key, value = linear_part[:, :, :self.dk], linear_part[:, :, self.dk:2*self.dk], linear_part[:, :, 2*self.dk:]
            else:
                linear_part = self.ws[i](x)
                key, value = linear_part[:, :, :self.dk], linear_part[:, :, self.dk:]
                query = self.wqs[i](queries)
            weight = torch.matmul(query, torch.transpose(key, 1, 2)) # weight.shape - (batch_size, length, length)
            weight = weight / np.sqrt(self.dk)
            # mask.shape - (1, length, length)
            if has_mask:
                weight = weight + mask
            weight = nn.Softmax(dim=2)(weight) 
            attention = torch.matmul(weight, value) # attention.shape - (batch_size, length, dv)
            multihead.append(attention)
        multihead = torch.cat(multihead, 2) # multihead.shape - (batch_size, length, h*dv)
        answer = self.wo(multihead) # answer.shape - (batch_size, length, dmodel)
        return answer    
    
    
class FeedForward(nn.Module):
    def __init__(self, exp_eps, dmodel=512, dff=2048):
        super().__init__()
        self.linear1 = MyLinear(dmodel, dff, exp_eps)
        self.linear1.register_backward_hook(hook_fn)
        self.linear2 = MyLinear(dff, dmodel, exp_eps)
        self.linear2.register_backward_hook(hook_fn)
    
    def forward(self, x):
        # x.shape - (batch_size, length, dmodel)
        l1 = self.linear1(x)
        l2 = self.linear2(nn.ReLU()(l1))
        return l2      
    
    
class EncoderBlock(nn.Module):
    def __init__(self, exp_eps, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        # LayerNorm (Learnable Parameters)?
        self.pdropout = pdropout
        self.multiheadattention = MultiHeadAttention(exp_eps, h, dmodel, dk, dv, qkv=True)
        self.feedforward = FeedForward(exp_eps, dmodel, dff)
        
    def forward(self, x):
        x_normed = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        x_multihead = self.multiheadattention(x_normed)
        x_dropout = nn.Dropout(p=self.pdropout)(x_multihead)
        current = x + x_dropout
        
        current_normed = nn.LayerNorm(current.size()[1:], elementwise_affine=False)(current)
        current_feedforward = self.feedforward(current_normed)
        current_dropout = nn.Dropout(p=self.pdropout)(current_feedforward)
        result = current + current_dropout
        return result
    
    
class Encoder(nn.Module):
    def __init__(self, exp_eps, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.enc_block1 = EncoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
        self.enc_block2 = EncoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
        self.enc_block3 = EncoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
            
    def forward(self, x):
        result = self.enc_block3(self.enc_block2(self.enc_block1(x)))
        return result  
    
class DecoderBlock(nn.Module):
    def __init__(self, exp_eps, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.pdropout = pdropout
        self.maskedmultiheadattention = MultiHeadAttention(exp_eps, h, dmodel, dk, dv, qkv=True)
        self.multiheadattention = MultiHeadAttention(exp_eps, h, dmodel, dk, dv, qkv=False)
        self.feedforward = FeedForward(exp_eps, dmodel, dff)
        
    def forward(self, x_enc, x_dec):
        x_dec_normed = nn.LayerNorm(x_dec.size()[1:], elementwise_affine=False)(x_dec)
        x_maskedmultihead = self.maskedmultiheadattention(x_dec_normed, has_mask=True)
        x_dropout = nn.Dropout(p=self.pdropout)(x_maskedmultihead)
        current = x_dec + x_dropout
    
        # queries, keys, values
        current_normed = nn.LayerNorm(current.size()[1:], elementwise_affine=False)(current)
        x_enc_normed = nn.LayerNorm(x_enc.size()[1:], elementwise_affine=False)(x_enc)
        current_multihead = self.multiheadattention(x_enc_normed, queries=current_normed)
        current_dropout = nn.Dropout(p=self.pdropout)(current_multihead)
        result = current_dropout + current
        
        result_normed = nn.LayerNorm(result.size()[1:], elementwise_affine=False)(result)
        result_feedforward = self.feedforward(result_normed)
        result_dropout = nn.Dropout(p=self.pdropout)(result_feedforward)
        result_total = result + result_dropout
        return result_total
    
    
class Decoder(nn.Module):
    def __init__(self, exp_eps, h=8, dmodel=512, dk=64, dv=64, dff=2048, pdropout=0.1):
        super().__init__()
        self.dec_block1 = DecoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
        self.dec_block2 = DecoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
        self.dec_block3 = DecoderBlock(exp_eps, h, dmodel, dk, dv, dff, pdropout)
            
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
    