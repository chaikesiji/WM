"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import torch
import torch.nn as nn
from model_transformer import MultiHeadAttention, PositionwiseFeedForward


class Single_Att_Layer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Single_Att_Layer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q_input, k_input, v_input, slf_attn_mask=None, pad_mask=None, is_decode=False):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask, pad_mask=pad_mask, is_decode=is_decode)
        # non_pad_mask = ~pad_mask
        # enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask.float()
        return enc_output, enc_slf_attn


class Self_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """

    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, input_feats, mask):
        # input_feats = input_feats.split(num_objs, dim=0)
        # input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)
        #
        # # -- Prepare masks
        # bsz = len(num_objs)
        # device = input_feats.device
        # pad_len = max(num_objs)
        # num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        # slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(
        #     1).expand(-1, pad_len, -1)  # (bsz, pad_len, pad_len),把（bsz,1,pad_len)中每一个(bsz,1,1)复制pad_len遍
        # non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(
        #     num_objs_).unsqueeze(-1)  # (bsz, pad_len, 1)
        expand_dim = mask.shape[1]
        mask = mask.unsqueeze(-1)
        slf_att_mask = mask.expand(-1, -1, expand_dim).transpose(1, 2).contiguous()
        slf_other_mask = mask.expand(-1, -1, expand_dim)
        slf_att_mask = torch.mul(slf_att_mask, slf_other_mask)
        slf_attn_mask = ~slf_att_mask.to(torch.bool)
        pad_mask = ~mask.to(torch.bool)
        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
            input_feats, input_feats, input_feats,
            slf_attn_mask=slf_attn_mask,
            pad_mask=pad_mask)
        return enc_output


class Cross_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """

    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, x_feats, other_feats, x_mask, other_mask, is_decode=False):
        x_mask, other_mask = x_mask.unsqueeze(-1), other_mask.unsqueeze(-1)
        # expand_dim = x_mask.shape[1]
        # x_expand_mask = x_mask.expand(-1, -1, expand_dim).transpose(1, 2).contiguous()
        # other_expand_msk = other_mask.expand(-1, -1, expand_dim).transpose(1, 2).contiguous()
        # x_expand_mask = x_expand_mask.permute(0, 2, 1).contiguous()
        # x_cross_mask = torch.mul(x_expand_mask, other_expand_msk)
        # x_cross_mask = ~x_cross_mask.to(torch.bool)
        x_cross_mask = x_mask * other_mask.transpose(1, 2).contiguous()
        x_cross_mask = ~x_cross_mask.to(torch.bool)
        pad_mask = ~x_mask.to(torch.bool)
        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
            x_feats, other_feats, other_feats,
            slf_attn_mask=x_cross_mask,
            pad_mask=pad_mask,
            is_decode=is_decode)
        return enc_output
