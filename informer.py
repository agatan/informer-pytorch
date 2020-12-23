import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads
        self.inv_sqrt_d_k = self.depth ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        query,
        key,
        value,
        prev_attn_logits=None,
        key_padding_mask=None,
        attn_mask=None,
    ):
        """
        Arguments:
            query: (L, N, E)
            key: (S, N, E)
            value: (S, N, E)
            prev_attn_logits: (N * H, L, S)
            key_padding_mask: (N, S)
            attn_mask: (L, S)
        """
        batch_size = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)
        query = self.q_proj(query)  # (L, N, E)
        key = self.k_proj(key)  # (S, N, E)
        value = self.v_proj(value)  # (S, N, E)
        query = (
            query.contiguous()
            .view(-1, batch_size * self.num_heads, self.depth)
            .transpose(0, 1)
        )  # (N * H, L, E')
        key = (
            key.contiguous()
            .view(-1, batch_size * self.num_heads, self.depth)
            .transpose(0, 1)
        )  # (N * H, S, E')
        value = (
            value.contiguous()
            .view(-1, batch_size * self.num_heads, self.depth)
            .transpose(0, 1)
        )  # (N * H, S, E')

        attn_logits = (
            torch.bmm(query, key.transpose(1, 2)) * self.inv_sqrt_d_k
        )  # (N * H, L, S)
        if prev_attn_logits is not None:
            attn_logits += prev_attn_logits
        if attn_mask is not None:
            attn_logits.masked_fill_(attn_mask.unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            attn_logits = attn_logits.view(
                batch_size, self.num_heads, tgt_len, src_len
            )  # (N, H, L, S)
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_logits = attn_logits.view(
                batch_size * self.num_heads, tgt_len, src_len
            )  # (N * H, L, S)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)  # (N * H, L, S)

        attn_outputs = torch.bmm(attn_weights, value)  # (N * H, L, E')
        attn_outputs = (
            attn_outputs.transpose(0, 1)
            .contiguous()
            .view(tgt_len, batch_size, self.embed_dim)
        )  # (L, N, E)
        attn_outputs = self.out_proj(attn_outputs)  # (L, N, E)
        return attn_outputs, attn_logits


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = ResidualMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask, src_key_padding_mask, prev_attn_logits=None):
        """
        Arguments:
            src: (L, N, E)
            src_mask: (L, L)
            src_key_padding_mask: (N, L)
            prev_attn_logits: (N * H, L, L)
        """
        src2, attn_logits = self.self_attn(
            src,
            src,
            src,
            prev_attn_logits,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_logits


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = ResidualMultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = ResidualMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_prev_attn_logits=None,
        memory_prev_attn_logits=None,
    ):
        """
        Arguments:
            tgt: (L, N, E)
            memory: (S, N, E)
            tgt_mask: (L, L)
            memory_mask: (L, S)
            tgt_key_padding_mask: (N, L)
            memory_key_padding_mask: (N, S)
            tgt_prev_attn_logits: (N * H, L, L)
            memory_prev_attn_logits: (N * H, L, S)
        """
        tgt2, tgt_attn_logits = self.self_attn(
            tgt,
            tgt,
            tgt,
            tgt_prev_attn_logits,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, memory_attn_logits = self.multihead_attn(
            tgt,
            memory,
            memory,
            memory_prev_attn_logits,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt_attn_logits, memory_attn_logits


class InformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Arguments:
            src: (L, N, E)
            mask: (L, L)
            src_key_padding_mask: (N, L)
        """
        output = src
        prev_attn_logits = None
        for mod in self.layers:
            output, prev_attn_logits = mod(
                output, mask, src_key_padding_mask, prev_attn_logits
            )
        return output


class InformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Arguments:
            tgt: (L, N, E)
            memory: (S, N, E)
            tgt_mask: (L, L)
            memory_mask: (L, S)
            tgt_key_padding_mask: (N, L)
            memory_key_padding_mask: (N, S)
        """
        output = tgt
        tgt_prev_attn_logits = None
        memory_prev_attn_logits = None
        for mod in self.layers:
            output, tgt_prev_attn_logits, memory_prev_attn_logits = mod(
                output,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                tgt_prev_attn_logits,
                memory_prev_attn_logits,
            )
        return output


class Informer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output
