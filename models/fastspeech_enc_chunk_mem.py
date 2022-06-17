from typing import Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)


class FastSpeech(nn.Module):
    def __init__(self, n_mel_channels,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 n_speakers, speaker_emb_weight, mask_type, separate_mem_attn,
                 local_window_size):
        super(FastSpeech, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            mask_type=mask_type,
            separate_mem_attn=separate_mem_attn,
            local_window_size=local_window_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim,
            )
        # n_speakers = int(n_speakers)
        n_speakers = 1
        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            mask_type=mask_type,
            separate_mem_attn=separate_mem_attn,
            local_window_size=local_window_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
        self.spk_encoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            mask_type=mask_type,
            separate_mem_attn=separate_mem_attn,
            local_window_size=local_window_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
        self.mem_encoder = FFTransformer(
            n_layer=2, n_head=1,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            mask_type=mask_type,
            separate_mem_attn=separate_mem_attn,
            local_window_size=local_window_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
        self.mfcc_embed1 = nn.Linear(13, symbols_embedding_dim)
        self.relu = nn.ReLU()
        self.mfcc_embed2 = nn.Linear(symbols_embedding_dim, symbols_embedding_dim)
        self.spk_encoder1 = nn.Linear(symbols_embedding_dim, symbols_embedding_dim)
        self.spk_encoder2 = nn.Linear(symbols_embedding_dim, symbols_embedding_dim)
        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)
        self.relu = nn.ReLU()
        self.local_window_size = local_window_size


    def forward(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75,
                apply_mask=True, dec_cond=None):

        (inputs, mel_lens,
         speaker) = inputs
        mfcc = self.mfcc_embed2(self.relu(self.mfcc_embed1(inputs)))
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        enc_out, enc_mask = self.encoder(mfcc, mel_lens, apply_mask=apply_mask, use_memory_mask=False)
        dec_out, dec_mask = self.decoder(enc_out, mel_lens, apply_mask=apply_mask, use_memory_mask=False)
        if dec_cond is not None:
            dec_cond = dec_cond.to(dec_out.device)
            dec_out = torch.cat([dec_cond, dec_out], dim=1)
        mem_out, dec_mask = self.mem_encoder(dec_out, mel_lens, apply_mask=apply_mask, use_memory_mask=True)
        mel_out = self.proj(mem_out)
        return (mel_out, dec_out)

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).to(inp.dtype)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, mask_type, separate_mem_attn, local_window_size, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm
        self.mask_type = mask_type
        self.local_window_size = local_window_size
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.separate_mem_attn = separate_mem_attn
        if self.mask_type == 'tri':
            self.attn_mask = {}
            for shape in [local_window_size, 400]:
                attn_mask_ = (torch.triu(torch.ones((shape,shape))) == 1)
                attn_mask_[torch.eye(shape).byte()] = False
                self.attn_mask[shape] = attn_mask_
        elif self.mask_type == 'window':
            self.attn_mask = {}
            for shape in [local_window_size, 400]:
                attn_mask_ = torch.ones((shape, shape))
                attn_mask_[torch.eye(shape).byte( )] = False
                for i in range(len(attn_mask_)):
                    attn_mask_[i, max(0, i-self.local_window_size):i+self.local_window_size+1] = False
                self.attn_mask[shape] = attn_mask_
        elif self.mask_type == 'causal_window':
            self.attn_mask = {}
            for shape in [local_window_size, 400]:
                attn_mask_ = torch.ones((shape, shape))
                attn_mask_[torch.eye(shape).byte( )] = False
                for i in range(len(attn_mask_)):
                    attn_mask_[i, max(0, i-self.local_window_size):i] = False
                self.attn_mask[shape] = attn_mask_
        elif self.mask_type == 'square':
            self.attn_mask = {}
            for shape in [400, local_window_size]:
                if shape == local_window_size:
                    attn_mask_ = torch.zeros((shape, shape))
                else:
                    attn_mask_ = torch.ones((shape, shape)).byte()
                    for i in range(0, shape//local_window_size+1):
                        attn_mask_[i*local_window_size:(i+1)*local_window_size, i*local_window_size:(i+1)*local_window_size] = False
                self.attn_mask[shape] = attn_mask_
        elif self.mask_type == 'random_square':
            self.attn_mask = {}
            for shape in [400]:
                    loc_dict = {}
                    start_locs = torch.arange(0, local_window_size)
                    for start_loc in start_locs:
                        attn_mask_ = torch.ones((shape, shape)).byte()
                        attn_mask_[0:start_loc, 0:start_loc] = False
                        for i in range(0, shape//local_window_size+1):
                            attn_mask_[(i*local_window_size)+start_loc:((i+1)*local_window_size)+start_loc, (i*local_window_size)+start_loc:((i+1)*local_window_size)+start_loc] = False
                            loc_dict[start_loc.item()] = attn_mask_
                    self.attn_mask[shape] = loc_dict
            # import matplotlib.pyplot as plt
            # plt.imshow(loc_dict[1].cpu().numpy())
            # plt.savefig('50')
            # exit()

    def forward(self, inp, attn_mask=None, apply_mask=True, use_memory_mask=False):
        return self._forward(inp, attn_mask, apply_mask=apply_mask, use_memory_mask=use_memory_mask)

    def _forward(self, inp, attn_mask=None, apply_mask=True, use_memory_mask=False):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if self.mask_type == 'default':
                if apply_mask == True:
                    attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
                    attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)

            elif self.mask_type == 'tri':
                if apply_mask == True:
                    # attn_mask2 = attn_mask.unsqueeze(1).to(attn_score.dtype)
                    # attn_mask2 = attn_mask2.repeat(n_head, attn_mask2.size(2), 1)
                    if inp.shape[1] in self.attn_mask:
                        attn_mask = self.attn_mask[inp.shape[1]].to(attn_score.device)
                    else:
                        attn_mask = (torch.triu(torch.ones((inp.shape[1], inp.shape[1]), device=attn_score.device)) == 1)
                        attn_mask[torch.eye(inp.shape[1]).byte()] = False
                    # attn_mask = attn_mask.unsqueeze(0)
                    # attn_mask = attn_mask.repeat(inp.shape[0], 1, 1)
                    # attn_mask = ~(~attn_mask*~attn_mask2.bool())

            elif self.mask_type == 'window':
                if use_memory_mask:
                    attn_mask = torch.ones((inp.shape[1], inp.shape[1]), device=attn_score.device)
                    attn_mask[torch.eye(inp.shape[1]).byte( )] = False
                    for i in range(len(attn_mask)):
                        attn_mask[i, max(0, i-self.local_window_size*2):i+self.local_window_size*2+1] = False
                else:
                # attn_mask2 = attn_mask.unsqueeze(1).to(attn_score.dtype)
                # attn_mask2 = attn_mask2.repeat(n_head, attn_mask2.size(2), 1)
                    if apply_mask == True:
                        if inp.shape[1] in self.attn_mask:
                            attn_mask = self.attn_mask[inp.shape[1]].to(attn_score.device)
                        else:
                            attn_mask = torch.ones((inp.shape[1], inp.shape[1]), device=attn_score.device)
                            attn_mask[torch.eye(inp.shape[1]).byte( )] = False
                            for i in range(len(attn_mask)):
                                attn_mask[i, max(0, i-self.local_window_size):i+self.local_window_size+1] = False


                # attn_mask = attn_mask.unsqueeze(0)
                # attn_mask = attn_mask.repeat(inp.shape[0], 1, 1)
                # attn_mask = ~(~attn_mask.bool()*~attn_mask2.bool())


            elif self.mask_type == 'causal_window':
                if use_memory_mask:
                    attn_mask = torch.ones((inp.shape[1], inp.shape[1]), device=attn_score.device)
                    attn_mask[torch.eye(inp.shape[1]).byte( )] = False
                    for i in range(len(attn_mask)):
                        attn_mask[i, max(0, i-self.local_window_size*2):i] = False
                else:
                    if apply_mask == True:
                        # attn_mask2 = attn_mask.unsqueeze(1).to(attn_score.dtype)
                        # attn_mask2 = attn_mask2.repeat(n_head, attn_mask2.size(2), 1)
                        if inp.shape[1] in self.attn_mask:
                            attn_mask = self.attn_mask[inp.shape[1]].to(attn_score.device)
                        else:
                            attn_mask = torch.ones((inp.shape[1], inp.shape[1]), device=attn_score.device)
                            attn_mask[torch.eye(inp.shape[1]).byte( )] = False
                            for i in range(len(attn_mask)):
                                attn_mask[i, max(0, i-self.local_window_size):i] = False
                    # attn_mask = attn_mask.unsqueeze(0)
                    # attn_mask = attn_mask.repeat(inp.shape[0], 1, 1)
                    # attn_mask = ~(~attn_mask.bool()*~attn_mask2.bool())
            elif self.mask_type == 'square':
                if use_memory_mask:
                    attn_mask = torch.ones((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                    for i in range(0, inp.shape[1]//(self.local_window_size*2)+1):
                        attn_mask[i*self.local_window_size*2:(i+1)*self.local_window_size*2, i*self.local_window_size*2:(i+1)*self.local_window_size*2] = False
                else:
                    if apply_mask == True:
                        if inp.shape[1] in self.attn_mask:
                            attn_mask = self.attn_mask[inp.shape[1]].to(attn_score.device)
                        else:
                            attn_mask = torch.ones((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                            for i in range(0, inp.shape[1]//self.local_window_size+1):
                                attn_mask[i*self.local_window_size:(i+1)*self.local_window_size, i*self.local_window_size:(i+1)*self.local_window_size] = False

            elif self.mask_type == 'random_square':
                if use_memory_mask:
                    if inp.shape[1] == self.local_window_size*2:
                        attn_mask = torch.zeros((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                    else:
                        start_loc = random.choice([i for i in range(self.local_window_size*2)])
                        attn_mask = torch.ones((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                        attn_mask[0:start_loc, 0:start_loc] = False
                        for i in range(0, inp.shape[1]//self.local_window_size):
                            attn_mask[(i*self.local_window_size*2)+start_loc:((i+1)*self.local_window_size*2)+start_loc, (i*self.local_window_size*2)+start_loc:((i+1)*self.local_window_size*2)+start_loc] = False
                else:
                    if apply_mask == True:
                        if inp.shape[1] in self.attn_mask:
                            dict_of_all_masks = self.attn_mask[inp.shape[1]]
                            choose_current_key = random.choice(list(dict_of_all_masks.keys()))
                            attn_mask = dict_of_all_masks[choose_current_key].to(attn_score.device)
                        elif inp.shape[1] == self.local_window_size:
                            attn_mask = torch.zeros((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                        else:
                            start_loc = random.choice([i for i in range(self.local_window_size)])
                            attn_mask = torch.ones((inp.shape[1], inp.shape[1])).byte().to(attn_score.device)
                            attn_mask[0:start_loc, 0:start_loc] = False
                            for i in range(0, inp.shape[1]//self.local_window_size+1):
                                attn_mask[(i*self.local_window_size)+start_loc:((i+1)*self.local_window_size)+start_loc, (i*self.local_window_size)+start_loc:((i+1)*self.local_window_size)+start_loc] = False

            else:
                raise NotImplementedError

            if apply_mask==True:
                if self.mask_type == 'default':
                    attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))
                if self.mask_type != 'default' and apply_mask == True:
                    attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))
                # import matplotlib.pyplot as plt
                # plt.imshow(attn_mask.detach().cpu().numpy())
                # plt.savefig('u1.png')
                # plt.clf()
                # exit()
        if apply_mask==False:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            # import matplotlib.pyplot as plt
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            # plt.imshow(attn_mask[0].detach().cpu().numpy())
            # plt.savefig('u.png')
            # plt.clf()
            # exit()
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = output.to(attn_out.dtype)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, mask_type, separate_mem_attn, local_window_size,
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, mask_type, separate_mem_attn, local_window_size, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None, apply_mask=True, use_memory_mask=False):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2), apply_mask=apply_mask, use_memory_mask=use_memory_mask)
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size, mask_type, separate_mem_attn, local_window_size,
                     dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model,
                                         padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout, mask_type, separate_mem_attn, local_window_size,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None, conditioning=0, apply_mask=True, use_memory_mask=False):
        if self.word_emb is None:
            inp = dec_inp
            mask = mask_from_lens(seq_lens, max_len=inp.shape[1]).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb + conditioning)

        for layer in self.layers:
            out = layer(out, mask=mask, apply_mask=apply_mask, use_memory_mask=use_memory_mask)

        # out = self.drop(out)
        return out, mask

def mask_from_lens(lens, max_len=400):
    if max_len is None:
        max_len = lens.max()

    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask
