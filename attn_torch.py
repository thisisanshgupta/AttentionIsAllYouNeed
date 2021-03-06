import torch
import torch.nn as nn
import torch.nn.functional as F

#Encoder

class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_len,num_layers,num_heads,ff_hid_dim,dropout=0.5):
       super().__init__()
       self.tok_embed = nn.Embedding(vocab_size,embed_dim)
       self.pos_embed = nn.Embedding(max_len,embed_dim)
       self.layers = nn.ModuleList([
          EncoderLayer(embed_dim,num_heads,ff_hid_dim,dropout)
          for _ in range(num_layers)
       ])
       self.dropout = nn.Dropout(dropout)

    def forward(self,src,src_mask):
       batch_size = src.size(0)
       seq_len = src.size(1)
       device = next(self.parameters()).device
       pos = (
           torch.arange(0, seq_len)
           .unsqueeze(0)
           .repeat(batch_size, 1)
           .to(device)
       )
       src = self.dropout(self.pos_embed(pos) + self.tok_embed(src))
       for layer in self.layers:
           src = layer(src,src_mask)
       return src

#EncoderLayer

class EncoderLayer(nn.Module):
   def __init__(self,embed_dim,num_heads,ff_hid_dim,dropout):
      super().__init__()
      self.ff_ln = nn.LayerNorm(embed_dim)
      self.attention_ln = nn.LayerNorm(embed_dim)
      self.ff = FeedForward(embed_dim, ff_hid_dim, dropout)
      self.attention = MultiHeadAttention(embed_dim, num_heads)
      self.dropout = nn.Dropout(dropout)

   def forward(self, src, src_mask):
       attention_out = self.attention(src, src, src, src_mask)
       attetion_ln_out = self.dropout(self.attention_ln(src + attention_out))
       ff_out = self.ff(attetion_ln_out)
       ff_ln_out = self.dropout(self.ff_ln(attetion_ln_out + ff_out))
       return ff_ln_out

#MultiHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,hid_dim,num_heads):
        super().__init__()
        assert hid_dim % num_heads == 0, "`hidden_dim` must be a multiple of `num_heads`"
        
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias=False) #value
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias=False) #key
        self.fc_q = nn.Linear(hid_dim, hid_dim, bias=False) #query
        self.fc = nn.Linear(hid_dim, hid_dim)

    
    def forward(self, value, key, query, mask=None):
        # keys.shape = [batch_size, seq_len, embed_dim]
        batch_size = query.size(0)
        
        V = self.fc_v(value)
        K = self.fc_k(key)
        Q = self.fc_q(query)
        # shape = [batch_size, seq_len, hid_dim]
        
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_t = K.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # V.shape = [batch_size, num_heads, value_len, head_dim]
        # K_t.shape = [batch_size, num_heads, head_dim, key_len]
        
        energy = torch.matmul(Q, K_t) / (self.hid_dim ** 1/2)
        # energy.shape = [batch_size, num_heads, query_len, key_len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        
        attention = F.softmax(energy, dim=-1)
        weighted = torch.matmul(attention, V)
        # weighted.shape = [batch_size, num_heads, seq_len, head_dim]
        weighted = weighted.permute(0, 2, 1, 3)
        # weighted.shape = [batch_size, seq_len, num_heads, head_dim]
        weighted = weighted.reshape(batch_size, -1, self.hid_dim)
        # weighted.shape = [batch_size, seq_len, hid_dim]
        out = self.fc(weighted)
        # out.shape = [batch_size, seq_len, hid_dim]
        return out

#FeedForward Layer

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hid_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(F.relu(self.dropout(self.fc1(x))))

#Decoder

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_len,
        num_layers,
        num_heads,
        ff_hid_dim,
        dropout=0.5,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, ff_hid_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, trg, trg_mask, enc_src, src_mask):
        batch_size = trg.size(0)
        seq_len = trg.size(1)
        device = next(model.parameters()).device
        pos = (
            torch.arange(0, seq_len)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(device)
        )
        trg = self.dropout(self.pos_embed(pos) + self.tok_embed(trg))
        for layer in self.layers:
            trg = layer(trg, trg_mask, enc_src, src_mask)
        out = self.fc(trg)
        return out

#DecoderLayer

class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim, 
                 num_heads, 
                 ff_hid_dim, 
                 dropout
                ):

        super().__init__()
        self.ff_ln = nn.LayerNorm(embed_dim)
        self.dec_attn_ln = nn.LayerNorm(embed_dim)
        self.enc_attn_ln = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hid_dim, dropout)
        self.dec_attn = MultiHeadAttention(embed_dim, num_heads)
        self.enc_attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, trg_mask, enc_src, src_mask):
        dec_attn_out = self.dropout(self.dec_attn(trg, trg, trg, trg_mask))
        dec_attn_ln_out = self.dec_attn_ln(trg + dec_attn_out)
        enc_attn_out = self.dropout(
            self.enc_attn(enc_src, enc_src, dec_attn_ln_out, src_mask)
        )
        enc_attn_ln_out = self.enc_attn_ln(dec_attn_ln_out + enc_attn_out)
        ff_out = self.dropout(self.ff(enc_attn_ln_out))
        ff_ln_out = self.ff_ln(ff_out + enc_attn_ln_out)
        return ff_ln_out

#Transformer

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size,
        src_pad_idx, 
        trg_pad_idx,
        embed_dim=512,
        max_len=100,
        num_layers=12,
        num_heads=8,
        ff_hid_dim=2048,
        dropout=0.5,

    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_dim,
            max_len,
            num_layers,
            num_heads,
            ff_hid_dim,
            dropout,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_dim,
            max_len,
            num_layers,
            num_heads,
            ff_hid_dim,
            dropout,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    
    def make_src_mask(self, src):
        # src.shape = [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src.shape = [batch_size, 1, 1, src_len]
        return src_mask
    
    def make_trg_mask(self, trg):
        batch_size = trg.size(0)
        seq_len = trg.size(1)
        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        seq_mask = torch.tril(torch.ones(seq_len, seq_len))
        trg_mask = pad_mask * seq_mask
        return trg_mask
    
    def forward(self, src, trg):
        device = next(model.parameters()).device
        src_mask = self.make_src_mask(src).to(device)
        trg_mask = self.make_trg_mask(trg).to(device)
        enc_src = self.encoder(src, src_mask)
        decoder_out = self.decoder(trg, trg_mask, enc_src, src_mask)
        return decoder_out

#Main method

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src = torch.tensor(
    [[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]
).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 0, 0, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(
    device
)

model = Transformer(
    src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx
).to(device)

model

