import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
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
        self.tok_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(max_len, embed_dim)
        self.layers = [
            EncoderLayer(embed_dim, num_heads, ff_hid_dim, dropout)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src, src_mask):
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        device = next(self.parameters()).device
        pos = (
            tf.range(0, seq_len, dtype=tf.int32)[None, :]
            .repeat(batch_size, 1)
            .to(device)
        )
        src = self.dropout(self.pos_embed(pos) + self.tok_embed(src))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, embed_dim, num_heads, ff_hid_dim, dropout
    ):
        super().__init__()
        self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForwardTF(embed_dim, ff_hid_dim, dropout)
        self.attention = MultiHeadAttentionTF(embed_dim, num_heads)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src, src_mask):
        attention_out = self.attention(src, src, src, src_mask)
        attetion_ln_out = self.dropout(self.attention_ln(src + attention_out))
        ff_out = self.ff(attetion_ln_out)
        ff_ln_out = self.dropout(self.ff_ln(attetion_ln_out + ff_out))
        return ff_ln_out

class MultiHeadAttentionTF(tf.keras.layers.Layer):
    def __init__(self, hid_dim, num_heads):
        super().__init__()
        
        assert hid_dim % num_heads == 0, "`hidden_dim` must be a multiple of `num_heads`"
        
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        
        self.fc_v = tf.keras.layers.Dense(hid_dim, use_bias=False)
        self.fc_k = tf.keras.layers.Dense(hid_dim, use_bias=False)
        self.fc_q = tf.keras.layers.Dense(hid_dim, use_bias=False)
        self.fc = tf.keras.layers.Dense(hid_dim, use_bias=False)
    
    def call(self, value, key, query, mask=None):
        # keys.shape = [batch_size, seq_len, embed_dim]
        batch_size = query.shape[0]
        
        V = self.fc_v(value)
        K = self.fc_k(key)
        Q = self.fc_q(query)
        # shape = [batch_size, seq_len, hid_dim]
        
        V = tf.reshape(V, [batch_size, -1, self.num_heads, self.head_dim])
        K = tf.reshape(K, [batch_size, -1, self.num_heads, self.head_dim])
        Q = tf.reshape(Q, [batch_size, -1, self.num_heads, self.head_dim])
        # V.shape = [batch_size, num_heads, value_len, head_dim]
        # K_t.shape = [batch_size, num_heads, head_dim, key_len]
        
        energy = tf.matmul(Q, K, transpose_b=True) / (self.hid_dim ** 1/2)
        # energy.shape = [batch_size, num_heads, query_len, key_len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        
        attention = tf.nn.softmax(energy, axis=-1)
        weighted = tf.matmul(attention, V)
        # weighted.shape = [batch_size, num_heads, seq_len, head_dim]
        weighted = tf.transpose(weighted, perm=[0, 2, 1, 3])
        # weighted.shape = [batch_size, seq_len, num_heads, head_dim]
        weighted = tf.reshape(weighted, [batch_size, -1, self.hid_dim])
        # weighted.shape = [batch_size, seq_len, hid_dim]
        out = self.fc(weighted)
        # out.shape = [batch_size, seq_len, hid_dim]
        return out

class FeedForwardTF(tf.keras.layers.Layer):
    def __init__(self, embed_dim, hid_dim, dropout):
        super(FeedForwardTF, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hid_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, x):
        return self.dense2(self.dropout(self.dense1(x)))

class Decoder(tf.keras.layers.Layer):
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
        self.tok_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(max_len, embed_dim)
        self.layers = [
            DecoderLayer(embed_dim, num_heads, ff_hid_dim, dropout)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, trg, trg_mask, enc_src, src_mask):
        batch_size = trg.shape[0]
        seq_len = trg.shape[1]
        device = next(self.parameters()).device
        pos = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
        trg = self.dropout(self.pos_embed(pos) + self.tok_embed(trg))
        for layer in self.layers:
            trg = layer(trg, trg_mask, enc_src, src_mask)
        out = self.fc(trg)
        return out

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_hid_dim, dropout):
        super().__init__()
        self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dec_attn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.enc_attn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForward(embed_dim, ff_hid_dim, dropout)
        self.dec_attn = MultiHeadAttention(embed_dim, num_heads)
        self.enc_attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, trg, trg_mask, enc_src, src_mask):
        dec_attn_out = self.dropout(self.dec_attn(trg, trg, trg, trg_mask))
        dec_attn_ln_out = self.dec_attn_ln(trg + dec_attn_out)
        enc_attn_out = self.dropout(
            self.enc_attn(enc_src, enc_src, dec_attn_ln_out, src_mask)
        )
        enc_attn_ln_out = self.enc_attn_ln(dec_attn_ln_out + enc_attn_out)
        ff_out = self.dropout(self.ff(enc_attn_ln_out))
        ff_ln_out = self.ff_ln(ff_out + enc_attn_ln_out)
        return ff_ln_out

class Transformer(tf.keras.layers.Layer):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size,
        src_pad_idx, 
        trg_pad_idx,
        embed_dim=512,
        max_len=100,
        num_layers=6,
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
        src_mask = (src != self.src_pad_idx).expand_dims(1).expand_dims(2)
        # src.shape = [batch_size, 1, 1, src_len]
        return src_mask

    def make_trg_mask(self,trg):
       batch_size = trg.size[0]
       seq_len = TRG.size[1]
       pad_mask = (trg != self.trg_pad_idx).expand_dims(1).expand_dims(2)
       seq_mask = tf.linalg.band_part(tf.ones(seq_len,seq_len))
       trg_mask = pad_mask * seq_mask
       return trg_mask

    def call(self,src,trg):
       src_mask = self.make_src_mask(src)
       trg_mask = self.make_trg_mask(trg)
       enc_src = self.encoder(src, src_mask)
       decoder_out = self.decoder(trg, trg_mask, enc_src, src_mask)
       return decoder_out

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

src = tf.constant(
    [[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]
)
trg = tf.constant(
    [[1, 7, 4, 3, 5, 0, 0, 0], [1, 5, 6, 2, 4, 7, 6, 2]]
)

model = Transformer(
    src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx
)

