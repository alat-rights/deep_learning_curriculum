'''
- Now we'll actually implement the code. Make sure each of these is completely correct - it's very easy to get the small details wrong.
    - Implement the positional embedding function first.
    - Then implement the function which calculates attention, given (Q,K,V) as arguments.
    - Now implement the masking function.
    - Put it all together to form an entire attention block.
    - Finish the whole architecture.
'''
import torch as t
from torch import nn, einsum
import einops

class MyEmbedding(nn.Module):
    def __init__(self, n_embeds, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(t.randn((n_embeds, embed_dim)))

    def forward(self, x):
        # x is a tensor. x has integers. x has shape (seq,)
        return self.weights[x]

class MyMultihead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyMultihead, self).__init__()
        head_size = embed_dim // num_heads
        d_v = embed_dim // num_heads
        self.head_size = head_size

        # Initialize W_Q, W_K, W_V, W_O
        self.weight_q = nn.Parameter(t.randn((num_heads, embed_dim, head_size)))
        self.weight_k = nn.Parameter(t.randn((num_heads, embed_dim, head_size)))
        self.weight_v = nn.Parameter(t.randn((num_heads, embed_dim, d_v)))
        self.weight_o = nn.Parameter(t.randn((num_heads * d_v, embed_dim)))

        self.bias_q = nn.Parameter(t.randn((num_heads, 1, head_size)))
        self.bias_k = nn.Parameter(t.randn((num_heads, 1, head_size)))
        self.bias_v = nn.Parameter(t.randn((num_heads, 1, d_v)))
        self.bias_o = nn.Parameter(t.randn((1, embed_dim)))

    def forward(self, q, k, v):
        qwq = t.einsum('bse, neh -> bnsh', q, self.weight_q)
        kwk = t.einsum('bse, neh -> bnsh', k, self.weight_k)
        vwv = t.einsum('bse, nev -> bnsv', v, self.weight_v)

        heads = self.attention(qwq + self.bias_q, kwk + self.bias_k, vwv + self.bias_v)
        headscat = einops.rearrange(heads, 'b n q v -> b q (n v)')

        output = t.einsum('bqj, je -> bqe', headscat, self.weight_o) + self.bias_o
        return output

    def attention(self, qwq, kwk, vwv):
        sqrtdk = self.head_size ** 0.5
        qktmul = t.einsum('bnqh, bnkh -> bnqk', qwq, kwk)
        sm = t.softmax(input=(qktmul / sqrtdk), dim=2)
        '''
        Take qktmul[0, 0].
        That is a k-sized vector which tells us, at position 0, how much do we attend
        to each key?

        We softmax over the keys in dim 2 to prevent paying "negative attention" to a key.

        Maybe we don't want to attend to that many things? Softmax lets us reduce the smaller entries.

        What happens if we just don't softmax?
        '''
        prod = t.einsum('bnqk, bnkv -> bnqv', sm, vwv)
        return prod

class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embeds = 1024
        self.embed_dim = 768
        self.vocab_size = 50257
        self.num_heads = 12
        self.dropoutp = 0.5

        self.ln1 = t.nn.LayerNorm(self.embed_dim)
        self.ln2 = t.nn.LayerNorm(self.embed_dim)
        self.attention = MyMultihead(self.embed_dim, self.num_heads)

        '''MLP Block'''
        self.lin1 = t.nn.Linear(in_features=self.embed_dim, out_features=4*self.embed_dim)
        self.lin2 = t.nn.Linear(in_features=4*self.embed_dim, out_features=self.embed_dim)
        self.gelu = t.nn.GELU()
        self.dropout = t.nn.Dropout(p=self.dropoutp)

    def forward(self, x):
        normed_x = self.ln1(x)
        attend_x = self.attention(normed_x, normed_x, normed_x)
        attend_x += x  # Skip

        # MLP
        normed_x = self.ln2(attend_x)
        normed_x = self.lin1(normed_x)
        normed_x = self.gelu(normed_x)
        normed_x = self.lin2(normed_x)
        normed_x = self.dropout(normed_x)  # Drop!
        normed_x += attend_x  # Skip

        return normed_x

class myTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embeds = 1024
        self.embed_dim = 768
        self.vocab_size = 50257
        self.dropoutp = 0.5

        self.dropout = t.nn.Dropout(p=self.dropoutp)
        # GPT Blocks
        self.gpt_blocks = [GPTBlock() for i in range(12)]
        # The rest
        self.tok_embed = self.token_embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = self.positional_embedding(self.n_embeds, self.embed_dim)
        self.ln = t.nn.LayerNorm(self.embed_dim)


    def forward(self, input):
        '''
        forward:
            input: t.Tensor[int]
                input.shape = (batch, seq)
                0 <= t.Tensor[i] < vocab_size
        '''
        b, s = input.shape
        # Embeddings
        embedding = self.tok_embed(input)
        pos_input = t.arange(s)
        pos_embedding = self.pos_embed(pos_input)
        x = embedding + pos_embedding
        for block in self.gpt_blocks:
            x = block(x)

        # LayerNorm
        x = self.ln(x)
        # Unembed
        x = einsum('b s e, v e -> b s v', x, self.tok_embed.weights)

        return x

    def positional_embedding(self, n_embeds: int=1024, embed_dim: int=768):
        return MyEmbedding(n_embeds, embed_dim) # input unused

    def token_embedding(self, n_embeds: int=1024, embed_dim: int=768):
        return MyEmbedding(n_embeds, embed_dim)

if __name__ == '__main__':
    trans = myTransformer()
    # TODO: Embed words
    tens = t.tensor([[5, 150]])
    print(trans(tens))
