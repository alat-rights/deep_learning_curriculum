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

class myEmbedding(nn.Module):
    def __init__(self, n_embeds, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(t.randn((n_embeds, embed_dim)))

    def forward(self, x):
        # x is a tensor. x has integers. x has shape (seq,)
        return self.weights[x]

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
        self.attention = t.nn.MultiheadAttention(self.embed_dim, self.num_heads)
        
        '''MLP Block'''
        self.lin1 = t.nn.Linear(in_features=self.embed_dim, out_features=4*self.embed_dim)
        self.lin2 = t.nn.Linear(in_features=4*self.embed_dim, out_features=self.embed_dim)
        self.gelu = t.nn.GELU()
        self.dropout = t.nn.Dropout(p=self.dropoutp)
    
    def forward(self, x):
        normed_x = self.ln1(x)
        attend_x = self.attention(normed_x, normed_x, normed_x)[0]
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
        return myEmbedding(n_embeds, embed_dim) # input unused

    def token_embedding(self, n_embeds: int=1024, embed_dim: int=768):
        return myEmbedding(n_embeds, embed_dim)

if __name__ == '__main__':
    trans = myTransformer()
    # TODO: Embed words
    tens = t.tensor([[5, 150]])
    print(trans(tens))