'''
- Now we'll actually implement the code. Make sure each of these is completely correct - it's very easy to get the small details wrong.
    - Implement the positional embedding function first. 
    - Then implement the function which calculates attention, given (Q,K,V) as arguments. 
    - Now implement the masking function. 
    - Put it all together to form an entire attention block. 
    - Finish the whole architecture.
'''
import torch as t
from torch import nn
import mypy

class myEmbedding(nn.Module):
    def __init__(self, n_embeds, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(t.randn((n_embeds, embed_dim)))

    def forward(self, x):
        # x is a tensor. x has integers. x has shape (seq,)
        return self.weights[x]

class myTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embeds = 1024
        self.embed_dim = 768
        self.vocab_size = 50257

        self.token_embed = t.rand((50257, 768))  # vocab_size x embed_dim
        pass

    def forward(self, input):
        '''
        forward:
            input: t.Tensor[int]
                input.shape = (batch, seq)
                0 <= t.Tensor[i] < vocab_size
        '''
        b, s = input.shape

        tok_embed = self.token_embedding(input, self.vocab_size, self.embed_dim)
        pos_input = t.arange(s)
        pos_embed = self.positional_embedding(pos_input, self.n_embeds, self.embed_dim)

    def positional_embedding(self, input: t.Tensor, n_embeds: int=1024, embed_dim: int=768):
        return myEmbedding(n_embeds, embed_dim)

    def token_embedding(self, input: t.Tensor, n_embeds: int=1024, embed_dim: int=768):
        return myEmbedding(n_embeds, embed_dim)