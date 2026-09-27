from dataclasses import dataclass

import torch 

import math

import os

import torch.nn as nn

from torch.nn import functional as F

@dataclass
class Config:
    vocab_size : int   #distinct tokens in the vocabulary
    block_size : int = 64  #context length -> how many past tokes 
    n_layers : int = 4  #number of transformer blocks. (consists of MLP + self attention + layer norm + residual conections)
    n_heads : int = 4  #number of attention heads in each transformer block
    n_embed : int = 128  #embedding dimension
    dropout : float = 0.0  #dropout rate



class Attention(nn.Module): # Multi-Head Self Attention (MHSA) layer (causal self Attention layer)
    
    def __init__(self, cfg : Config):
        super().__init__()
        self.n_head = cfg.n_heads
        self.n_embed = cfg.n_embed
        assert cfg.n_embed % cfg.n_heads == 0 ## must be divisible by number of heads


        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed)  #query, key, value linear layers

        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)  #output projection layer
        
        self.attn_drop = nn.Dropout(cfg.dropout)  #dropout layer for attention weights

        self.resid_drop = nn.Dropout(cfg.dropout)  #dropout layer for output of attention layer

        mask = torch.tril(torch.ones(cfg.block_size,cfg.block_size)) ## lower triangular matrix to mask future tokens in the attention mechanism
        self.register_buffer("mask",mask.view(1,1,cfg.block_size,cfg.block_size))  ## register the mask as a buffer so that it is not considered a parameter of the model




    def forward(self,x):

        B , T , C = x.shape #B = batch size, T = sequence length, C = embedding dimension

        hs = C // self.n_head  #dimension of each head



        q , k, v = self.qkv(x).split(self.n_embed, dim=2)  #split the output of the linear layer into query, key and value tensors

        q = q.view(B,T,self.n_head,hs).transpose(1,2) #reshape and transpose to get (B, n_head, T, hs)
        k = k.view(B,T,self.n_head,hs).transpose(1,2) #reshape and transpose to get (B, n_head, T, hs)
        v = v.view(B,T,self.n_head,hs).transpose(1,2) #reshape and transpose to get (B, n_head, T, hs)


        attn = (q @ k.transpose(-2,-1) * (1.0 / math.sqrt(hs))) #compute attention scores (B, n_head, T, T)


        attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0 , float('-inf')) #mask future tokens in the attention scores
        print("BEFORE SOFTMAX")
        print("shape:", attn.shape)
        print("dtype:", attn.dtype)

        print("has NaN:", torch.isnan(attn).any().item())
        print("has +inf:", torch.isposinf(attn).any().item())
        print("has -inf:", torch.isneginf(attn).any().item())

        finite = torch.isfinite(attn)

        print("number finite:", finite.sum().item())
        print("total values:", attn.numel())

        if finite.any():
            print("finite min:", attn[finite].min().item())
            print("finite max:", attn[finite].max().item())
        # Check whether any row has at least one finite value
        row_has_finite = torch.isfinite(attn).any(dim=-1)

        print("rows with no finite values:",
            (~row_has_finite).sum().item())

        attn = F.softmax(attn,dim=1) #apply softmax to get attention weights


        test = F.softmax(attn, dim=-1)

        print("test softmax NaN:", torch.isnan(test).any().item())
        print("test softmax finite:", torch.isfinite(test).all().item())
        print("test row 0:", test[0, 0, 0])
        print("test row 0 sum:", test[0, 0, 0].sum())



        print("softmax:", torch.isfinite(attn).all().item())
        print("softmax NaN:", torch.isnan(attn).any().item())
        attn = self.attn_drop(attn) #apply dropout to attention weights

        

        y = attn @ v #compute the output of the attention layer (B, n_head, T, hs)


        y = y.transpose(1,2).contiguous().view(B,T,C) #reshape and transpose to get (B, T, C)

        y = self.resid_drop(self.proj(y)) #apply dropout to the output

        return y

class MLP(nn.Module): # Multi-Layer Perceptron ( fully connected feed forward neural network)
    
    def __init__(self,cfg : Config):
        super().__init__()

        self.fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)

        self.proj =  nn.Linear(4 * cfg.n_embed, cfg.n_embed)

        self.drop = nn.Dropout(cfg.dropout)

    
    def forward(self,x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x) 

        return x 
    


class Block(nn.Module):   ## 1 transformer block = 1 layer of self attention + MLP + layer norm + residual connections + dropout 

    def __init__(self,cfg : Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attention = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg)


    def forward(self,x):
        x = x + self.attention(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module): ## GPT model = embedding layer + n transformer blocks + layer norm + linear layer to output logits for each token in the vocabulary
    
    def __init__(self,cfg: Config):
        super().__init__()

        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size,cfg.n_embed)  #token embedding layer
        self.pos_emb = nn.Embedding(cfg.block_size,cfg.n_embed)  #positional embedding layer

        self.drop = nn.Dropout(cfg.dropout)  #dropout layer , added after the embedding layer to prevent overfitting


        self.blocks = nn.ModuleList (
            [Block(cfg) for _ in range(cfg.n_layers)]
        )  # list of transformer blocks


        self.ln_f = nn.LayerNorm(cfg.n_embed)  #layer norm after the transformer blocks


        self.final_head = nn.Linear (cfg.n_embed,cfg.vocab_size, bias=False)  #linear layer to output logits for each token in the vocabulary


        ## weight tying = The model learns one shared representation space for tokens rather than independently learning an input embedding and an output projection.

        
        self.final_head.weight = self.token_emb.weight  #tie the weights of the final linear layer to the token embedding layer


        self.apply(self._init_weights)  #initialize the weights of the model (for all the layers)

        n_params = sum(p.numel() for p in self.parameters())  #count the number of parameters in the model
        print(f"number of parameters in the model : {n_params/1e6:.2f} M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # initialize random weights of linear layers with normal distribution and biases with zeros
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # initialize random weights of embedding layers with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, idx, targets=None):  #idx = input token indices, targets = target token indices for computing loss

        B , T = idx.shape #B = batch size, T = sequence length

        pos = torch.arange(T, device=idx.device)
        pos = self.pos_emb(pos)  #get positional embeddings for input indices (T, C)

        tok = self.token_emb(idx)  #get token embeddings for input indices (B, T, C)

        x = self.drop(tok + pos)  #add token and positional embeddings and apply dropout


        for block in self.blocks:  #pass through each transformer block
            x = block(x)
   


        
        x = self.ln_f(x)  #apply layer norm after the transformer blocks
       

        logits = self.final_head(x)  #get logits for each token in the vocabulary (B, T, vocab_size)
        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss
        

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20,temperature=1.0,top_k=None):

        self.eval() ## reomve dropout layers during inference 

        for _ in range(max_new_tokens):

            idx_cond = idx[:, - self.cfg.block_size:]  #crop the context to the last block_size tokens
            logits, _ = self.forward(idx_cond)   #get logits for the last token in the context (B, T, vocab_size)

            logits = logits[:, -1, :] / temperature  #scale the logits by temperature (B, vocab_size)

            if top_k is not None:  #if top_k is specified, keep only the top_k logits and set the rest to -inf
                v, _ = torch.topk(logits, top_k)  #get the top_k logits
                logits[logits < v[:, [-1]]] = -float('Inf')  #set the rest to -inf 

                probs = F.softmax(logits, dim=-1)  #apply softmax to get probabilities (B, vocab_size)

                next_id = torch.multinomial(probs, num_samples=1)  #sample the next token from the probabilities (B, 1) performs sampling

                idx = torch.cat((idx, next_id), dim=1)  #append the next token to the context (B, T+1)
        self.train() ## set the model back to training mode after inference

        return idx 

## Data loading and preprocessing

BUILTIN_TEXT = (
    "A transformer reads a sequence of tokens and predicts the next one. "
    "Attention lets every token look back at the tokens that came before it, "
    "and decide which of them matter for what comes next. The model has no "
    "idea what a word is at first; it only sees numbers. Through many small "
    "updates, the weights shift until the guesses stop being random and start "
    "being language. Each block mixes information across tokens with attention, "
    "then refines each token on its own with a small neural network. Stack "
    "enough of these blocks, feed them enough text, and patterns emerge: "
    "spelling, then words, then grammar, then meaning. This is the whole idea, "
    "written small enough to fit in one file and read in one sitting.\n"
) * 400  # repeated so we have enough data to learn visible structure
 
 
def load_text():
    if os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            return f.read()
    return BUILTIN_TEXT
                


# ---- hyperparameters for training (small so it runs on a laptop CPU) --------
batch_size = 32
max_iters = 3000
eval_interval = 300 # interval to evaluate the model on validation set
eval_iters = 50  # number of iterations to evaluate the model on validation set
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
 
text = load_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}   # char -> int
itos = {i: c for c, i in stoi.items()}       # int  -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
 
cfg = Config(vocab_size=vocab_size)
 
# encode the whole dataset, split 90/10 into train/val
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
 


def get_batch(split):
 # Grab batch_size random chunks. x is the chunk, y is x shifted by one.
    d = train_data if torch.split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix]) # get the input sequence of length block_size
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix]) # get the target sequence of length block_size (shifted by one)
    return x.to(device), y.to(device)



@torch.no_grad()
def estimate_loss(model):
    """Average loss over a few batches, for train and val, with dropout off."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
 



def main():
    print(f"device: {device} | vocab_size: {vocab_size} | chars: {len(data)}")
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
 
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
 
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
 
    # sample from the trained model, starting from a newline
    print("\n----- sample -----")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=None)
    print(decode(out[0].tolist()))
 
 
if __name__ == "__main__":
    main()