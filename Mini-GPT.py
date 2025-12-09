# mini_gpt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

# -------------------------
#  CONFIG
# -------------------------
class Config:
    vocab_size = 256           # byte-level tokenizer (0-255)
    block_size = 128           # context length
    n_layers = 6
    n_heads = 8
    n_embd = 512
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    batch_size = 32
    max_iters = 2000
    eval_interval = 200
    seed = 42

C = Config()

torch.manual_seed(C.seed)
random.seed(C.seed)

# -------------------------
#  SIMPLE BYTE-LEVEL TOKENIZER
# -------------------------
# We will treat text as raw bytes (utf-8), mapping each byte to 0..255
def encode(s: str) -> torch.LongTensor:
    b = s.encode('utf-8')
    return torch.tensor(list(b), dtype=torch.long)

def decode(t: torch.LongTensor) -> str:
    return bytes([int(x) for x in t.cpu().numpy()]).decode('utf-8', errors='replace')

# -------------------------
#  MINI GPT: Attention, MLP, Block, Model
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # combined projection for qkv
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, T, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, T, head_dim)

        # compute attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)

        # causal mask: prevent attending to future
        i = torch.arange(T, device=x.device)
        causal_mask = i.view(1, 1, T) >= i.view(1, T, 1)  # (1, T, T) bool
        att = att.masked_fill(~causal_mask, float('-inf'))

        if attn_mask is not None:
            att = att + attn_mask  # allow passing additional masks if needed

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size
        tok_emb = self.token_emb(idx)            # (B, T, n_embd)
        pos_emb = self.pos_emb[:, :T, :]        # (1, T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                   # (B, T, vocab)
        if targets is None:
            return logits
        # compute loss: shift logits and targets for autoregressive next-token prediction
        B, T, V = logits.size()
        loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        idx: (B, T) starting context
        returns: (B, T+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            B, T = idx.size()
            if T > self.block_size:
                idx_cond = idx[:, -self.block_size:]
            else:
                idx_cond = idx
            logits = self(idx_cond)[:, -1, :]  # (B, vocab)
            logits = logits / temperature

            # filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)

            if top_p is not None and top_p < 1.0:
                # nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                # mask tokens with cumulative prob above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(1, indices_to_remove.view(B, -1), -1e10)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# -------------------------
#  DATA LOADING (toy example)
# -------------------------
# For real training use large text corpora. Here we demonstrate with a small dataset.
def load_data_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    data_tensor = encode(data)
    return data_tensor

def get_batch(data_tensor, batch_size, block_size, device):
    # sample random substrings
    n = data_tensor.size(0) - block_size
    starts = torch.randint(0, max(1, n), (batch_size,))
    x = torch.stack([data_tensor[s:s+block_size] for s in starts]).to(device)
    y = torch.stack([data_tensor[s+1:s+block_size+1] for s in starts]).to(device)
    return x, y

# -------------------------
#  TRAIN LOOP
# -------------------------
def train(data_path):
    data = load_data_from_file(data_path)
    model = MiniGPT(C).to(C.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr)

    for it in range(C.max_iters):
        model.train()
        xb, yb = get_batch(data, C.batch_size, C.block_size, C.device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % C.eval_interval == 0 or it == C.max_iters - 1:
            print(f"iter {it} loss {loss.item():.4f}")
            # generate a small sample
            model.eval()
            start = xb[0:1, :10]  # take first 10 tokens of a batch as prompt
            out = model.generate(start, max_new_tokens=100, temperature=1.0, top_k=50)
            print("=== sample ===")
            print(decode(out[0].tolist()))
            print("==============")

    # save model
    torch.save(model.state_dict(), "mini_gpt.pth")
    print("model saved to mini_gpt.pth")

# -------------------------
#  USAGE / QUICK DEMO
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.txt', help='path to text file (utf-8)')
    parser.add_argument('--train', action='store_true', help='start training')
    parser.add_argument('--sample', action='store_true', help='sample using saved model')
    parser.add_argument('--prompt', type=str, default='Hello', help='prompt for sampling')
    args = parser.parse_args()

    if args.train:
        print("Training on", args.data)
        train(args.data)
    elif args.sample:
        model = MiniGPT(C).to(C.device)
        model.load_state_dict(torch.load("mini_gpt.pth", map_location=C.device))
        model.eval()
        context = encode(args.prompt).unsqueeze(0).to(C.device)
        out = model.generate(context, max_new_tokens=200, temperature=1.0, top_k=50)
        print(decode(out[0].tolist()))
    else:
        print("no action. use --train or --sample")
