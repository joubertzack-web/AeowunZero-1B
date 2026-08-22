import os
import sys
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Optional

# Ensure project root is in path relative to this file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Change working directory to project root for consistent file paths
os.chdir(project_root)

from aeowun_zero.model import AEOWUNModel
from aeowun_zero.config import AEOWUNConfig
from aeowun_zero.optimizer import Muon
from aeowun_zero.tokenizer import AEOWUNTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AEOWUNTrainer:
    def __init__(self, config: AEOWUNConfig, tokenizer: AEOWUNTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model = AEOWUNModel(config).to(self.device)
        # Muon handles p.ndim == 2 internally. 
        # For non-2D params it falls back to SGD with momentum.
        self.optimizer = Muon(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        self.step = 0
        self.tokens_processed = 0
        
    def load_corpus(self, path):
        bin_path = path.replace('.txt', '.bin')
        if os.path.exists(bin_path):
            print(f"Loading pre-tokenized corpus from {bin_path}...")
            tokens = torch.load(bin_path, weights_only=True)
            return tokens.long()
        
        print(f"Tokenizing raw corpus from {path} (this happens once)...")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        tokens_list = self.tokenizer.encode(content)
        tokens = torch.tensor(tokens_list, dtype=torch.uint16) # uint16 is enough for 8k vocab
        
        print(f"Saving tokenized corpus to {bin_path}...")
        torch.save(tokens, bin_path)
        return tokens.long() # Convert to long for the model

    def get_batch(self, data):
        ix = torch.randint(len(data) - self.config.max_seq_len, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.max_seq_len] for i in ix])
        y = torch.stack([data[i+1:i+self.config.max_seq_len+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self, train_data, val_data, eval_iters=10):
        out = {}
        self.model.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(data)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self, path):
        torch.serialization.add_safe_globals([AEOWUNConfig])
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
            'tokens_processed': self.tokens_processed
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        torch.serialization.add_safe_globals([AEOWUNConfig])
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.tokens_processed = checkpoint.get('tokens_processed', 0)
        print(f"Checkpoint loaded from {path} at step {self.step}")

    def train(self, max_steps, train_data, val_data, log_interval=10, eval_interval=100, ckpt_interval=500):
        print(f"Starting training on {self.device}...")
        self.model.train()
        
        pbar = tqdm(range(max_steps), desc="Training AEOWUN Model Zero-A")
        
        for i in pbar:
            self.step += 1
            
            if self.step % eval_interval == 0:
                losses = self.estimate_loss(train_data, val_data)
                pbar.write(f"Step {self.step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            X, Y = self.get_batch(train_data)
            
            t0 = time.time()
            logits, loss = self.model(X, Y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            t1 = time.time()
            
            dt = t1 - t0
            num_tokens = self.config.batch_size * self.config.max_seq_len
            self.tokens_processed += num_tokens
            
            if self.step % log_interval == 0:
                tokens_per_sec = num_tokens / dt
                vram = 0
                if torch.cuda.is_available():
                    vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "VRAM": f"{vram:.0f}MB"
                })

            if self.step % ckpt_interval == 0:
                self.save_checkpoint(f"checkpoints/ckpt_step_{self.step}.pt")

        self.save_checkpoint("checkpoints/final_model.pt")
        print(f"Training finished. Total tokens: {self.tokens_processed}")

if __name__ == "__main__":
    set_seed(42)
    config = AEOWUNConfig()
    tokenizer = AEOWUNTokenizer.load("data/aeowun_tokenizer.json")
    
    trainer = AEOWUNTrainer(config, tokenizer)
    
    # Verify parameter count
    stats = trainer.model.get_parameter_count()
    print(f"Model Parameters: {stats['total']:,}")
    assert stats['total'] == 17_308_032, f"Model size mismatch: {stats['total']}"
    
    data = trainer.load_corpus("data/corpus.txt")
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # 200M Token Experiment
    # 200,000,000 / (batch_size * seq_len) = 200,000,000 / 2048 ~= 97,656 steps
    MAX_STEPS = 98000 
    
    print(f"\n--- Starting AEOWUN Model Zero-A Full Training ({MAX_STEPS} steps) ---")
    trainer.train(
        max_steps=MAX_STEPS, 
        train_data=train_data, 
        val_data=val_data, 
        log_interval=100, 
        eval_interval=500, 
        ckpt_interval=2000
    )
