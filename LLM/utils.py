import math
import torch
import torch.nn as nn
from einops import rearrange

class Softmax(nn.Module):
    def __init__(self, dim = -1):
        self.dim = dim
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = torch.max(x, self.dim, keepdim=True)
        exp_x = torch.exp(x - x_max) # numerator
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True) # denominator
        output = exp_x / sum_exp_x
        return output

class CrossEntropy(nn.Module):
    def forward(self, x: torch.Tensor, target: torch.tensor):
        log_sum_exp = torch.logsumexp(x, dim=-1)
        predicted_logits = x[torch.arange(x.shape[0]), target]
        losses = log_sum_exp - predicted_logits
        avg_loss = torch.mean(losses)
        return avg_loss

def lr_cosine_scheduler(current_it, max_lr, min_lr, warmup_it, cosine_annealing_it):
    '''A scheduler is simply a function takes the current step t and other relevant parameters (such as the 
    initial and final learning rates), and returns the learning rate to use for the gradient update at step t.'''

    if current_it < warmup_it:
        return (current_it / warmup_it) * max_lr
    elif warmup_it <= current_it <= cosine_annealing_it:
        return min_lr + (1/2) * (1 + math.cos(((current_it - warmup_it) / (cosine_annealing_it - warmup_it)) * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr

def generate_text(model, tokenizer, start_text, max_length=100, temperature=0.0, device=torch.device('cpu')):
    model.eval()
    softmax = Softmax()
    tokens = tokenizer.encode(start_text)
    input_tensor = torch.tensor(tokens)

    input_tensor = rearrange(input_tensor, "i -> 1 i").to(device)
    generated_tokens = tokens

    for _ in range(max_length):
        input_context = input_tensor[:, -model.context_length:]

        with torch.no_grad():
            logits = model(input_context)
        
        last_logits = logits[0, -1, :]
        # last_logits = last_logits / temperature
        probabilities = softmax(last_logits)

        next_token_id = torch.multinomial(probabilities, num_samples=1).item()

        generated_tokens.append(next_token_id)
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]]).to(device)], dim = 1)

        if next_token_id == tokenizer.special_tokens.get('<|endoftext|>'):
            break
    
    generate_text = tokenizer.decode(generated_tokens)
    return generate_text