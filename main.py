import os
import torch
from torch import nn, optim, utils
from tqdm import tqdm
from einops import rearrange
import math
from Tokenizer import BPE
from LLM import LM, Linear, Embedding, ScaledDotProductAttention, MultiHeadSelfAttenion, utils
from data import LMDataset

# training or inference
TRAINING_TOKENIZER = False
TRAINING_MODEL = True
INFERENCE_MODEL = False

TOKENIZER_SAVE = "BPE_tokenizer_save"
TOKENIZER_TRAIN_DATA = "Dataset/TinyStoriesV2-GPT4-valid.txt"
PRETOKENIZE_DATA =  "Dataset/TinyStoriesV2-GPT4-train.txt"
PRETOKENIZE_DATA_LOC_TRAIN = "Dataset/pretokenizerTinyStoriesV2-GPT4-train.npy"
PRETOKENIZE_DATA_LOC_VALID = "Dataset/pretokenizerTinyStoriesV2-GPT4-valid.npy"
FINAL_MODEL_SAVE = "final_model_train.pt"

# Dataset hyperparameters
batch_size = 256

# Define model hyperparameters
vocab_size = 5000
context_length = 512
d_model = 512
num_layers = 4
num_heads = 8
d_ff = 2048
save_factor = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Define training hyperparameter
num_epochs = 4
warmup_steps = 1000
cosine_annealing_steps = 3000
max_lr = 5e-4
min_lr = 1e-5
global_step = 0

if TRAINING_TOKENIZER:
    tokenizer = BPE.BPETokenizer(vocab_size=vocab_size, merges={}, special_tokens={"<UNK>": 259, "<|endoftext|>" : 258}, file_path=TOKENIZER_TRAIN_DATA)
    tokenizer.train_tokenizer()
    tokenizer.save(TOKENIZER_SAVE)
    tokenizer.pretokenize_file(file_path=PRETOKENIZE_DATA, out_path=PRETOKENIZE_DATA_LOC)
else:
    print("Loading the tokenizer...")
    tokenizer = BPE.BPETokenizer.load(TOKENIZER_SAVE)


print(f"Actual vocab size after training: {len(tokenizer.vocab)}")
print(f"Max token ID in vocab: {max(tokenizer.vocab.values())}")
print(f"Special tokens: {tokenizer.special_tokens}")



train_dataset = LMDataset.LMDataset(PRETOKENIZE_DATA_LOC_TRAIN, block_size=context_length) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = LMDataset.LMDataset(PRETOKENIZE_DATA_LOC_VALID, block_size=context_length)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle validation data

model = LM.LanguageModel(
    vocab_size=len(tokenizer.vocab) , 
    context_length=context_length, 
    d_model=d_model, 
    num_layers=num_layers, 
    num_heads=num_heads, 
    d_ff=d_ff, 
    device=device
).to(device)

cross_entropy = utils.CrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr = 1e-3)


if TRAINING_MODEL:
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1}/ {num_epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = model(x) # (batch, seq_len, vocab_size)

            # reshaping for loss calculation    
            logits_flat = rearrange(logits, "b s v -> (b s) v")
            y_flat = rearrange(y, "b s -> (b s)")

            # cross entropy loss
            loss = cross_entropy(logits_flat, y_flat)
            epoch_loss += loss.item()
            # print(loss)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating LR
            global_step += 1
            lr = utils.lr_cosine_scheduler(global_step, max_lr, min_lr, warmup_steps, cosine_annealing_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        avg_train_loss = epoch_loss / len(train_loader)
        # print(f"Epoch {epoch + 1} finished. Avg training loss {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc = f"Epoch {epoch + 1} / {num_epochs} [Validation]")
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                logits_flat = rearrange(logits, "b s v -> (b s) v")
                y_flat = rearrange(y, "b s -> (b s)")

                loss = cross_entropy(logits_flat, y_flat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_val_loss))

        print(f"Epoch {epoch + 1} finished.")
        print(f"Average Training loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Perplexity: {perplexity:.4f}")

        # Save model after each epoch
        if epoch % save_factor == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, save_path)
            print(f"Model saved to {save_path}")

    # Save the final model after the last epoch
    final_save_path = FINAL_MODEL_SAVE
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

if INFERENCE_MODEL:
    final_load_path = FINAL_MODEL_SAVE
    model.load_state_dict(torch.load(final_load_path))
    model.eval()
    start_phrase = " "
    generated_output = utils.generate_text(model, tokenizer, start_phrase)
    print("--- Generated Text ---")
    print(generated_output)
    