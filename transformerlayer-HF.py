#!pip install transformers

from types import SimpleNamespace
from collections import Counter
import os
import re
import pathlib
import array
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import matplotlib.pyplot as plt
import time

from transformers import BertConfig, BertModel, AutoConfig, AutoModel


DATASET_VERSION = 'ca-100'
COMPETITION_ROOT = '../input/wordvectors'
DATASET_ROOT = f'../input/text-preprocessing/data/{DATASET_VERSION}'
WORKING_ROOT = f'data/{DATASET_VERSION}'
DATASET_PREFIX = 'ca.wiki'

class Vocabulary(object):
    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.token2idx = {}
        self.idx2token = []
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        if pad_token is not None:
            self.pad_index = self.add_token(pad_token)
        if unk_token is not None:
            self.unk_index = self.add_token(unk_token)
        if eos_token is not None:
            self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def get_index(self, token):
        if isinstance(token, str):
            return self.token2idx.get(token, self.unk_index)
        else:
            return [self.token2idx.get(t, self.unk_index) for t in token]

    def get_token(self, index):
        return self.idx2token[index]

    def __len__(self):
        return len(self.idx2token)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

def batch_generator(idata, target, batch_size, shuffle=True):
    nsamples = len(idata)
    if shuffle:
        perm = np.random.permutation(nsamples)
    else:
        perm = range(nsamples)

    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if target is not None:
            yield idata[batch_idx], target[batch_idx]
        else:
            yield idata[batch_idx], None

def load_preprocessed_dataset(prefix):
    # Try loading precomputed vocabulary and preprocessed data files
    token_vocab = Vocabulary()
    token_vocab.load(f'{prefix}.vocab')
    data = []
    for part in ['train', 'valid', 'test']:
        with np.load(f'{prefix}.{part}.npz') as set_data:
            idata, target = set_data['idata'], set_data['target']
            data.append((idata, target))
            print(f'Number of samples ({part}): {len(target)}')
    print("Using precomputed vocabulary and data files")
    print(f'Vocabulary size: {len(token_vocab)}')
    return token_vocab, data

def train(model, criterion, optimizer, idata, target, batch_size, device, log=False):
    batch_losses = []
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    for X, y in batch_generator(idata, target, batch_size, shuffle=True):
        # Get input and target sequences from batch
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        ntokens += y.numel()
        niterations += 1
        if niterations == 200 or niterations == 500 or niterations % 1000 == 0:
            print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={100*ncorrect/ntokens:.1f}, loss={total_loss/ntokens:.2f}')
            batch_losses.append((niterations,loss.item()))

            
    total_loss = total_loss / ntokens
    accuracy = 100 * ncorrect / ntokens
    if log:
        print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={accuracy:.1f}, loss={total_loss:.2f}')

    plot_batch_loss(batch_losses)
    return accuracy, total_loss

def validate(model, criterion, idata, target, batch_size, device):
    model.eval()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    y_pred = []
    with torch.no_grad():
        for X, y in batch_generator(idata, target, batch_size, shuffle=False):
            # Get input and target sequences from batch
            X = torch.tensor(X, dtype=torch.long, device=device)
            output = model(X)
            if target is not None:
                y = torch.tensor(y, dtype=torch.long, device=device)
                loss = criterion(output, y)
                total_loss += loss.item()
                ncorrect += (torch.max(output, 1)[1] == y).sum().item()
                ntokens += y.numel()
                niterations += 1
            else:
                pred = torch.max(output, 1)[1].detach().to('cpu').numpy()
                y_pred.append(pred)

    if target is not None:
        total_loss = total_loss / ntokens
        accuracy = 100 * ncorrect / ntokens
        return accuracy, total_loss
    else:
        return np.concatenate(y_pred)

def save_results(results, filename="results.csv"):
    """Saves the results to a CSV file.

    Args:
        results (list): A list of dictionaries, where each dictionary contains model results.
        filename (str, optional): The name of the CSV file to save to. Defaults to "results.csv".
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
def plot_loss_accuracy(results, filename="loss_accuracy.png"):
    """Creates a plot of loss and accuracy over epochs for each model,
    including training, wiki validation, and El Periodico validation.

    Args:
        results (list): A list of dictionaries, where each dictionary contains model results,
                        including 'Loss', 'Accuracy', 'Wiki_acc', 'Valid_acc', 'Wiki_loss', 'Valid_loss' lists.
        filename (str, optional): The name of the plot image file to save to. Defaults to "loss_accuracy.png".
    """

    num_models = len(results)
    fig, axes = plt.subplots(num_models, 2, figsize=(12, 6 * num_models))

    for i, result in enumerate(results):
        model_name = result['Model']
        epochs = range(1, len(result['Loss']) + 1)

        # Combined Loss Plot (Training, Wiki, El Periodico)
        ax1 = axes[i, 0] if num_models > 1 else axes[0]
        ax1.plot(epochs, result['Loss'], label='Training Loss')
        ax1.plot(epochs, result.get('Wiki_loss', []), label='Wiki Loss')
        ax1.plot(epochs, result.get('Valid_loss', []), label='El Periodico Loss')
        ax1.set_title(f'{model_name} - Losses and Accuracies') # More comprehensive title
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss/Accuracy')
        ax1.grid(True)
        ax1.legend()

        # Accuracy Plot (Training, Wiki, El Periodico)
        ax2 = axes[i, 1] if num_models > 1 else axes[1]
        ax2.plot(epochs, result['Accuracy'], label='Training Accuracy')
        ax2.plot(epochs, result.get('Wiki_acc', []), label='Wiki Accuracy')
        ax2.plot(epochs, result.get('Valid_acc', []), label='El Periodico Accuracy')
        ax2.set_title(f'{model_name} - Accuracies')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Loss and accuracy plot saved to {filename}")
    plt.show()

def plot_batch_loss(batch_losses, filename="batch_loss.png"):
    """Creates a plot of the training loss for each batch.

    Args:
        batch_losses (list): A list of loss values, where each value represents the loss for a single batch.
        filename (str, optional): The name of the plot image file to save to. Defaults to "batch_loss.png".
    """
    iterations, losses = zip(*batch_losses)  # Unpack the list of tuples
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses)
    plt.title('Training Loss per Batch (vs. Iteration)')
    plt.xlabel('Iteration Number')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    print(f"Batch loss plot saved to {filename}")
    plt.show()



""" #################################### MODELO ####################################"""


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project and split into multiple heads
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Multi-head self-attention
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class Predictor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, context_words=6, num_layers=2, num_heads=8):
        super().__init__()
        # Embedding layer
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
        # Linear output layer
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        
        # Transformer layers
        transformer_layers = [
            TransformerLayer(
                d_model=embedding_dim, 
                num_heads=num_heads, 
                dim_feedforward=embedding_dim * 4
            ) for _ in range(num_layers)
        ]
        self.transformer_stack = nn.ModuleList(transformer_layers)
        
        # Positional embedding
        self.position_embedding = nn.Parameter(torch.Tensor(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, input):
        # Embedding
        e = self.emb(input)
        
        # Add positional embedding
        u = e + self.position_embedding
        
        # Apply transformer layers
        v = u
        for transformer_layer in self.transformer_stack:
            v = transformer_layer(v)
        
        # Global average pooling
        x = v.sum(dim=1)
        
        # Output projection
        y = self.lin(x)
        
        return y



""" ############################### Hugging Face Model ########################## """

# -------------------------------
# New Predictor using Hugging Face Transformers
# -------------------------------
class HFPredictor(nn.Module):
    def __init__(self, num_embeddings, context_words, hf_embedding_dim, hf_layers, hf_heads, hf_pretrained_model=None):
        """
        Parameters:
         - num_embeddings: vocabulary size.
         - context_words: total input sequence length.
         - hf_embedding_dim: hidden size for the transformer.
         - hf_layers: number of transformer layers.
         - hf_heads: number of attention heads.
         - hf_pretrained_model: if provided, loads a pretrained model by name.
        """
        super(HFPredictor, self).__init__()
        self.context_words = context_words

        if hf_pretrained_model:
            # Load a pretrained model. Note that the pretrained model's vocabulary might not match num_embeddings.
            self.config = AutoConfig.from_pretrained(hf_pretrained_model, 
                                                     hidden_size=hf_embedding_dim,
                                                     num_hidden_layers=hf_layers,
                                                     num_attention_heads=hf_heads,
                                                     vocab_size=num_embeddings,
                                                     max_position_embeddings=context_words)
            self.transformer = AutoModel.from_pretrained(hf_pretrained_model, config=self.config)
        else:
            # Build a model from scratch using a BERT configuration
            self.config = BertConfig(
                vocab_size=num_embeddings,
                hidden_size=hf_embedding_dim,
                num_hidden_layers=hf_layers,
                num_attention_heads=hf_heads,
                intermediate_size=hf_embedding_dim * 4,
                max_position_embeddings=context_words,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            self.transformer = BertModel(self.config)
        
        # Linear layer to predict the target token from the central hidden state
        self.fc = nn.Linear(self.config.hidden_size, num_embeddings)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len) with seq_len == context_words
        outputs = self.transformer(input_ids)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        # Assuming the central word is at position context_words//2
        center_idx = self.context_words // 2
        central_hidden = hidden_states[:, center_idx, :]  # (batch, hidden_size)
        logits = self.fc(central_hidden)
        return logits


""" #################################### MAIN ####################################"""
print("0. START PROGRAM")
params = SimpleNamespace(
    embedding_dim = 256,
    window_size = 7,
    batch_size = 2048,
    epochs = 4,
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    train = True
)

params = SimpleNamespace(
    model_name = "MultiHeadAttention with pooling",
    embedding_dim = 256,
    window_size = 7,
    batch_size = 2048,
    epochs = 4,
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    train = True,
    layers = 2,
    heads = 8,
)

params = SimpleNamespace(
    model_name = "Hugging Face Transformer Predictor",
    embedding_dim = 256,   # hf_embedding_dim
    window_size = 7,       # total sequence length (3 before, 1 center, 3 after)
    batch_size = 2048,
    epochs = 4,
    lr = 0.001,          # learning rate for the optimizer
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    modelname = f'{WORKING_ROOT}/{DATASET_VERSION}_hf.pt',
    train = True,
    hf_layers = 2,       # number of transformer layers
    hf_heads = 8,        # number of attention heads
    hf_pretrained_model = None  # set to a string like "bert-base-uncased" to load a pretrained model; otherwise, build from scratch
)

# Create working dir
pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)

# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")

vocab, data = load_preprocessed_dataset(params.preprocessed)

# 'El Periodico' validation dataset
valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
tokens = valid_x_df.columns[1:]
valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

print("1. DATASET LOADED!")
# model = Predictor(len(vocab), params.embedding_dim).to(device)


### MODELO 1 ####
if False:
    model = Predictor(
        num_embeddings=len(vocab), 
        embedding_dim=params.embedding_dim,  # 256
        context_words=params.window_size-1,  # 6
        num_layers=params.layers,  # Number of transformer layers
        num_heads=params.heads    # Number of attention heads
    ).to(device)


### HF MODEL ####
model = HFPredictor(
    num_embeddings=len(vocab),
    context_words=params.window_size,            # 7 tokens total
    hf_embedding_dim=params.embedding_dim,        # 256
    hf_layers=params.hf_layers,                   # e.g., 2 transformer layers
    hf_heads=params.hf_heads,                     # e.g., 8 attention heads
    hf_pretrained_model=params.hf_pretrained_model  # None or e.g., "bert-base-uncased"
).to(device)

print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(reduction='sum')

train_accuracy = []
wiki_accuracy = []
valid_accuracy = []
train_losses = []
wiki_losses = []
valid_losses = []
results = [] # Store the result in the list.
model_name = 'Custom Transformer (Model 1)' # Example model name
training_time = 0  # Initialize training time

print("2. START TRAINING")
start_time = time.time() # Start timing

for epoch in range(params.epochs):
    acc, loss = train(model, criterion, optimizer, data[0][0], data[0][1], params.batch_size, device, log=True)
    train_accuracy.append(acc)
    train_losses.append(loss)
    print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
    acc, loss = validate(model, criterion, data[1][0], data[1][1], params.batch_size, device)
    wiki_accuracy.append(acc)
    wiki_losses.append(loss)
    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (wikipedia)')
    acc, loss = validate(model, criterion, valid_x, valid_y, params.batch_size, device)
    valid_accuracy.append(acc)
    valid_losses.append(loss)
    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (El Periódico)')


end_time = time.time() # End timing
training_time = end_time - start_time # Calculate training time

# Save model
torch.save(model.state_dict(), params.modelname)

# 'El Periodico' test dataset
valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
y_pred = validate(model, None, test_x, None, params.batch_size, device)
y_token = [vocab.get_token(index) for index in y_pred]

submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
print(submission.head())
submission.to_csv('submission.csv', index=False)

results.append({
    'Model': params.model_name,
    'Loss': train_losses,
    'Accuracy': train_accuracy,
    'Wiki_acc': wiki_accuracy,
    'Valid_acc': valid_accuracy,
    'Wiki_loss': wiki_losses,
    'Valid_loss': valid_losses,
    'Training Time (s)': training_time,
    'Parameters': sum(p.numel() for p in model.parameters()),
    'Hyperparameters': {
        'Layers': params.hf_layers,
        'Embed_dim': params.embedding_dim,
        'Heads': params.hf_heads,
        'Pretrained': params.hf_pretrained_model if params.hf_pretrained_model else "None"
    }
})


save_results(results)
plot_loss_accuracy(results)