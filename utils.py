#!/usr/bin/env python
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, pipeline

from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

import numpy as np

import matplotlib.pyplot as plt

def calc_ppl(model,
             text_data,
             tokenizer,
             stride: int = 1024,
             device='cuda:1'
            ) -> float:
    '''
    model - trained GPT2 model
    text_data - pandas Series with one text sample per row
    stride - perplexity parameter, the greater stride - the faster the computation,
      but the perplexity estimation becomes biased up (worse) 
    '''
    encodings = tokenizer('\n\n'.join(text_data), return_tensors='pt')

    max_length = model.config.n_positions

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def count_parameters(model):
    """
    Return model parameters.
    """
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
def jokes_collate_fn(batch):
    batch_encoded = tokenizer.batch_encode_plus(
        batch,
        padding='max_length',
        return_tensors='pt',
        max_length=128
    )
    input_ids = batch_encoded['input_ids'].to(device)
    attention_mask = batch_encoded['attention_mask'].to(device)
    labels = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
    return (input_ids, attention_mask, labels)
    
def generate_sentences(model, tokenizer, num_return_sequences=250, max_length=None, promt=""):
    """
    Return generated sentences.
    """
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_sentences = generator(promt, num_return_sequences=num_return_sequences, max_length=max_length)
    return [s['generated_text'] for s in generated_sentences]

def toploss_reduction(gen_attentions, nat_attentions):
    """
    Calculate topological loss (L2 form).
    """
    diff = torch.sum((gen_attentions - nat_attentions)**2) / (gen_attentions.shape[0]*128*128)
    return diff
    
def prepare_attention(attentions, t=0.92):
    """
    Prepare attention matricies with threshold.
    """
    return attentions.masked_fill(attentions <= t, 0.0).sum(axis=-1).sum(axis=-1)

def get_generated_attention(model, tokenizer, num_return_sentences, device="cuda:1"):
    """
    Generate new sentences and return attention matricies.
    """
    generated_sentences = generate_sentences(model, tokenizer, num_return_sequences=num_return_sentences)
    gen_dataloader = DataLoader(generated_sentences, batch_size=3)
    gen_attentions = []
    for i, batch in enumerate(gen_dataloader):
        batch_encoded = tokenizer.batch_encode_plus(
            batch,
            padding='max_length',
            return_tensors='pt',
            max_length=128
        )
        input_ids = batch_encoded['input_ids'].to(device)
        attention_mask = batch_encoded['attention_mask'].to(device)

        output = model(
            input_ids=input_ids.to(device), 
            attention_mask=attention_mask.to(device), 
            labels=input_ids.to(device), 
            output_attentions=True
        )
        gen_attentions.append(
            prepare_attention(
                torch.stack([tensor.to("cpu") for tensor in output.attentions], dim=1)
            )
        )
    gen_attentions = torch.cat(gen_attentions, dim=0)
    return gen_attentions

def plot_head_dist(gen, nat, layer=0, head=0):
    """
    Create one head distribtion plot
    """
    fig = plt.figure(figsize=(7, 4))
    wd = wasserstein_distance(gen, nat)
    plt.hist(gen, bins=10, label='generated', color='b', alpha=0.5, density=True)
    plt.hist(nat, bins=10, label='natural', color='r', alpha=0.5, density=True)
    plt.legend()
    plt.title(f"Statistics dist. for layer, head: {layer}, {head}; Wasserstein distance: {wd:.3f}")
    return fig

def save_dist_plots(gen_attentions, nat_attentions, bid=0, logdir="logs/jokes/with_toploss/pictures"):
    """
    Save distributions of heads.
    """
    layers, heads = gen_attentions.shape[1:]
    for layer in range(layers):
        for head in range(heads):
            fig = plot_head_dist(gen_attentions[:, layer, head], nat_attentions[:, layer, head], layer, head)
            fig.savefig(logdir + '/' + f"n{bid}_l{layer}_h{head}.png")
            plt.close(fig)

def train_model(
        model, 
        tokenizer, 
        train_dataloader, 
        val_dataloader=None,
        epochs=30, 
        logging="logs/jokes/default_with_labels", 
        optimizer=None,
        use_labels=False,
        toploss=False):
    """
    Train model without topological loss.
    
    logging: directory for logging results
    """
    if logging is not None:
        writer = SummaryWriter(log_dir=logging)
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=3e-5)
        
    for epoch in tqdm(range(epochs), desc="Training on epoch"):
        model.train()
        for i, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = batch
            if not use_labels:
                labels = input_ids
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            
            writer.add_scalar("Train/loss", output.loss.cpu().item(), len(train_dataloader)*epoch + i)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                if not use_labels:
                    labels = input_ids
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += output.loss.cpu().item()
            writer.add_scalar("Val/loss", val_loss/len(val_dataloader), len(train_dataloader)*epoch)
    return model

def get_topological_loss(model, tokenizer, batches, device='cuda:1', use_labels=False):
    nat_attentions = []
    for batch in batches:
        input_ids, attention_mask, labels = batch
        if not use_labels:
            labels = input_ids
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device), output_attentions=True)
        nat_attentions.append(
            prepare_attention(
                torch.stack([tensor.to("cpu") for tensor in output.attentions], dim=1)
            )
        )
    nat_attentions = torch.cat(nat_attentions, dim=0)
    gen_attentions = get_generated_attention(model, tokenizer, nat_attentions.shape[0], device=device)
    topological_loss = toploss_reduction(nat_attentions, gen_attentions).to(device)
    return nat_attentions, gen_attentions, topological_loss

def train_model_with_toploss(
        model, 
        tokenizer, 
        train_dataloader, 
        val_dataloader=None,
        epochs=30, 
        logging="logs/jokes/with_toploss", 
        optimizer=None,
        use_labels=False,
        device="cuda:1"
    ):
    """
    Train model with topological loss.
    
    logging: directory for logging results
    """
    toploss_every_n_batches = 2000
    
    if logging is not None:
        writer = SummaryWriter(log_dir=logging)
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=3e-5)
        
    batches = [] # Natural batches to be saved
    batches_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training on epoch"):
        model.train()
        for i, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = batch
            batches.append((input_ids.cpu(), attention_mask.cpu(), labels.cpu()))
            
            if not use_labels:
                labels = input_ids
                
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
            loss = output.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if (i+1) % toploss_every_n_batches == 0:
                nat_attentions = []
                gen_attentions = []
                iterable = list(batch_iterable(batches[-500:], 25))
                for b in tqdm(iterable, desc="Calculating topological loss", leave=False):
                    nat_attentions_, gen_attentions_, topological_loss = get_topological_loss(
                        model, tokenizer, b, loss.device, use_labels
                    )
                    optimizer.zero_grad()
                    topological_loss.backward()
                    optimizer.step()
                    
                    writer.add_scalar("Train/topological_loss", topological_loss.cpu().item(), batches_counter)
                    batches_counter += 1
                    gen_attentions.append(gen_attentions_.detach().numpy())
                    nat_attentions.append(nat_attentions_.detach().numpy())
                    
                if (i+1) % (5*toploss_every_n_batches) == 0:
                    save_dist_plots(
                        np.concatenate(gen_attentions), np.concatenate(nat_attentions), bid=(i+1)//(5*toploss_every_n_batches), logdir=logging+ "/pictures")
                
                batches = []
            
            writer.add_scalar("Train/loss", output.loss.cpu().item(), len(train_dataloader)*epoch + i)
        if val_dataloader is not None:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = batch
                    if not use_labels:
                        labels = input_ids
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += output.loss.cpu().item()
            writer.add_scalar("Val/loss", val_loss/len(val_dataloader), len(train_dataloader)*epoch)
    return model