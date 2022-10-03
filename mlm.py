#!/usr/bin/env python
# coding: utf-8

# In[37]:


from transformers import BertTokenizerFast, BertModel
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import collections
import numpy as np
from transformers import default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch import nn


# In[2]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['[newtoken]'])
model = BertModel.from_pretrained('bert-base-uncased')
model.embeddings
weights = model.embeddings.word_embeddings.weight.data
new_weights = torch.cat((weights, weights[101:102]), 0)
new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
model.embeddings.word_embeddings = new_emb
# out = model(**tokenized)
# out.last_hidden_state


# ## Adding a new token to the tokenizer

# In[3]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# If we use `[CLS]`, it will work:

# In[4]:


tokenizer.tokenize("[CLS] Hello world, how are you?")


# But `[newtoken]` will not since it's not in the tokenizer vocab:

# In[5]:


tokenizer.tokenize("[newtoken] Hello world, how are you?")


# Let's try to add it and try again:

# In[6]:


tokenizer.add_tokens(['[newtoken]'])

tokenizer.tokenize("[newtoken] Hello world, how are you?")


# Let's see what the index is:

# In[7]:


tokenized = tokenizer("[newtoken] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
print(tokenized['input_ids'])

tkn = tokenized['input_ids'][0, 0]
print("First token:", tkn)
print("Decoded:", tokenizer.decode(tkn))


# ## Adding a new token to bert

# In[8]:


model = BertModel.from_pretrained('bert-base-uncased')

model.embeddings


# This will not work, as the new token is out of index by default:

# In[9]:


try:
    out = model(**tokenized)
    out.last_hidden_state
except Exception as e:
    print(e)


# Let's see the dimensions of the embeddings weights

# In[10]:


weights = model.embeddings.word_embeddings.weight.data
print(weights.shape)


# We need to add a new dimension. we will copy `[CLS]` and use it to initialize the new index:

# In[11]:


new_weights = torch.cat((weights, weights[101:102]), 0)

new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
new_emb


# Let's add that new `nn.Embedding` back to our model:

# In[12]:


model.embeddings.word_embeddings = new_emb

model.embeddings


# And try to run our tokenized text into the model again:

# In[13]:


out = model(**tokenized)
out.last_hidden_state


# Optional sanity check: let's see if it the result will be the same if we replace the token with `[CLS]`

# In[14]:


model = BertModel.from_pretrained('bert-base-uncased')


# In[15]:


out2 = model(
    **tokenizer("[CLS] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
)


# In[16]:


torch.all(out.last_hidden_state == out2.last_hidden_state)


# ## Picking a pretrained model for masked language modeling
# pick a model here https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads with fill-mask filter
# 

# In[17]:


model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


# In[18]:


distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")


# In[19]:





# In[20]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[21]:


text = "This is a great [MASK]."
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


# In[22]:


imdb_dataset = load_dataset("imdb")
imdb_dataset


# In[23]:


sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")


# In[24]:


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets


# In[25]:


# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
    


# In[29]:


chunk_size = 128
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")


# In[30]:


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets


# In[31]:


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")


# In[32]:


wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)                                


# In[36]:


train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    # fp16=True,
    logging_steps=logging_steps,
)


# In[38]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)


# In[39]:


import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# In[40]:


trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
