#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.cuda.empty_cache()
torch.manual_seed(42)


# In[2]:


mode = "ebae"
# mode = "ebae-ebar"


# In[3]:


model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
# model_name = "meta-llama/Llama-3.2-1B"


# In[4]:


import pickle

if mode == "ebae":
    # read chunks using pickle
    with open("wiki_chunks_list_ebae.pkl", "rb") as f:
        chunks = pickle.load(f)

elif mode == "ebae-ebar":
    with open("wiki_chunks_list_ebae_ebar.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("wiki_next_sentences_list_ebae_ebar.pkl", "rb") as f:
        next_sentences = pickle.load(f)
else:
    raise ValueError("Invalid mode")


# Train the model only with train data on a step-basis. In the paper, they performed 10k steps. Using Qwen 0.5B, the training using approx. 5.5k chunks occupies 22.7GB GPU, with batch_size of 4 and gradient accumulation steps of 8.

# In[5]:


from train_ebae import train_steps
from train_hf import main

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

# train_steps(
#     model_name=model_name,
#     chunks=chunks,
#     next_sentences=next_sentences,
#     seq_length=1024,
#     batch_size=4,
#     learning_rate=1e-6,
#     epochs=3
# )

main(
    model_name=model_name,
    chunks=chunks,
    seq_length=1024,
    batch_size=2,
    learning_rate=1e-6,
    epochs=2
)


# In[ ]:


# import re
# import matplotlib.pyplot as plt

# # File path (update this with the actual path of your .txt file)
# file_path = "output.log"

# # Lists to store extracted loss values
# ebae_losses = []
# ebar_losses = []

# # Regular expression to match loss values
# loss_pattern = re.compile(r"EBAE loss: ([\d.]+), EBAR loss: ([\d.]+)")

# # Read and extract losses
# with open(file_path, "r") as file:
#     for line in file:
#         match = loss_pattern.search(line)
#         if match:
#             ebae_losses.append(float(match.group(1)))
#             ebar_losses.append(float(match.group(2)))

# # Plotting the losses
# plt.figure(figsize=(10, 5))
# plt.plot(ebae_losses, label="EBAE Loss", marker="o", linestyle="-")
# plt.plot(ebar_losses, label="EBAR Loss", marker="s", linestyle="--")
# plt.xlabel("Iteration")
# plt.ylabel("Loss Value")
# plt.title("EBAE and EBAR Loss Over Iterations")
# plt.legend()
# plt.grid()

# # Show plot
# plt.show()


# In[ ]:


# import matplotlib.pyplot as plt
# import torch

# # Load saved data
# data = torch.load("training_metrics.pth")
# losses_dict = data["losses"]
# gradient_norms_dict = data["gradient_norms"]

# losses = []
# for epoch, loss_list in losses_dict.items():
#     losses.extend(loss_list)

# gradient_norms = []
# for epoch, gradient_norm_list in gradient_norms_dict.items():
#     gradient_norms.extend(gradient_norm_list)

# # Move gradient norms to CPU if they are GPU tensors
# gradient_norms = [g.cpu().item() if isinstance(g, torch.Tensor) else g for g in gradient_norms]

# # Plot Loss
# plt.figure(figsize=(10, 5))
# plt.plot(losses, label='Loss')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Training Loss per Step')
# plt.legend()
# plt.show()

# # Plot Gradient Norms
# plt.figure(figsize=(10, 5))
# plt.plot(gradient_norms, label='Gradient Norm')
# plt.xlabel('Step')
# plt.ylabel('Gradient Norm')
# plt.title('Gradient Norms per Step')
# plt.legend()
# plt.show()


# In[ ]:


import torch
import matplotlib.pyplot as plt

# Load saved log history from the HF Trainer.
# This file should contain a list of dictionaries with logged metrics.
log_history = torch.load("training_metrics_hf.pth")

# Initialize lists to store metrics
train_losses = []
eval_losses = []
grad_norms = []

# Iterate over each log entry and extract the metrics.
for entry in log_history:
    # Check if a training loss is logged.
    if "loss" in entry:
        train_losses.append(entry["loss"])
    # Check if an evaluation loss is logged.
    if "eval_loss" in entry:
        eval_losses.append(entry["eval_loss"])
    # Check if a gradient norm is logged.
    if "grad_norm" in entry:
        grad_norms.append(entry["grad_norm"])

# Plot training losses and evaluation losses in one plot.
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(eval_losses, label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.show()

# Plot gradient norms in a separate plot.
plt.figure(figsize=(10, 5))
plt.plot(grad_norms, label="Gradient Norms")
plt.xlabel("Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms")
plt.legend()
plt.show()


# In[3]:


# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from huggingface_hub import HfApi, HfFolder, Repository

# # Replace these with your paths and model name
# model_path = "./ebar-ebae-model"
# model_name = "leon-hecht/Qwen-2.5-0.5B-Instruct-spanish-ir"

# # Push the model to the Hugging Face Hub
# from transformers import AutoModelForCausalLM

# # Load model
# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Push the model and tokenizer
# # model.push_to_hub(model_name)
# # tokenizer.push_to_hub(model_name)

