from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from loss_calculation import ebae_loss  # or import ebae_ebar_loss if preferred
# set visible devices to GPU:1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import DataCollatorWithPadding

import sys
print("Executable: ", sys.executable)


class CustomDataCollator(DataCollatorWithPadding):
    """
    A custom data collator that adds a 'return_loss' flag to the batch.
    """
    def __call__(self, features):
        # Use the default collator behavior for padding.
        batch = super().__call__(features)
        # Add a flag to ensure loss is returned during evaluation.
        batch["return_loss"] = True
        return batch


class ChunkDataset(Dataset):
    """Custom PyTorch Dataset for tokenized chunks."""
    def __init__(self, tokenized_chunks):
        self.input_ids = tokenized_chunks["input_ids"]
        self.attention_mask = tokenized_chunks["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class EBAE_EBARDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized EBAE and EBAR data.
    Separates the input prompt and the next sentence.
    """
    def __init__(self, prompts, next_sentences, tokenizer, seq_length):
        """
        Initialize the dataset.

        :param prompts: List of input prompts for EBAE.
        :param next_sentences: List of next sentences for EBAR.
        :param tokenizer: Tokenizer to process the text.
        :param seq_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Tokenize the prompts
        self.prompt_data = tokenizer(
            prompts,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize the next sentences separately
        self.next_sentence_data = tokenizer(
            next_sentences,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.prompt_data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.prompt_data["input_ids"][idx],
            "attention_mask": self.prompt_data["attention_mask"][idx],
            "next_input_ids": self.next_sentence_data["input_ids"][idx],
            "next_attention_mask": self.next_sentence_data["attention_mask"][idx],
        }


class CustomTrainer(Trainer):
    """
    A custom Trainer that overrides compute_loss to use the custom ebae_loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss using the custom ebae_loss function.

        :param model: The language model.
        :param inputs: Batch from the dataset.
        :param return_outputs: Whether to return model outputs along with the loss.
        :return: Loss (and optionally the outputs).
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        device = next(model.parameters()).device
        # Use custom loss function; here we use ebae_loss.
        loss = ebae_loss(model, input_ids, attention_mask, self.tokenizer, device)
        return (loss, inputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Overrides the default prediction_step to compute the evaluation loss
        using the custom ebae_loss function.
        """
        with torch.no_grad():
            # Retrieve input_ids and attention_mask from the batch.
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            # Get device from model parameters.
            device = next(model.parameters()).device

            # Compute loss using your custom loss function.
            loss = ebae_loss(model, input_ids, attention_mask, self.processing_class, device)
            # Detach and move the loss to CPU to free GPU memory.
            loss = loss.detach().cpu()

            if prediction_loss_only:
                # Return loss only.
                return loss, None, None
            else:
                # Optionally, perform a forward pass to get predictions.
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                # Return loss, logits, and no labels (or you can pass labels if you have them).
                return loss, logits, None


def main(model_name, chunks, seq_length, batch_size, learning_rate, epochs, next_sentences=None):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # # Construct the prompts by adding cues (EBAE and EBAR).
    # prompts = [
    #     f"{chunk} The input text is: {tokenizer.eos_token} The next sentence is: {tokenizer.eos_token}"
    #     for chunk in chunks
    # ]

    prompts = [f"{chunk} El texto de entrada es: {tokenizer.eos_token}" for chunk in chunks]  # Construct the prompt

    # # Perform an 80/20 train-test split.
    # train_prompts, eval_prompts, train_next, eval_next = train_test_split(
    #     prompts, next_sentences, test_size=0.2, random_state=42
    # )

    # Tokenize the chunks
    tokenized_prompts = tokenizer(
        prompts,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    full_dataset = ChunkDataset(tokenized_prompts)

    # Optional: perform an 80/20 train-test split
    indices = list(range(len(full_dataset)))
    train_indices, eval_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, eval_indices)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./output-model",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=150,
        gradient_accumulation_steps=16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=150,
        max_grad_norm=50.0,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
    )

    # Instantiate the custom Trainer.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=CustomDataCollator(tokenizer),
        # compute_metrics=compute_metrics,
    )

    # Train the model.
    trainer.train()

    # Save the model and tokenizer.
    torch.save(trainer.state.log_history, "training_metrics_hf.pth")
    trainer.save_model()
    tokenizer.save_pretrained("./output-model")


if __name__ == "__main__":
    main()
