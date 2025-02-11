# import peft
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from loss_calculation import ebae_ebar_loss, ebae_loss, ebae_loss2
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from sklearn.model_selection import train_test_split


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


def evaluate_model(model, dataloader, tokenizer, device):
    """
    Evaluate the model on the given dataloader and return the average loss.
    
    :param model: The language model.
    :param dataloader: DataLoader for the evaluation dataset.
    :param tokenizer: The tokenizer.
    :param device: Device (e.g., "cuda:1").
    :return: Average loss over the evaluation dataset.
    """
    model.eval()
    total_loss = 0.0
    nb_eval_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Using ebae_loss for evaluation (you can switch to ebae_ebar_loss if desired)
            loss = ebae_loss(model, input_ids, attention_mask, tokenizer, device)
            total_loss += loss.item()
            nb_eval_steps += 1

    avg_loss = total_loss / nb_eval_steps if nb_eval_steps > 0 else float("inf")
    return avg_loss


def train_steps(
    model_name: str,
    chunks: list,
    next_sentences: list,
    seq_length: int = 1024,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    epochs: int = 3,
    device: str = "cuda:1",
):
    """
    Fine-tune a language model using LoRA for a specific number of epochs.
    """
    print(f"Model: {model_name}")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Resize model embeddings to match new tokenizer vocabulary
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Tokenize chunks with EBAE prompts and the <\s> token
    print("Tokenizing chunks with EBAE prompts and eos token...")
    ebae_chunks = [f"{chunk} El texto de entrada es: {tokenizer.eos_token}" for chunk in chunks]  # Construct the prompt
    # ebae_ebar_chunks = [f"{chunk} The input text is: {tokenizer.eos_token} The next sentence is: {tokenizer.eos_token}" for chunk in chunks]  # Construct the prompt

    # Split the data into training and evaluation sets (80/20 split)
    train_chunks, eval_chunks = train_test_split(
        ebae_chunks, test_size=0.2, random_state=42
    )

    # for idx, chunk in enumerate(ebae_ebar_chunks):
    #     tokenized_chunk = tokenizer(chunk)["input_ids"]
    #     if len(tokenized_chunk) > seq_length:
    #         print(f"Chunk {idx} is too long: {len(tokenized_chunk)} tokens.")

    # tokenized_chunks = tokenizer(
    #     ebae_ebar_chunks,
    #     max_length=seq_length,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )

    # if not all([tokenizer.eos_token_id in seq for seq in tokenized_chunks["input_ids"]]):
    #     raise ValueError("Some tokenized sequences are missing the EOS token.")

    # print("Tokenization complete.")

    # Prepare dataset and dataloader
    # dataset = EBAE_EBARDataset(ebae_ebar_chunks, next_sentences, tokenizer, seq_length)
    train_dataset = ChunkDataset(train_chunks)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   

    eval_dataset = ChunkDataset(eval_chunks)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)   
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    averaged_losses = {}
    loss_buffer = {}
    gradient_norms = {}

    max_grad_norm = 50.0  # Maximum gradient norm

    # Set the number of accumulation steps to simulate a larger batch size
    gradient_accumulation_steps = 8  # Simulates a batch size of 4 * 8 = 32

    # Training loop
    model.train()

    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")

        averaged_losses[e] = []
        loss_buffer[e] = []
        gradient_norms[e] = []

        step_count = 0
        for step, batch in enumerate(train_dataloader):
            # Extract inputs and masks
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            # next_input_ids = batch["next_input_ids"]
            # next_attention_mask = batch["next_attention_mask"]

            # Compute the custom loss for the last token
            # loss = ebae_ebar_loss(model, input_ids, next_input_ids, tokenizer, device)
            loss = ebae_loss(model, input_ids, attention_mask, tokenizer, device)

            # Normalize the loss by the number of accumulation steps
            loss = loss / gradient_accumulation_steps

            # Handle `None` loss
            if loss is None:
                print(f"Skipping batch {step} due to an error.")
                gradient_norms[e].append(float('nan'))  # Log NaN for skipped batch
                continue  # Skip this batch and move to the next one

            loss.backward()  # Calculate gradients

            # Record loss
            loss_buffer[e].append(loss.item())  # Add to accumulation buffer

            # Update parameters after every `gradient_accumulation_steps`
            if (step + 1) % gradient_accumulation_steps == 0:
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                gradient_norms[e].append(grad_norm)

                optimizer.step()  # Perform optimizer step (update model's parameters)
                optimizer.zero_grad()  # Reset gradients

                # Average losses from this accumulation cycle
                avg_loss = sum(loss_buffer[e]) / gradient_accumulation_steps
                averaged_losses[e].append(avg_loss)
                loss_buffer[e] = []  # Reset buffer for next cycle

                # Increment effective step count
                step_count += 1
                print(f"Effective Step: {step_count}, Averaged Loss: {avg_loss}")

        # In case the last batch doesn't complete an accumulation cycle
        if len(loss_buffer[e]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = sum(loss_buffer[e]) / len(loss_buffer[e])
            averaged_losses[e].append(avg_loss)
            loss_buffer[e] = []
            print(f"End-of-epoch step adjustment, Averaged Loss: {avg_loss}")

        # Evaluate the model at the end of the epoch
        eval_loss = evaluate_model(model, eval_dataloader, tokenizer, device)
        print(f"Epoch {e + 1} Evaluation Loss: {eval_loss}")
        
    # Save the LoRA-adapted model
    output_dir = "./ebar-ebae-model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"EBAR-EBAE-adapted model and tokenizer saved to {output_dir}")

    # Save results for plotting
    torch.save({"losses": averaged_losses, "gradient_norms": gradient_norms}, "training_metrics.pth")
    print("Training metrics saved to training_metrics.pth")
