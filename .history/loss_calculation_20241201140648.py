import torch
from utils.debug_utils import save_attention_mask_to_file


def ebae_loss(model, input_ids, attention_mask, tokenizer, device):
    """
    Computes the custom loss for all tokens in the input sequence.
    
    :param model: Hugging Face model (e.g., LLaMA)
    :param input_ids: Tokenized input IDs
    :param attention_mask: Attention mask for the input
    :param tokenizer: Hugging Face tokenizer
    :param device: Device to run the computations (e.g., "cuda" or "cpu")
    :return: Loss value
    """

    # Move data to device
    input_ids = input_ids.to(device)

    for seq in input_ids:
        assert tokenizer.eos_token_id in seq, "EOS token missing!"
        assert len(seq) == 1024, "Sequence length mismatch!"

    attention_mask = attention_mask.to(device)
    # print(attention_mask)

    # Forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Last hidden layer (batch_size, seq_length, hidden_dim)

    # Extract the embedding for the <|end_of_text|> token (e_t)
    eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
    # Check if eos_positions is empty
    if eos_positions[0].numel() < input_ids.size(0):
        raise ValueError("Not all sequences in the batch contain the EOS token.")
    last_token_embeddings = hidden_states[eos_positions]  # Shape: (batch_size, hidden_dim)

    # Use input embedding weights
    vocab_weights = model.get_input_embeddings().weight  # Shape: (vocab_size, hidden_dim)
    vocab_weights_T = vocab_weights.T  # Transpose for (hidden_dim, vocab_size)

    # Compute eT * W_v (logits for full vocabulary)
    vocab_logits = torch.einsum("bh,hv->bv", last_token_embeddings, vocab_weights_T)  # Shape: (batch_size, vocab_size)

    # Compute log softmax over the full vocabulary
    log_prob_vocab = torch.log_softmax(vocab_logits, dim=1)  # Log-probabilities for vocabulary (batch_size, vocab_size)

    # Use input_ids to extract corresponding log probabilities for the sequence tokens
    log_prob_seq = log_prob_vocab.gather(dim=1, index=input_ids)  # Shape: (batch_size, seq_length)

    # Loss computation
    sum_log_prob_seq = log_prob_seq.sum(dim=1)  # Sum along the sequence length (batch_size)

    # Compute the sequence length for each example in the batch
    sequence_lengths = (attention_mask != 0).sum(dim=1)  # Shape: (batch_size)

    # print(f"Sequence lengths:", sequence_lengths)

    # Divide by the sequence length |T|
    avg_log_prob_seq = sum_log_prob_seq / sequence_lengths  # Shape: (batch_size)

    # Compute the loss by averaging over the batch and applying a negative sign
    avg_loss = -avg_log_prob_seq.mean()

    # print(f"vocab_logits range: {vocab_logits.min().item()}, {vocab_logits.max().item()}")
    # print(f"log_prob_vocab: {log_prob_vocab}")
    # print(f"log_prob_seq: {log_prob_seq}")
    # print(f"sum_log_prob_seq: {sum_log_prob_seq}")

    return avg_loss


import torch


def ebae_ebar_loss(model, input_ids, next_input_ds, tokenizer, device):
    """
    Computes the combined EBAE and EBAR loss for the input sequence.

    :param model: Hugging Face model (e.g., LLaMA)
    :param input_ids: Tokenized input IDs
    :param attention_mask: Attention mask for the input
    :param tokenizer: Hugging Face tokenizer
    :param device: Device to run the computations (e.g., "cuda" or "cpu")
    :return: Combined loss value
    """

    # Move data to device
    input_ids = input_ids.to(device)
    next_input_ds = next_input_ds.to(device)

    # save_attention_mask_to_file(attention_mask, "attention_mask.txt")
    
    # Generate custom attention mask
    try:
        custom_attention_mask = create_ebae_ebar_attention_mask(input_ids, tokenizer).to(device)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    # add 1 dimension at index 1
    custom_attention_mask = custom_attention_mask.unsqueeze(1)

    # save_attention_mask_to_file(custom_attention_mask, "custom_attention_mask.txt")

    # Forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=custom_attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Last hidden layer (batch_size, seq_length, hidden_dim)

    # Find the positions of the <\s> tokens (EBAE and EBAR positions)
    sep_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
    assert sep_positions[0].numel() >= 2 * input_ids.size(0), "Each sequence must have at least two <\s> tokens."

    # Extract embeddings for the first and second <\s> tokens
    ebae_embeddings = hidden_states[sep_positions[0][::2], sep_positions[1][::2]]  # EBAE <\s> embeddings
    ebar_embeddings = hidden_states[sep_positions[0][1::2], sep_positions[1][1::2]]  # EBAR <\s> embeddings

    # Use input embedding weights
    vocab_weights = model.get_input_embeddings().weight.to(device)  # Shape: (vocab_size, hidden_dim)
    vocab_weights_T = vocab_weights.T  # Transpose for (hidden_dim, vocab_size)

    # Compute vocab logits for EBAE
    ebae_vocab_logits = torch.einsum("bh,hv->bv", ebae_embeddings, vocab_weights_T)  # Shape: (batch_size, vocab_size)
    ebae_log_prob_vocab = torch.log_softmax(ebae_vocab_logits, dim=1)
    ebae_log_prob_seq = ebae_log_prob_vocab.gather(dim=1, index=input_ids[:, :ebae_vocab_logits.size(0)])
    ebae_sum_log_prob_seq = ebae_log_prob_seq.sum(dim=1)
    ebae_avg_loss = -ebae_sum_log_prob_seq.mean()

    # Compute vocab logits for EBAR
    ebar_vocab_logits = torch.einsum("bh,hv->bv", ebar_embeddings, vocab_weights_T)  # Shape: (batch_size, vocab_size)
    ebar_log_prob_vocab = torch.log_softmax(ebar_vocab_logits, dim=1)
    ebar_log_prob_seq = ebar_log_prob_vocab.gather(dim=1, index=next_input_ds[:, :ebar_vocab_logits.size(0)])
    ebar_sum_log_prob_seq = ebar_log_prob_seq.sum(dim=1)
    ebar_avg_loss = -ebar_sum_log_prob_seq.mean()

    # Combine losses for EBAE and EBAR
    combined_loss = ebae_avg_loss + ebar_avg_loss / 2

    return combined_loss


import torch


def mask_padding_tokens(attention_mask, input_ids, pad_token_id):
    """
    Masks padding tokens in the attention mask by setting their values to 0.
    
    :param attention_mask: Custom attention mask (batch_size, seq_length, seq_length)
    :param input_ids: Tokenized input IDs (batch_size, seq_length)
    :param pad_token_id: ID of the padding token
    :return: Updated attention mask
    """
    # Create padding mask by checking for pad tokens
    padding_mask = input_ids == pad_token_id

    # Set padding tokens in the attention mask to 0
    attention_mask[padding_mask] = 0

    return attention_mask


def create_ebae_ebar_attention_mask(input_ids, tokenizer):
    """
    Creates a custom attention mask that is both auto-regressive and ensures mutual invisibility
    between "The input text is: <\s>" (SELF) and "The next sentence is: <\s>" (NEXT).
    
    :param input_ids: Tokenized input IDs (batch_size, seq_length)
    :param tokenizer: Tokenizer object
    :return: Custom attention mask (batch_size, seq_length, seq_length)
    """
    batch_size, seq_length = input_ids.shape

    # Initialize the base auto-regressive mask (triangular mask)
    autoregressive_mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.float))
    # Expand it to (batch_size, seq_length, seq_length)
    attention_mask = autoregressive_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    # Convert prompts to token IDs
    self_prompt_ids = [576, 1946, 1467, 374, 25]
    next_prompt_ids = [576, 1790, 11652, 374, 25]

    for i, seq in enumerate(input_ids):
        # Find the start positions of the SELF and NEXT prompts
        self_start = find_subsequence(seq.tolist(), self_prompt_ids)
        next_start = find_subsequence(seq.tolist(), next_prompt_ids)

        if self_start is None or next_start is None:
            raise ValueError("Prompts not found in the sequence. Check tokenization.")

        # Modify attention mask to enforce mutual invisibility
        # SELF tokens cannot attend to NEXT tokens and vice versa
        attention_mask[i, next_start:, self_start:next_start] = 0  # NEXT -> SELF region is invisible

    # set padding tokens to be invisible
    attention_mask = mask_padding_tokens(attention_mask, input_ids, tokenizer.pad_token_id)

    return attention_mask


def find_subsequence(sequence, subsequence):
    """
    Finds the start index of a subsequence within a sequence.
    
    :param sequence: Main sequence (list of token IDs)
    :param subsequence: Subsequence to locate (list of token IDs)
    :return: Start index of the subsequence, or None if not found
    """
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            return i
    return None
