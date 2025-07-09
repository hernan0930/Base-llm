import torch
import tiktoken
from data.dataloader_gpt import *
from torch.utils.data import DataLoader

def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Creates a PyTorch DataLoader for GPT-2 training.
    Args:
        txt (str): Input text data.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum sequence length accepted by the model.
        stride (int): Step size for sliding window (half of the previous input text).
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Drop the last incomplete batch.
        num_workers (int): Number of subprocesses for data loading.
    Returns:
        DataLoader: PyTorch DataLoader object.
    """

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def text_to_token_ids(text, tokenizer):
    """
        Converts input text to a tensor of token IDs using the provided tokenizer.
        Args:
            text (str): Input text.
            tokenizer: Tokenizer object.
        Returns:
            torch.Tensor: Tensor of token IDs with batch (dummy) dimension.
        """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Converts a tensor of token IDs back to text.
    Args:
        token_ids (torch.Tensor): Tensor of token IDs (with batch dimension).
        tokenizer: Tokenizer object.
    Returns:
        str: Decoded text.
    """
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates text by autoregressively sampling from the model.
    Args:
        model: Language model.
        idx (torch.Tensor): Initial context token IDs (batch, n_tokens).
        max_new_tokens (int): Number of tokens to generate.
        context_size (int): Maximum context window size for the model.
    Returns:
        torch.Tensor: Tensor of token IDs including generated tokens.
    """
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
        
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates and prints a sample text from the model given a start context.
    Args:
        model: Language model.
        tokenizer: Tokenizer object.
        device: Torch device (cpu or cuda).
        start_context (str): Initial text prompt.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    # model.train()


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates the cross-entropy loss for a single batch.
    Args:
        input_batch (torch.Tensor): Input token IDs.
        target_batch (torch.Tensor): Target token IDs.
        model: Language model.
        device: Torch device.
    Returns:
        torch.Tensor: Loss value.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the average loss over a DataLoader.
    Args:
        data_loader (DataLoader): DataLoader object.
        model: Language model.
        device: Torch device.
        num_batches (int, optional): Number of batches to evaluate. If None, use all.
    Returns:
        float: Average loss over the evaluated batches.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches