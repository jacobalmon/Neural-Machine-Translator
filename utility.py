import torch
from torchtext.data import Field, TabularDataset

def load_data():
    # Defining Source Language (English).
    src = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', 
                init_token='<sos>', eos_token='<eos>', lower=True
            )
    
    # Defining Target Language (Japanese).
    trg = Field(tokenize='spacy', tokenizer_language='ja_core_news_sm', 
                init_token='<sos>', eos_token='<eos>', lower=True
            )
    
    # Defining Dataset.
    data_fields = [('src', src), ('trg', trg)]

    # Loading Data.
    training_data, valid_data, testing_data = TabularDataset.splits(
        path='data/', train='train.tsv', valid='valid.tsv', test='test.tsv',
        format='tsv', fields=data_fields
    )

    # Building Voabularies for the languages.
    src.build_vocab(training_data, min_freq=2)
    trg.build_vocab(training_data, min_freq=2)

    return src, trg, training_data, valid_data, testing_data

def train(model, iterator, optimizer, criterion, clip):
    # Set Model to Training Mode.
    model.train()

    # Initialize Total Epoch Loss.
    epoch_loss = 0
    
    # Iterate over each batch within the iterator.
    for i, batch in enumerate(iterator):
        src = batch.src # Source Tensor.
        trg = batch.trg # Target Tensor.

        optimizer.zero_grad() # Reset Gradients.
        output = model(src, trg) # Output Tensor.
        output_dim = output.shape[-1] # Dimension of Output.

        # Reshaping Tensors to Calculate Loss.
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg) # Calculates Loss between Tensors.
        loss.backward()

        # Clipping Gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update Model.
        optimizer.step()

        # Update the total loss.
        epoch_loss += loss.item()
    
    # Return the Average Loss.
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    # Set Model to Evaluation Mode.
    model.eval()

    # Initialize the total epoch loss.
    epoch_loss = 0
    
    # Disables Gradient Calculation for Evaluation.
    with torch.no_grad():
        # Iterate over each batch within the iterator.
        for i, batch in enumerate(iterator):
            src = batch.src # Source Tensor.
            trg = batch.trg # Target Tensor.

            output = model(src, trg, 0) # Output Tensor.
            output_dim = output.shape[-1] # Dimension of Output.

            # Reshaping Tensors to Calculate Loss.
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # Calculates Loss between Tensors.
            loss = criterion(output, trg)

            # Update the total loss.
            epoch_loss += loss.item()

    # Return the Average Loss.
    return epoch_loss / len(iterator)