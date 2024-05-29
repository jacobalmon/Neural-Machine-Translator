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

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
     # Set Model to Evaluation Mode.
    model.eval()

    # Tokenize the input sentence and convert to lowercase.
    tokens = [token.lower() for token in src_field.tokenize(sentence)]

    # Add the sos and eos tokens.
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # Convert tokens to their corresponding indices in the vocabulary.
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # Convert the list of indices to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    # Disables Gradient Calculation for Evaluation.
    with torch.no_grad():
        # Encodes the source sentence.
        hidden, cell = model.encoder(src_tensor)
    
    # Initialize the target sequence with the sos token.
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # Generate the target sequence one token at a time.
    for _ in range(max_len):
        # Get the last predicted token and convert it to a tensor.
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            # Decodes the token to get the next token's probabilities.
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        # Get the token with the highest probability.
        pred_token = output.argmax(1).item()

        # Append the predicted token to the target sequence.
        trg_indexes.append(pred_token)
        
        # Stop if the eos token is predicted.
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # Convert the target sequence of indices back to tokens.
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # Return the target sequence excluding the <sos> token.
    return trg_tokens[1:]