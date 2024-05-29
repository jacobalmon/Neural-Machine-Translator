import random
import spacy
import torch
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
from model import Encoder, Decoder, Seq2Seq
from utility import load_data, train, evaluate, translate_sentence

# Setting a Random Seed.
seed = 1234
random.seed(seed)
torch.manual_speed(seed)
torch.backends.cudnn.deterministic = True

# Loading Data.
src, trg, train_data, valid_data, test_data = load_data()

# Create Iterators for Data.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Define Model's Attributes.
input_dim = len(src.vocab)
output_dim = len(trg.vocab)
encoder_embedded_dim = 256
decoder_embedded_dim = 256
hidden_dim = 512
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

# Initialize Encoder and Decoder.
encoder = Encoder(input_dim, encoder_embedded_dim, hidden_dim, num_layers, encoder_dropout)
decoder = Decoder(output_dim, decoder_embedded_dim, hidden_dim, num_layers, decoder_dropout)

# Define Device to Run Model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Model.
model = Seq2Seq(encoder, decoder, device).to(device)

# Define Optimizer.
optimizer = optim.Adam(model.parameters())

# Define Loss Function
trg_pad_index = trg.vocab.stoi[trg.pad_token]
criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_index)

num_epochs = 10
clip = 1

# Getting Loss from training and evaluating.
for epoch in range(num_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_iterator, criterion)

    # Testing Purposes.
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Test.
sentence = "I love cats."
translation = translate_sentence(sentence, src, trg, model, device)
print("Translation:", " ".join(translation))