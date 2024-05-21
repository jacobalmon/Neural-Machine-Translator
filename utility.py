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
