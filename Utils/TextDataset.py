# Define a Dataset class
from torch.utils.data import Dataset, Subset
import torch
import numpy as np


# Define a Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.labels = labels
        self.tokenizer = tokenizer
        tokenized_output = self.tokenizer(list(texts), truncation=True, padding=True, max_length=max_length,
                                          return_offsets_mapping=True)
        self.encodings = {key: tokenized_output[key] for key in tokenized_output if key != 'offset_mapping'}
        # this to is for later map substrings of text back to tokens
        self.offset_mapping = tokenized_output['offset_mapping']
        self.texts = list(texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_original_texts(self, idx):
        return self.texts[idx]

    def match_original_text_to_tokens(self, idx):
        """
        If you simply use decode then it will not map back to original token.
        :param idx:
        :return:
        """
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        offsets = self.offset_mapping[idx]
        original_text = self.texts[idx]

        word_pieces = []
        for token_id, (start, end), attn in zip(input_ids, offsets, attention_mask):
            if attn == 0:
                # Padding token
                word_pieces.append('[PAD]')
            elif start == end:
                # Special token (non-textual, e.g., [CLS], [SEP])
                word_pieces.append(f'SPECIAL_{self.tokenizer.decode([token_id]).strip()}')
            else:
                # Regular token
                word_pieces.append(original_text[start:end])

        return word_pieces
