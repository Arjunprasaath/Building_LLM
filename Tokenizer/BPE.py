# imports
import os
import json
import math
import pickle
import numpy as np
import regex as re
from tqdm import tqdm
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size, merges, special_tokens, file_path = None):
        self.PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.vocab = {bytes([idx]): idx for idx in range(256)}
        self.merges = merges or {}
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {}
        self.chunk_size = 8 * 1024 * 1024 # 8MB
        self.file_path = file_path
        if self.special_tokens:
            for token, token_id in self.special_tokens.items():
                self.vocab[token.encode()] = token_id

        # load for small file or no file_path provided
        self.splits = []
        self.streaming = False
        if file_path:
            file_size = self._get_file_size(file_path)
            if file_size > 100 * 1024 * 1024: # 100 MB
                self.streaming = True
            else:
                # load normally
                for chunk in self.read_in_chunks(file_path):
                    words = re.findall(self.PAT, chunk)
                    for word in words:
                        # splits input document into binary document
                        # [[b'H', b'e', b'l', b'l', b'o'],
                        #  [b' ', b't', b'h', b'i', b's']]
                        self.splits.append([char.encode() for char in word])

    def _get_file_size(self, file_path):
        return os.path.getsize(file_path)
    
    def read_in_chunks(self, file_path):
        """Generator to read file in chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            overlap = ""
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    if overlap:
                        yield overlap
                    break
                if len(chunk) == self.chunk_size:
                    last_space = chunk.rfind(' ')
                    if last_space > 0:
                        complete_chunk = overlap + chunk[:last_space]
                        overlap = chunk[last_space:]
                        yield complete_chunk
                    else:
                        yield overlap + chunk
                        overlap = ""
                else:
                    yield overlap + chunk
                    break


    def count_pairs(self):
        """
        Counts the number of time 2 characters appears together.
        Take the splits lists and counts for occurrences.
        returns a dictionary with key being 2 binary characters and value being the number of times they appear together.
        eg: defaultdict(int,
            {(b'H', b'e'): 1,
             (b'e', b'l'): 1,
             (b'l', b'l'): 1,
             (b'l', b'o'): 1,)})
        """
        pair_counts = defaultdict(int)
        if self.streaming and self.file_path:
            # stream and count pairs
            for chunk in self.read_in_chunks(self.file_path):
                chunk_splits = self._get_chunk_splits(chunk)
                for split in chunk_splits:
                    if len(split) <= 1:
                        continue
                    for i in range(len(split) - 1):
                        pair = (split[i] , split[i + 1])
                        pair_counts[pair] += 1
        else:
            # count pairs directly
            for split in self.splits:
                if len(split) == 1:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i] , split[i + 1])
                    pair_counts[pair] += 1
            # print(pair_counts)
        return pair_counts

    def _get_chunk_splits(self, chunk):
        """convert text chunks to splits and apply current merges"""
        words = re.findall(self.PAT, chunk)
        chunk_splits = []
        for word in words:
            split  = [char.encode() for char in word]
            for merge_pair, merged_tokens in self. merges.items():
                if len(split) <= 1:
                    break
                i = 0
                while i < len(split) - 1:
                    if split[i] == merge_pair[0] and split[i + 1] == merge_pair[1]:
                        split = split[: i] + [merged_tokens] + split[i + 2:]
                    else:
                        i += 1
            chunk_splits.append(split)
        return chunk_splits

    def max_pairs(self, pair_counts):
        """
        returns the max count pair from the pair counts and also combines them to form a new unicode representation.
        eg: (((b't', b'h'), 2), b'th')
        """
        if not pair_counts:
            return  None, None
        max_pair = max(pair_counts.items(), key= lambda x: x[1])
        new_pair_bytes = max_pair[0][0] + max_pair[0][1]
        return max_pair, new_pair_bytes

    def update_splits(self, max_pair):
        """
        Updated the original splits with the merged characters.
        eg: OLD: [b' ', b't', b'h', b'i', b's']
            NEW: [b' ', b'th', b'i', b's']
        """
        # for streaming mode, merges are applied in _get_chunks_splits function
        if not self.streaming:
            for idx in range(len(self.splits)):
                split = self.splits[idx]
                if len(split) <= 1:
                    continue
                i = 0
                while i < len(split) - 1:
                    if split[i] == max_pair[0][0] and split[i + 1] == max_pair[0][1]:
                        # print("OLD:", split)
                        split = split[:i] + [max_pair[0][0] + max_pair[0][1]] + split[i + 2:]
                        self.splits[idx] = split
                        # print("NEW:", split)
                    else:
                        i += 1
        return self.splits

    def train_tokenizer(self):
        reserved_ids = {tok: self.vocab_size - 1 - i for i, tok in enumerate(self.special_tokens.keys())}
        for token, token_id in  reserved_ids.items():
            self.vocab[token.encode()] = token_id
            self.special_tokens[token] = token_id

        next_id = 256

        pbar = tqdm(desc = "Training BPE tokenizer", unit = " merges")
        iteration = 0

        while next_id < min(self.vocab_size - len(reserved_ids), self.vocab_size):
            iteration += 1
            # count paris
            pairs = self.count_pairs()
            if not pairs:
                print("Tokenizer training completed training early!")
                break
            
            # get the max count pair and their merged pair
            max_pair, new_pair_bytes = self.max_pairs(pairs)
            if max_pair is None:
                break
            
            if new_pair_bytes not in self.vocab:
                # add them to the merges dictionary
                self.merges[max_pair[0]] = new_pair_bytes
                # update the new merged pair to the vocabulary
                self.vocab[new_pair_bytes]= next_id
                next_id += 1

            # update the merged pair in the splits
            self.splits = self.update_splits(max_pair)

            pbar.set_description(f"Training BPE tokenizer (vocab: {len(self.vocab)}/{self.vocab_size})")
            pbar.update(1)

            if iteration > self.vocab_size * 2:
                print(f"Breaking due to too many iterations: {iteration}")
                break

        pbar.close()
        print(f"Training completed. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")
        return self.vocab, self.merges

    def encode(self, input_text, allow_unknows = True):

        # first handle special tokens
        encoded_ids = []
        if self.special_tokens:
            tokens = re.split(f"({'|'.join(map(re.escape, self.special_tokens.keys()))})", input_text)
        else:
            tokens = [input_text]

        for token in tokens:
            if token == "":
                continue
            if token in self.special_tokens:
                encoded_ids.append(self.special_tokens[token])
                continue

            # split the input text into original splits format (list(list(b'characters')))
            input_splits = [[char.encode() for char in word] for word in re.findall(self.PAT, token)]
            # iterate over merged dictionary
            for merge_pairs, merged_token in self.merges.items():
                for idx in range(len(input_splits)):
                    if len(input_splits[idx]) <= 1:
                        continue
                i = 0
                while i < len(input_splits[idx]) - 1:
                    # merge the input tokens that have been merged (learnt) by the tokenizer
                    if input_splits[idx][i] == merge_pairs[0] and input_splits[idx][i + 1] == merge_pairs[1]:
                        # print("OLD",input_splits[idx])
                        input_splits[idx] = input_splits[idx][:i] + [merged_token] + input_splits[idx][i + 2:]
                        # print("NEW", input_splits[idx])
                    else:
                        i += 1
        for segment in input_splits:
            for token in segment:
                if token in self.vocab:
                    # use the vocabulary to convert the tokens into integers
                    encoded_ids.append(self.vocab[token])
                elif allow_unknows and '<UNK>' in self.special_tokens:
                    encoded_ids.append(self.special_tokens['<UNK>'])
        # print(encoded_ids)
        return encoded_ids


    def pretokenize_file(self, file_path, out_path = 'tokens.npy', dtype = np.int32):
        """Pre-tokenize a large text file in streaming mode and save tokens as a memory mapped numpy file"""
        print("Counting tokens ...")
        
        file_size = self._get_file_size(file_path)
        est_chunks = math.ceil(file_size / self.chunk_size) if self.chunk_size > 0 else None

        total_tokens = 0
        with tqdm(total=est_chunks, desc="Counting tokens", unit=" chunk") as pbar:
            for chunk in self.read_in_chunks(file_path):
                total_tokens += len(self.encode(chunk))
                pbar.update(1)
        print(f"Total tokens: {total_tokens}")

        token_memmap = np.memmap(out_path, dtype=dtype, mode='w+', shape=(total_tokens, ))

        print("Writing tokens ...")
        
        offset = 0
        with tqdm(total=est_chunks, desc="Writing tokens", unit=" chunk") as pbar:
            for chunk in self.read_in_chunks(file_path):
                ids = self.encode(chunk)
                token_memmap[offset: offset + len(ids)] = np.array(ids, dtype=dtype)
                offset += len(ids)
                pbar.update(1)
        token_memmap.flush()
        
        print(f"Saved pre-tokenized dataset to {out_path}")
        return out_path

    def decode(self, encoded_ids):
        # contains the inverse of the vocabulary
        # eg: {'0:b'\x00, ... } (unicode: binary)
        inv_vocab = {idx: byte for byte, idx in self.vocab.items()}
        decoded_text = []
        for ids in encoded_ids:
            token = inv_vocab[ids]
            # decode special tokens back to str
            if token.decode(errors="ignore") in self.special_tokens:
                decoded_text.append(token.decode(errors="ignore"))
            else:
                # convert unicode to binary and decode it
                decoded_text.append(token.decode(errors="ignore"))
        return  "".join(decoded_text)
    
    def save(self, file_prefix):
        # save the model and the vocabulary
        model_data = {
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'pattern': self.PAT,
            'special_tokens' : self.special_tokens 
        }
        with open(f"{file_prefix}.model", 'wb') as f:
            pickle.dump(model_data, f)
        # vocabulary needs to be saved as an str not binary
        vocab_str = {str(k): v for k, v in self.vocab.items()}
        with open(f"{file_prefix}.vocab.json", 'w') as f:
            json.dump(vocab_str, f, indent=2)

    @classmethod
    def load(cls, file_prefix):
        # load the model and the vocabulary
        tokenizer = cls.__new__(cls)

        with open(f"{file_prefix}.model", 'rb') as f:
            model_data = pickle.load(f)

        tokenizer.merges = model_data['merges']
        tokenizer.vocab_size = model_data['vocab_size']
        tokenizer.PAT = model_data['pattern']
        tokenizer.special_tokens = model_data['special_tokens']
        tokenizer.chunk_size = 8 * 1024 * 1024
        tokenizer.splits = []
        tokenizer.streaming = False
        tokenizer.input_path = None

        with open(f"{file_prefix}.vocab.json", 'r') as f:
            vocab_str = json.load(f)
        # converting the str back to binary
        tokenizer.vocab = {}
        for k_str, v in vocab_str.items():
            k = eval(k_str)
            tokenizer.vocab[k] = v
        return tokenizer

