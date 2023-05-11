'''
Dataset:
https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data
'''

import sys, os
import random
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

NP_TYPE = np.int64


class AminoStrawboss():
    def __init__(self):
        """ list of letters in Amino Acid
        """
        self.amino = [  'A', ## Alanine
                        'R', ## Arginine
                        'N', ## Asparagine
                        'D', ## Aspartic_Acid
                        'C', ## Cysteine
                        'E', ## Glutamic_Acid
                        'Q', ## Glutamine
                        'G', ## Glycine
                        'H', ## Histidine
                        'I', ## Isoleucine
                        'L', ## Leucine
                        'K', ## Lysine
                        'M', ## Methionine
                        'F', ## Phenylalanine
                        'P', ## Proline
                        'S', ## Serine
                        'T', ## Threonine
                        'W', ## Tryptophan
                        'Y', ## Tyrosine
                        'V', ## Valine
                    ]

        self.char2idx = {}
        self.idx2char = {}
        self._create_index()

    def _create_index(self):

        # letter to index mapping
        for idx, char in enumerate(self.amino):
            self.char2idx[char] = idx + 7 # +7 token initially

        # adding speacial token at end
        l = len(self.amino)
        self.char2idx['_'] = l+0  #pad
        self.char2idx['$'] = l+1  #start
        self.char2idx['#'] = l+2  #end
        self.char2idx['*'] = l+3  #Mask
        self.char2idx["'"] = l+4  #apostrophe U+0027
        self.char2idx['%'] = l+5  #unused
        self.char2idx['!'] = l+6  #unused

        # index to letter mapping
        for char, idx in self.char2idx.items():
            self.idx2char[idx] = char

    def size(self):
        return len(self.char2idx)


    def protein2npvec(self, word):
        """ Converts given  protein string to vector(numpy)
        Also adds tokens for start and end
        """
        try:
            vec = [self.char2idx['$']] #start token
            for i in list(word):
                vec.append(self.char2idx[i])
            vec.append(self.char2idx['#']) #end token

            vec = np.asarray(vec, dtype=np.int64)
            return vec

        except Exception as error:
            print("Error In word:", word, "Error Char not in Token:", error)
            sys.exit()

    def npvec2protein(self, vector):
        """ Converts vector(numpy) to protein string
        """
        char_list = []
        for i in vector:
            char_list.append(self.idx2char[i])

        word = "".join(char_list).replace('$','').replace('#','') # remove tokens
        word = word.replace("_", "").replace('*','') # remove tokens
        return word


##======== Data Reading ==========================================================

class SeqNESPData(Dataset):
    """
    Enzyme Thermal Stability Dataset Loader
    src: [str] enzyme sequence + (pH)
    tgt: [float] Thermal Stability Number
    """
    def __init__(self, csv_file, mode = 'train',
                    src_amino_obj = AminoStrawboss(),
                    padding = True, max_seq_size = None,
                 ):
        """
        task: 'train' or 'test'
        padding: Set True if Padding with zeros is required for Batching
        max_seq_size: Size for Padding both input and output, Longer words will be truncated
                      If unset computes maximum of source, target seperate
        """
        self.src_amino = src_amino_obj
        __svec = self.src_amino.protein2npvec

        #Load data
        if mode == 'train':
            src_strs, tgt_vals = self._train_csvloader(csv_file)
            self.src = [ __svec(s)  for s in src_strs]
            self.tgt = tgt_vals
            self._getitem = self._train_item

        elif mode == 'infer':
            src_strs, seqid = self._test_csvloader(csv_file)
            self.src = [ __svec(s)  for s in src_strs]
            self.name = seqid
            self._getitem = self._test_item

        else:
            raise 'unknown task type'

        self.padding = padding
        if max_seq_size:
            self.max_src_size = max_seq_size
        else:
            self.max_src_size = max(len(t) for t in self.src)
        print("DataClass MaxSequence Size: ", self.max_src_size)


    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return len(self.src)

    def _train_item(self, index):
        x = self.src[index][:self.max_src_size] #trucate Seq to MaxSize
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_src_size)

        y = self.tgt[index]
        return x,y, x_sz

    def _test_item(self, index):
        x = self.src[index][:self.max_src_size] #trucate Seq to MaxSize
        seqid = self.name[index] #trucate Seq to MaxSize
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_src_size)
        return x, seqid, x_sz


    def _train_csvloader(self, csv_file):
        ''' Convert CSV into required Data
        '''
        datadf = pd.read_csv(csv_file)
        x = []; y = []
        for i in range(len(datadf)):
            x.append(datadf.loc[i,'protein_sequence'])
            y.append(datadf.loc[i,'tm'])
        return x, y

    def _test_csvloader(self, csv_file):
        ''' Convert CSV into required Data
        '''
        datadf = pd.read_csv(csv_file)
        x = []; seqid = []
        for i in range(len(datadf)):
            x.append(datadf.loc[i, 'protein_sequence'])
            seqid.append(datadf.loc[i, 'seq_id'])
        return x, seqid


    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _char_class_weights(self, x_list, scale = 10):
        """For handling class imbalance in the characters
        Return: 1D-tensor will be fed to CEloss weights for error calculation
        """
        from collections import Counter
        full_list = []
        for x in x_list:
            full_list += list(x)
        count_dict = dict(Counter(full_list))

        class_weights = np.ones(self.tgt_amino.size(), dtype = np.float32)
        for k in count_dict:
            class_weights[k] = (1/count_dict[k]) * scale

        return class_weights



if __name__ == "__main__":
    exp_dataset = SeqNESPData(  csv_file='datasets/train_split.csv',
                    src_amino_obj = AminoStrawboss(),
                    padding = True, max_seq_size = 2200)

    for i in range(len(exp_dataset)):
        print(exp_dataset.__getitem__(i))
        if i > 100: break




###============== FootNotes ====================================================
"""
Sequence Lengths
75%ile: 537
90%ile: 851
99%ile: 2108
max:    8798
"""