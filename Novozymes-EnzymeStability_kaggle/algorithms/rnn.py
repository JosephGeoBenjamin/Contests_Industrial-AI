import torch
import torch.nn as nn
from torchsummary import summary
from utilities.runUtils import START_SEED


class RNNEncoder(nn.Module):
    '''
    Simple RNN based encoder network
    '''
    def __init__(self, input_dim, embed_dim, hidden_dim ,
                       rnn_type = 'gru', enc_layers = 1,
                       bidirectional = True,
                       dropout = 0 ):
        """
        embed_dim: [int] Dimension of learnable embedding
                TODO:if 0 One-Hot encoding (non-trainable) will be created
        """
        super(RNNEncoder, self).__init__()
        START_SEED()

        self.input_dim = input_dim #src_vocab_sz
        self.enc_embed_dim = embed_dim
        self.enc_hidden_dim = hidden_dim
        self.enc_rnn_type = rnn_type
        self.enc_layers = enc_layers
        self.enc_directions = 2 if bidirectional else 1

        ## Create Embedding
        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)
        ## Embedding initialise to OHE
        # self.embedding.weight.data.copy_(torch.eye(self.input_dim, self.enc_embed_dim))

        if self.enc_rnn_type == "gru": #TODO: Foward Support for GRU
            self.enc_rnn = nn.GRU(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional)
        elif self.enc_rnn_type == "lstm":
            self.enc_rnn = nn.LSTM(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional)
        else:
            raise Exception("unknown RNN type mentioned")

    def forward(self, x, x_sz):
        '''
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)
        '''
        batch_sz = x.shape[0]
        # x: batch_size, max_length, enc_embed_dim
        x = self.embedding(x)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, enc_embed_dim --> hidden from all timesteps
        # hidden: n_layer**num_directions, batch_size, hidden_dim | if LSTM (h_n, c_n)
        output, (h_n, c_n) = self.enc_rnn(x)

        ### Unpack pack_pad
        # ## pad the sequence to the max length in the batch
        # # output: max_length, batch_size, enc_emb_dim*directions)
        # output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # # output: batch_size, max_length, hidden_dim
        # output = output.permute(1,0,2)

        hid = h_n.permute(1,0,2)
        # hid: batch_size, enc_emb_dim*directions*layers
        hid = hid.reshape(batch_sz, -1)
        return hid


if __name__ == "__main__":

    model = RNNEncoder(input_dim = 26, embed_dim= 256, hidden_dim= 512,
                       rnn_type = 'lstm', enc_layers = 2, bidirectional = True,
                       dropout = 0,)
    print(model)
    # summary(model, (2, 256, 50, 50), depth=2 ) ## TODO: Debug Summary writer