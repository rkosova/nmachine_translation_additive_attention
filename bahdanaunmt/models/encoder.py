from torch import nn

class Encoder(nn.Module):
    """Simple RNN encoder with learned embedding and GRU cells
    """
    def __init__(self, input_vocab_len, hidden_size, dropout_p=0.1) -> None:
        """
        Args:
            input_vocab_len (int): Length of the input language vocabulary
            hidden_size (int): Size of the encoder hidden states
            drouput_p (float) = 0.1: Dropout percentage for embedding layer
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_vocab_len, hidden_size)
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sentence
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)

        return output, hidden
