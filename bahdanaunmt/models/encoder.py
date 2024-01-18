from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_len, hidden_size, dropout_p=0.1) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_vocab_len, hidden_size)
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)

        return output, hidden
