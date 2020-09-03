import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):

        super().__init__()

        #num of rows in embedding matrix
        num_words = embedding_matrix.shape[0]

        #columns in matrix
        embed_dim = embedding_matrix.shape[1]

        #input embedding layer
        self.embedding = nn.Embedding(num_embeddings= num_words, embedding_dim=embed_dim)

        #embedding matrix used as weights for embedding layer
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        # not training pretrained embedding layer
        self.embedding.weight.requires_grad = False


        #bidirectional LSTM with 128 hidden size
        self.lstm = nn.LSTM(embid_dim, 128, bidirectional=True, batch_first=True)

        self.out = nn.Linear(512, 1)

        def forward(self, x):
            x = self.embedding(x)

            x, _ = self.lstm(x)

            #mean and max pooling to lstm
            avg_pool = torch.mean(x, 1)
            max_pool = torch.max(x, 1)

            #concat max & mean pool  128 +128 512(output)
            out = torch.cat((avg_pool, max_pool), 1)

            out = self.out(out)

            return out
