import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplementaryTypeTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder-decoder for type transition
        self.encoder = nn.Linear(config.TYPE_EMB_DIM, config.TYPE_EMB_DIM // 2)
        self.decoder = nn.Linear(config.TYPE_EMB_DIM // 2, config.TYPE_EMB_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        
    def forward(self, query_type_embedding):
        # Encode query type
        h = self.dropout(F.relu(self.encoder(query_type_embedding)))
        # Decode to complementary type space
        complementary_base = self.decoder(h)
        return complementary_base