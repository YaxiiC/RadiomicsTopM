# classification_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class InteractionLogisticRegression(nn.Module):
    def __init__(self, input_size=6156, output_size=3):
        super(InteractionLogisticRegression, self).__init__()
        # Linear layer for original features
        self.linear = nn.Linear(input_size, output_size)
        # Interaction weights for pairwise feature interactions
        self.interaction_weights = nn.Parameter(torch.randn(input_size, input_size))
        nn.init.xavier_uniform_(self.interaction_weights)  # Initialize weights
        # Final layer to combine linear logits and interaction logits
        self.final_layer = nn.Linear(input_size + output_size, output_size)

    def forward(self, x):
        # Linear logits from original features
        linear_logits = self.linear(x)  # Shape: [batch_size, output_size]
        
        # Calculate interaction terms (pairwise interactions)
        interactions = torch.einsum("bi,bj,ij->bij", x, x, self.interaction_weights)
        interactions = interactions.sum(dim=2)  # Reduce to [batch_size, input_size]
        
        # Concatenate linear logits and interaction terms
        combined = torch.cat((linear_logits, interactions), dim=1)  # Shape: [batch_size, input_size + output_size]
        
        # Final layer to produce output logits
        logits = self.final_layer(combined)  # Shape: [batch_size, output_size]
        
        return logits
