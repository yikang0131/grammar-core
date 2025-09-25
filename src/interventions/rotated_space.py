import torch
import torch.nn as nn


class RotatedSpaceIntervention(nn.Module):
    
    def __init__(self, hidden_size, proj_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_num = proj_num
        
        rotation_matrix = torch.empty(hidden_size, hidden_size)
        # nn.utils.parametrizations.orthogonal()
        nn.init.orthogonal_(rotation_matrix)
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=True)
        
        var_proj = torch.empty(proj_num, hidden_size)
        nn.init.kaiming_uniform_(var_proj)
        self.var_proj = nn.Parameter(var_proj, requires_grad=True)

    def get_projection_weights(self):
        return torch.sigmoid(self.var_proj)
        
    def extract_interpretable_representations(self, hidden_states, intervention_variable, top_k, rotated_back=False):
        """Extract interpretable representations by projecting to the learned basis"""
        rotated_states = torch.matmul(hidden_states, self.rotation_matrix)
        projection_weights = self.get_projection_weights()
        
        # Select top_k dimensions for this variable
        top_indices = self.select_subspace(top_k, intervention_variable)
        
        # Create a mask for the selected dimensions
        mask = torch.zeros_like(projection_weights[intervention_variable])
        mask[top_indices] = 1.0
        
        output = rotated_states * (projection_weights[intervention_variable] * mask)

        if rotated_back:
            output = torch.matmul(output, self.rotation_matrix.T)

        return output

    def select_subspace(self, top_k, var_idx):
        """Select top_k dimensions based on projection weights for variable var_idx"""
        _, top_indices = torch.topk(self.var_proj[var_idx], top_k)
        return top_indices
    
    def forward(self, base, source, intervention_variables, top_k):
        """
        Samples in a batch should have the same intervention positions
        
        Args:
            base: base hidden states [batch, seq, hidden_size]
            source: source hidden states [batch, seq, hidden_size] 
            intervention_variables: list of variable indices to intervene on
            top_k: number of top dimensions to select for intervention
        """
        # Rotate to learned basis
        rotated_base = torch.matmul(base, self.rotation_matrix)
        rotated_source = torch.matmul(source, self.rotation_matrix)
        
        # Start with base representation
        # rotated_base_intervened = rotated_base.clone()
        rotated_base_intervened = torch.zeros_like(rotated_base)
        
        # Apply interventions only on selected top_k dimensions for each variable
        for var_idx in intervention_variables:
            # Select top_k dimensions for this variable
            top_indices = self.select_subspace(top_k, var_idx)
            
            # Create intervention mask - only intervene on top_k dimensions
            intervention_mask = torch.zeros(self.hidden_size, device=base.device)
            intervention_mask[top_indices] = 1.0
            
            # Apply intervention: replace base with source only for selected dimensions
            rotated_base_intervened = (
                rotated_base_intervened * (1 - intervention_mask) +  # Keep non-selected dims from base
                rotated_source * intervention_mask  # Replace selected dims with source
            )
        
        # Rotate back to original basis
        return torch.matmul(rotated_base_intervened, self.rotation_matrix.T)