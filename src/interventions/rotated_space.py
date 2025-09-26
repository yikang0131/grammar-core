import torch
import torch.nn as nn


class RotatedSpaceIntervention(nn.Module):
    
    def __init__(self, hidden_size, proj_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_num = proj_num
        
        rotation_matrix = torch.empty(hidden_size, hidden_size)
        nn.init.orthogonal_(rotation_matrix)
        # nn.utils.parametrizations.orthogonal(rotation_matrix)
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=True)
        
        var_proj = torch.empty(proj_num, hidden_size)
        nn.init.kaiming_normal_(var_proj)
        self.var_proj = nn.Parameter(var_proj, requires_grad=True)

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