import math


class TopKScheduler:
    """
    Scheduler for dynamically adjusting top_k parameter during training.
    Supports various scheduling strategies: linear, exponential, cosine, step-wise.
    """
    
    def __init__(
        self,
        initial_top_k: int,
        final_top_k: int,
        total_steps: int,
        schedule_type: str = "linear",
        warmup_steps: int = 0,
        step_size: int = None,
        gamma: float = 0.1
    ):
        """
        Args:
            initial_top_k: Starting value of top_k
            final_top_k: Final value of top_k
            total_steps: Total training steps
            schedule_type: "linear", "exponential", "cosine", "step"
            warmup_steps: Number of steps to keep initial_top_k (warmup period)
            step_size: For step scheduler, steps between reductions
            gamma: For exponential/step scheduler, decay factor
        """
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.step_size = step_size or (total_steps // 4)  # Default to 4 reductions
        self.gamma = gamma
        
        self.current_step = 0
        self.current_top_k = initial_top_k
        
        # Validate inputs
        assert initial_top_k >= final_top_k, "initial_top_k must be >= final_top_k"
        assert schedule_type in ["linear", "exponential", "cosine", "step"], \
            "schedule_type must be one of: linear, exponential, cosine, step"
    
    def step(self):
        """Update the current step and recalculate top_k"""
        self.current_step += 1
        self.current_top_k = self.get_top_k()
        return self.current_top_k
    
    def get_top_k(self) -> int:
        """Calculate current top_k based on schedule"""
        # Warmup period - keep initial value
        if self.current_step <= self.warmup_steps:
            return self.initial_top_k
        
        # Adjust step for warmup
        adjusted_step = self.current_step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps
        
        if adjusted_step >= adjusted_total:
            return self.final_top_k
        
        # Calculate progress ratio
        progress = adjusted_step / adjusted_total
        
        if self.schedule_type == "linear":
            top_k = self.initial_top_k - (self.initial_top_k - self.final_top_k) * progress
            
        elif self.schedule_type == "exponential":
            # Exponential decay: top_k = initial * (final/initial)^progress
            decay_factor = (self.final_top_k / self.initial_top_k) ** progress
            top_k = self.initial_top_k * decay_factor
            
        elif self.schedule_type == "cosine":
            # Cosine annealing
            top_k = self.final_top_k + (self.initial_top_k - self.final_top_k) * \
                     (1 + math.cos(math.pi * progress)) / 2
                     
        elif self.schedule_type == "step":
            # Step-wise reduction
            num_reductions = adjusted_step // self.step_size
            top_k = self.initial_top_k * (self.gamma ** num_reductions)
            top_k = max(top_k, self.final_top_k)  # Don't go below final_top_k
        
        return max(int(round(top_k)), self.final_top_k)
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'current_step': self.current_step,
            'current_top_k': self.current_top_k
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.current_step = state_dict['current_step']