import json


class Task:

    def __init__(self, interv_configs, output_space, multi_token=False):
        self.interv_configs = interv_configs
        self.output_space = output_space
        self.multi_token = multi_token
        if self.multi_token:
            raise NotImplementedError("Multi-token tasks are not implemented yet.")
        self.num_variables = len(interv_configs)

    def update_interv_configs(self, intervention_modules):
        # Only update the modules that interventions will be applied to
        if len(intervention_modules) != self.num_variables:
            raise ValueError("The number of intervention modules must match the number of variables.")
        for i, module_name in enumerate(intervention_modules):
            self.interv_configs[i]["interv_at"] = module_name

    def save_task_config(self, save_path):
        config = {
            "task_type": self.__class__.__name__,
            "interv_configs": self.interv_configs,
            "output_space": self.output_space,
            "multi_token": self.multi_token
        }
        with open(save_path, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_task_config(cls, load_path):
        with open(load_path, "r") as f:
            config = json.load(f)
        return cls(
            interv_configs=config["interv_configs"],
            output_space=config["output_space"],
            multi_token=config.get("multi_token")
        )
    
    def generate_data(self, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

