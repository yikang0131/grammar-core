import json
from src.tasks.task import Task
from src.tasks.coreference import Coreference


TASKS = {
    "Coreference": Coreference,
}


def init_task(task_name: str, **kwargs):
    if task_name in TASKS:
        return TASKS[task_name](**kwargs)
    else:
        raise ValueError(f"Task {task_name} not found in available tasks: {list(TASKS.keys())}")
    

def load_task(load_path: str):
    with open(load_path, "r") as f:
        config = json.load(f)
    task_name = config.get("task_name", None)
    if task_name is None:
        raise ValueError("Task configuration file must contain 'task_name'.")
    return init_task(task_name, **{k: v for k, v in config.items() if k != "task_name"})