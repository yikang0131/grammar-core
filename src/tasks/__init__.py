from src.tasks.coreference import Coreference


TASKS = {
    "Coreference": Coreference,
}


def init_task(task_name: str, **kwargs):
    if task_name in TASKS:
        return TASKS[task_name](**kwargs)
    else:
        raise ValueError(f"Task {task_name} not found in available tasks: {list(TASKS.keys())}")