from src.tasks.task import Task

"""
I want you to generate psycholinguistic stimuli for the coreference resolutiont task. 
I have 3 variables: 

1. gender_direction [person1, person2, neutral]
2. binding_direction [person1, person2, neutral]
3. context_direction [person1, person2, neutral]

Here are some examples of the stimuli:

# Example 1
Sentence: The brother asks the sister to take care of herself.
Question: Who does "herself" refer to? 
A. the brother 
B. the sister

- gender_direction = person2
- binding_direction = person2
- context_direction = neutral

# Example 2
Sentence: Tom asks Jerry to take care of him.
Question: Who does "him" refer to?
A. Tom
B. Jerry

- gender_direction = neutral
- binding_direction = person1
- context_direction = neutral

# Example 3
Sentence: Tom bullies Terry. The teacher wants to punish him.
Question: Who does "him" refer to?
A. Tom
B. Terry

- gender_direction = neutral
- binding_direction = neutral
- context_direction = person1

I want you to generate more such stimuli, ensuring the coreference is clear and unambiguous based on the provided variables.
"""

interv_configs = [
    {"name": "person1_gender"},
    {"name": "person2_gender"},
    {"name": "gender_direction"},
    {"name": "binding_direction"},
    {"name": "context_direction"},
]

output_space = ["A", "B"]
multi_token = False

TASK_KWARGS = {
    "interv_configs": interv_configs,
    "output_space": output_space,
    "multi_token": multi_token
}


class Coreference(Task):

    def __init__(self):
        super().__init__(**TASK_KWARGS)