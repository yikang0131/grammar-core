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
    {"id": 0, "name": "gender_direction"},
    {"id": 1, "name": "binding_direction"},
    {"id": 2, "name": "context_direction"},
]


output_space = ["A", "B"]
chat = True
multi_token = False


TASK_KWARGS = {
    "interv_configs": interv_configs,
    "output_space": output_space,
    "chat": chat,
    "multi_token": multi_token
}


class Coreference(Task):

    def __init__(self, **kwargs):
        if not kwargs:
            kwargs = TASK_KWARGS
        super().__init__(**kwargs)

    def format_example(self, example):
        example["person1"] = example["person1"].replace("The ", "the ")
        example["person2"] = example["person2"].replace("The ", "the ")
        question = f"Sentence: {example['sentence']}\nQuestion: Who does \"{example['pronoun']}\" refer to?\nA. {example['person1']}\nB. {example['person2']}\nRespond with \"A\" or \"B\"."
        answers = []
        for cue in ["binding", "gender", "context"]:
            if example[f"{cue}_direction"] == "neutral":
                continue
            answers.append(example[f"{cue}_direction"])
        # check there is no conflict in answers
        if len(set(answers)) > 1:
            raise ValueError(f"Conflicting answers: {answers}")
        else:
            answer = answers[0]
        example["question"] = question
        example["answer"] = answer
        return example

    def causal_model(self, gender_direction, binding_direction, context_direction, decision):
        ...

    def generate_data(self, data_path, sample=True, train_test_split=True):
        import random
        import pandas as pd

        df = pd.read_json(data_path, lines=True)
        df.person1 = df.person1.apply(lambda x: x.replace("The ", "the "))
        df.person2 = df.person2.apply(lambda x: x.replace("The ", "the "))

        questions = df.apply(
            lambda row: f"Sentence: {row.sentence}\nQuestion: Who does \"{row.pronoun}\" refer to?\nA. {row.person1}\nB. {row.person2}\nRespond with \"A\" or \"B\".", 
            axis=1
        ).tolist()

        answers = []
        for i, row in df.iterrows():
            answers_ = []
            for cue in ["binding", "gender", "context"]:
                if row[f"{cue}_direction"] == "neutral":
                    continue
                answers_.append(row[f"{cue}_direction"])
            # check there is no conflict in answers
            if len(set(answers_)) > 1:
                raise ValueError(f"Conflicting answers for row {i}: {answers_}")
            else:
                answer = answers_[0]
                answers.append(answer)

        df["question"] = questions
        df["answer"] = answers

        intervention_data = {
            "base_question": [],
            "source_question": [],
            "base_answer": [],
            "source_answer": [],
            "intervention_variables": []
        }

        # Make intervention: binding-based
        data = df[(df.context_direction == "neutral") & (df.gender_direction == "neutral")]
        person1_data = data[data.binding_direction == "person1"]
        person2_data = data[data.binding_direction == "person2"]
        # Make m x n pairs
        for i, (_, row1) in enumerate(person1_data.iterrows()):
            for j, (_, row2) in enumerate(person2_data.iterrows()):
                base_row = random.choice([row1, row2])
                source_row = row2 if base_row.equals(row1) else row1
                intervention_data["base_question"].append(base_row.question)
                intervention_data["source_question"].append(source_row.question)
                intervention_data["base_answer"].append(base_row.answer)
                intervention_data["source_answer"].append(source_row.answer)
                intervention_data["intervention_variables"].append(
                    [self.var2id("binding_direction")]
                )
        if sample:
            binding_based_data = pd.DataFrame(intervention_data).sample(n=1000, random_state=42)
        else:
            binding_based_data = pd.DataFrame(intervention_data)
        # Save to tmp file for checking
        # binding_based_data.to_json("coreference_intervention_binding.jsonl", lines=True, orient="records")

        # Make intervention: context-based
        intervention_data = {
            "base_question": [],
            "source_question": [],
            "base_answer": [],
            "source_answer": [],
            "intervention_variables": []
        }
        data = df[(df.binding_direction == "neutral") & (df.gender_direction == "neutral")]
        person1_data = data[data.context_direction == "person1"]
        person2_data = data[data.context_direction == "person2"]
        # Make m x n pairs
        for i, (_, row1) in enumerate(person1_data.iterrows()):
            for j, (_, row2) in enumerate(person2_data.iterrows()):
                base_row = random.choice([row1, row2])
                source_row = row2 if base_row.equals(row1) else row1
                intervention_data["base_question"].append(base_row.question)
                intervention_data["source_question"].append(source_row.question)
                intervention_data["base_answer"].append(base_row.answer)
                intervention_data["source_answer"].append(source_row.answer)
                intervention_data["intervention_variables"].append(
                    [self.var2id("context_direction")]
                )
        if sample:
            context_based_data = pd.DataFrame(intervention_data).sample(n=1000, random_state=42)
        else:
            context_based_data = pd.DataFrame(intervention_data)
        # context_based_data.to_json("coreference_intervention_context.jsonl", lines=True, orient="records")

        # Make intervention: gender-based
        intervention_data = {
            "base_question": [],
            "source_question": [],
            "base_answer": [],
            "source_answer": [],
            "intervention_variables": []
        }
        data = df[(df.binding_direction == "neutral") & (df.context_direction == "neutral")]
        person1_data = data[data.gender_direction == "person1"]
        person2_data = data[data.gender_direction == "person2"]
        # Make m x n pairs
        for i, (_, row1) in enumerate(person1_data.iterrows()):
            for j, (_, row2) in enumerate(person2_data.iterrows()):
                base_row = random.choice([row1, row2])
                source_row = row2 if base_row.equals(row1) else row1
                intervention_data["base_question"].append(base_row.question)
                intervention_data["source_question"].append(source_row.question)
                intervention_data["base_answer"].append(base_row.answer)
                intervention_data["source_answer"].append(source_row.answer)
                intervention_data["intervention_variables"].append(
                    [self.var2id("gender_direction")]
                )

        if sample:
            gender_based_data = pd.DataFrame(intervention_data).sample(n=1000, random_state=42)
        else:
            gender_based_data = pd.DataFrame(intervention_data)
        # gender_based_data.to_json("coreference_intervention_gender.jsonl", lines=True, orient="records")

        if train_test_split:
            train_binding = binding_based_data.sample(frac=0.8, random_state=42)
            test_binding = binding_based_data.drop(train_binding.index)
            train_context = context_based_data.sample(frac=0.8, random_state=42)
            test_context = context_based_data.drop(train_context.index)
            train_gender = gender_based_data.sample(frac=0.8, random_state=42)
            test_gender = gender_based_data.drop(train_gender.index)

            train_data = pd.concat([train_binding, train_context, train_gender]).sample(frac=1, random_state=42).reset_index(drop=True)
            test_data = pd.concat([test_binding, test_context, test_gender])
            train_data.base_answer = train_data.base_answer.apply(lambda x: "A" if x == "person1" else "B")
            train_data.source_answer = train_data.source_answer.apply(lambda x: "A" if x == "person1" else "B")
            test_data.base_answer = test_data.base_answer.apply(lambda x: "A" if x == "person1" else "B")
            test_data.source_answer = test_data.source_answer.apply(lambda x: "A" if x == "person1" else "B")

            return {"train": train_data, "validation": test_data}
        
        else:
            all_data = pd.concat([binding_based_data, context_based_data, gender_based_data]).sample(frac=1, random_state=42).reset_index(drop=True)
            all_data.base_answer = all_data.base_answer.apply(lambda x: "A" if x == "person1" else "B")
            all_data.source_answer = all_data.source_answer.apply(lambda x: "A" if x == "person1" else "B")
            return all_data