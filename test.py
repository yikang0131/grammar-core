# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# import torch
# from transformers import AutoTokenizer
# from src.models import load_intervenable_model


# import pandas as pd
# from tqdm import tqdm

# df = pd.read_json("data/coreference.jsonl", lines=True)
# df.person1 = df.person1.apply(lambda x: x.replace("The ", "the "))
# df.person2 = df.person2.apply(lambda x: x.replace("The ", "the "))

# questions = df.apply(
#     lambda row: f"Sentence: {row.sentence}\nQuestion: Who does \"{row.pronoun}\" refer to?\nA. {row.person1}\nB. {row.person2}\nRespond with \"A\" or \"B\".", 
#     axis=1
# ).tolist()

# print(questions[0])

# model_path = "/data/ykliu/models/qwen3-8b"
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# formatted_questions = []
# for question in questions:
#     formatted_questions.append(
#         tokenizer.apply_chat_template(
#             conversation=[{"role": "user", "content": question}], 
#             add_generation_prompt=True,
#             tokenize=False,
#             enable_thinking=False
#         )
#     )

# print(formatted_questions[0])

# tokenizer.padding_side = "left"
# model = load_intervenable_model(model_path, device_map="auto")
# print(type(model))

# model_predictions = []
# A_token_id = tokenizer.convert_tokens_to_ids("A")
# B_token_id = tokenizer.convert_tokens_to_ids("B")

# for batch in tqdm(range(0, len(formatted_questions), 8)):
#     batch_questions = formatted_questions[batch: batch + 8]
#     inputs = tokenizer(batch_questions, return_tensors="pt", padding=True)
#     inputs.to(model.device)
#     with torch.no_grad():
#         logits = model.intervenable_forward(**inputs).logits[:, -1, :]
#         # logits = model(**inputs).logits[:, -1, :]

#     for logit in logits:
#         if logit[A_token_id] > logit[B_token_id]:
#             model_predictions.append("person1")
#         else:
#             model_predictions.append("person2")

# acc, total = 0, 0
# for i, row in df.iterrows():
#     answers = []
#     for cue in ["binding", "gender", "context"]:
#         if row[f"{cue}_direction"] == "neutral":
#             continue
#         answers.append(row[f"{cue}_direction"])
#     # check there is no conflict in answers
#     if len(set(answers)) > 1:
#         print("Conflict in answers:", answers)
#         continue
#     else:
#         answer = answers[0]
    
#     if model_predictions[i] == answer:
#         acc += 1
#     total += 1

# print("Accuracy:", acc / total)

from src.tasks.coreference import Coreference

coref_task = Coreference()
# coref_task.generate_data("data/coreference.jsonl", ambiguous=False, train_test_split=False)
coref_task.update_interv_configs(
    ["model.layers[10]", "model.layers[10]", "model.layers[10]"]
)
coref_task.save_task_config("coreference_task_config.json")