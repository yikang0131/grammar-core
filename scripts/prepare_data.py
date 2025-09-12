import spacy
import pyinflect
import inflect
import pandas as pd
from textblob import Word
# from textblob.en import inflect

nlp = spacy.load("en_core_web_sm")
p = inflect.engine()

def prepare_sv_agree_with_distractor(fn):
    df = pd.read_json(fn)

    data = {}

    df = df[df.task.isin(["agr_sv_num_subj-relc", "agr_sv_num_pp"])]
    base_inputs = df.base.apply(lambda x: "".join(x).replace("<|endoftext|>", "<eos>")).tolist()
    source_inputs = df.src.apply(lambda x: "".join(x).replace("<|endoftext|>", "<eos>")).tolist()

    all_sentences = base_inputs + source_inputs
    all_labels = df.base_label.tolist() + df.src_label.tolist()

    all_rows = []
    for sentence, label in zip(all_sentences, all_labels):
        sentence = sentence.replace("<eos>", "")
        doc = nlp(sentence)
        sent = next(doc.sents)
        tokens = [token for token in sent]
        ori_sentence = sentence
        row = {}
        for i, token in enumerate(tokens):
            if i < len(tokens) - 1:
                noun_type = "target"
            else:
                noun_type = "distractor"

            if token.pos_ == "NOUN":
                if token.lemma_ != token.text:
                    noun_number = "plural"
                    plural = token.text
                    singular = token.lemma_
                else:
                    noun_number = "singular"
                    plural = p.plural(token.text)
                    # plural = inflect.pluralize(token.text, pos=inflect.NOUN)
                    # str(Word(token.text).pluralize())
                    singular = token.text

                row[f"{noun_type}_number"] = noun_number
                row[f"{noun_type}_plural"] = plural
                row[f"{noun_type}_singular"] = singular
                row["label"] = label
        row["ori_sentence"] = ori_sentence
        row["sentence_no_distractor"] = " ".join([t.text for t in tokens[:2]])
        all_rows.append(row)
    data = pd.DataFrame(all_rows)
    import re

    def replace_with_boundary(text, old, new):
        pattern = r"\b" + re.escape(old) + r"\b"
        return re.sub(pattern, new, text)


    sentences = []
    for i, row in data.iterrows():
        sentence00 = replace_with_boundary(
            replace_with_boundary(row.ori_sentence, row.target_plural, row.target_singular),
            row.distractor_plural, row.distractor_singular
        )
        sentence01 = replace_with_boundary(
            replace_with_boundary(row.ori_sentence, row.target_plural, row.target_singular),
            row.distractor_singular, row.distractor_plural
        )
        sentence10 = replace_with_boundary(
            replace_with_boundary(row.ori_sentence, row.target_singular, row.target_plural),
            row.distractor_plural, row.distractor_singular
        )
        sentence11 = replace_with_boundary(
            replace_with_boundary(row.ori_sentence, row.target_singular, row.target_plural),
            row.distractor_singular, row.distractor_plural
        )
        sentence0 = replace_with_boundary(row.sentence_no_distractor, row.target_plural, row.target_singular)
        sentence1 = replace_with_boundary(row.sentence_no_distractor, row.target_singular, row.target_plural)
        sentences.append({
            "sentence00": "<eos>" + sentence00,
            "sentence01": "<eos>" + sentence01,
            "sentence10": "<eos>" + sentence10,
            "sentence11": "<eos>" + sentence11,
            # "sentence0": "<eos>" + sentence0,
            # "sentence1": "<eos>" + sentence1,
        })

    sentences = pd.DataFrame(sentences)
    sentences = sentences.drop_duplicates(subset=["sentence00"])
    return sentences


def filter_solve_cases(data, model, tokenizer):
    solved = []
    for i, row in data.iterrows():
        row_corr = []
        for sent_type in ["sentence00", "sentence01", "sentence10", "sentence11"]:
            inputs = tokenizer(row[sent_type], return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = logits[0, -1, 9885] > logits[0, -1, 1490] # 9885: is, 1940: are
            if prediction and sent_type.startswith("sentence0"):
                row_corr.append(1)
            elif (not prediction) and sent_type.startswith("sentence1"):
                row_corr.append(1)
            else:
                row_corr.append(0)
        solved.append(sum(row_corr) == 4)
    data["solved"] = solved
    solved_cases = data[data.solved]
    unsolved_cases = data[~data.solved]
    print(f"Solved cases: {len(solved_cases)}, Unsolved cases: {len(unsolved_cases)}")
    return solved_cases, unsolved_cases


train_set = prepare_sv_agree_with_distractor("data/causalgym/train.json")
dev_set = prepare_sv_agree_with_distractor("data/causalgym/dev.json")
test_set = prepare_sv_agree_with_distractor("data/causalgym/test.json")


from transformers import AutoModelForCausalLM
from src.tokenization import BNCTokenizer

tokenizer = BNCTokenizer.from_pretrained("results/checkpoint-97790/bnc_word2c5.jsonl")
model = AutoModelForCausalLM.from_pretrained("results/checkpoint-97790").to("cuda")

train_solved, trian_unsolved = filter_solve_cases(train_set, model, tokenizer)
dev_solved, dev_unsolved = filter_solve_cases(dev_set, model, tokenizer)
test_solved, test_unsolved = filter_solve_cases(test_set, model, tokenizer)


train_solved.to_json("data/svagree/solved.train.jsonl", orient="records", lines=True, force_ascii=False)
trian_unsolved.to_json("data/svagree/unsolved.train.jsonl", orient="records", lines=True, force_ascii=False)
dev_solved.to_json("data/svagree/solved.dev.jsonl", orient="records", lines=True, force_ascii=False)
dev_unsolved.to_json("data/svagree/unsolved.dev.jsonl", orient="records", lines=True, force_ascii=False)
test_solved.to_json("data/svagree/solved.test.jsonl", orient="records", lines=True, force_ascii=False)
test_unsolved.to_json("data/svagree/unsolved.test.jsonl", orient="records", lines=True, force_ascii=False)