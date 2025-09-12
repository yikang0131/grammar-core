import re
import pandas as pd
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET


fns = glob("bnc/Texts/*/*/*.xml")

all_sentences = []
all_labels = [] # whether the sentence matches the pattern

for fn in tqdm(fns):
    tree = ET.parse(fn)
    root = tree.getroot()
    for s in root.iter("s"):
        # get all children of s
        sentence = "".join([t.text for t in s if t.text])
        # c5_list = []
        # for t in s:
        #     if t.get("c5"):
        #         c5_list.append(t.get("c5"))

        # c5_pattern = " ".join(c5_list)
        # if not re.search(r'NN1(?:\s+(?!V)\w+)*\s+(?:PRF|PRP|CJC|PUN)(?:\s+\w+)*\s+NN1(?:\s+\w+)*\s+V\w+', c5_pattern):
        #     all_labels.append(0)
        # else:
        #     all_labels.append(1)

        all_sentences.append(sentence)

with open("train.txt", "w") as f:
    for sentence in all_sentences:
        f.write(sentence + "\n")


# df = pd.DataFrame({
#     "sentence": all_sentences,
#     "label": all_labels
# })
# df[df.label==1].to_json("filtered_bnc.json", orient="records", lines=True, force_ascii=False)
# df[df.label==0].to_json("filtered_bnc_negative.json", orient="records", lines=True, force_ascii=False)