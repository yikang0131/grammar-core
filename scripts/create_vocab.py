import json
from glob import glob
from tqdm import tqdm
from collections import Counter
import xml.etree.ElementTree as ET


fns = glob("bnc/Texts/*/*/*.xml")

word2c5 = {}

for fn in tqdm(fns):
    tree = ET.parse(fn)
    root = tree.getroot()
    for s in root.iter("s"):

        for t in s:
            if t.text:
                word = t.text.strip().lower()
                if not word in word2c5:
                    word2c5[word] = []

                c5 = t.get("c5")
                if c5:
                    word2c5[word].append(str(c5))

word2c5 = {k: Counter(v) for k, v in word2c5.items()}
with open("bnc_word2c5.json", "w") as f:
    json.dump(word2c5, f, indent=4)
print("Vocabulary created and saved to bnc_word2c5.json")