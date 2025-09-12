from dataclasses import dataclass
from typing import List


@dataclass
class Concept:
    name: str
    id: int
    values: List
    at: str


class ConceptConfig:
    def __init__(self, concepts: List[Concept]):
        self.concepts = concepts

    def __post_init__(self):
        self.concepts[-1].name == "label"

    @property
    def input_concepts(self):
        return ConceptConfig([c for c in self.concepts if c.name != "label"])

    @classmethod
    def from_dict(cls, concept_dict):
        concepts = [Concept(name=k, **v) for k, v in concept_dict.items()]
        return cls(concepts)

    def __getitem__(self, key):
        if isinstance(key, int):
            for concept in self.concepts:
                if concept.id == key:
                    return concept
            raise KeyError(f"No concept with intervention_id {key}")
        elif isinstance(key, str):
            for concept in self.concepts:
                if concept.name == key:
                    return concept
            raise KeyError(f"No concept with name {key}")
        else:
            raise TypeError("Key must be an int (intervention_id) or str (name)")