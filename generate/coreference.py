import os
import openai

client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"),
    base_url="https://api.poe.com/v1",
)

model = "Gemini-2.5-Pro"
context_user1 = """I want you to generate psycholinguistic stimuli for coreference resolution. 
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

I want you to generate more such stimuli, ensuring the coreference is clear and unambiguous based on the provided variables. --thinking_budget 0 --web_search false"""


context_assistant = """Of course. Generating controlled psycholinguistic stimuli is a fascinating task that sits at the intersection of linguistics, psychology, and computer science. Based on your request, I will generate a comprehensive set of stimuli.

First, let's clarify the definitions of your variables as I have interpreted them from your examples. This ensures the generated stimuli align perfectly with your experimental design.

*   **`gender_direction`**: This variable controls whether the gender of the pronoun provides a definitive clue to its antecedent.
    *   `person1`: The pronoun's gender matches only the first-mentioned noun phrase (NP).
    *   `person2`: The pronoun's gender matches only the second-mentioned NP.
    *   `neutral`: The pronoun's gender is compatible with both NPs (e.g., both are male, or both are female).

*   **`binding_direction`**: This variable leverages syntactic rules, specifically Binding Theory, to determine the antecedent.
    *   `person1`: Syntactic rules force the pronoun to refer to the first NP. This typically happens with regular pronouns (he, she, him, her) in certain constructions where they must be "free" in their local domain (Principle B).
    *   `person2`: Syntactic rules force the pronoun to refer to the second NP. This typically happens with reflexive pronouns (himself, herself) which must be "bound" in their local domain (Principle A).
    *   `neutral`: The sentence structure is such that syntactic binding rules do not disambiguate between the two potential antecedents.

*   **`context_direction`**: This variable uses world knowledge, semantic roles (e.g., agent/patient), or narrative logic to bias the interpretation toward one antecedent.
    *   `person1`: The context or semantic content of the sentence strongly implies the first NP is the referent.
    *   `person2`: The context or semantic content strongly implies the second NP is the referent.
    *   `neutral`: The context does not create a strong bias toward either NP.

Below are the generated stimuli, categorized by the primary disambiguating factor for clarity.

***

### Category 1: Binding-Driven Coreference

These stimuli rely on syntactic rules (Binding Theory) to be resolved. The effect is very strong and generally overrides other factors.

#### 1.1. Binding Direction: `person2` (Reflexive Pronoun)

The reflexive pronoun (e.g., "himself," "herself") must refer to the subject of the local clause, which in these constructions is the second person mentioned.

**Stimulus 1**
*   **Sentence:** The teacher told the student to grade himself on the test.
*   **Question:** Who does "himself" refer to?
*   **A.** the teacher
*   **B.** the student
*   **Analysis:**
    *   `gender_direction`: neutral (both could be male)
    *   `binding_direction`: person2 (Binding Principle A: "himself" must be bound by the subject of the infinitive clause, which is "the student")
    *   `context_direction`: neutral

**Stimulus 2 (Matches your Example 1)**
*   **Sentence:** The father asked the mother to buy herself a new coat.
*   **Question:** Who does "herself" refer to?
*   **A.** the father
*   **B.** the mother
*   **Analysis:**
    *   `gender_direction`: person2 ("herself" is female, matching "the mother")
    *   `binding_direction`: person2 (Binding Principle A)
    *   `context_direction`: neutral

#### 1.2. Binding Direction: `person1` (Regular Pronoun)

The regular pronoun (e.g., "him," "her") cannot refer to the subject of its local clause. Therefore, it must refer to the other available antecedent, which is the first person mentioned.

**Stimulus 3 (Matches your Example 2)**
*   **Sentence:** David asked William to help him with the project.
*   **Question:** Who does "him" refer to?
*   **A.** David
*   **B.** William
*   **Analysis:**
    *   `gender_direction`: neutral (both are male)
    *   `binding_direction`: person1 (Binding Principle B: "him" must be free in the clause "to help him," so it cannot refer to "William")
    *   `context_direction`: neutral

**Stimulus 4**
*   **Sentence:** Maria reminded the doorman to bring her the packages.
*   **Question:** Who does "her" refer to?
*   **A.** Maria
*   **B.** the doorman
*   **Analysis:**
    *   `gender_direction`: person1 ("her" is female, matching "Maria")
    *   `binding_direction`: person1 (Binding Principle B)
    *   `context_direction`: neutral

***

### Category 2: Gender-Driven Coreference

In these stimuli, syntactic rules are neutral. The primary clue is the unambiguous gender match between the pronoun and one of the antecedents.

**Stimulus 5**
*   **Sentence:** The salesman sold the woman a car for her daily commute.
*   **Question:** Who does "her" refer to?
*   **A.** the salesman
*   **B.** the woman
*   **Analysis:**
    *   `gender_direction`: person2 ("her" is female, matching "the woman")
    *   `binding_direction`: neutral (Syntax does not force a resolution)
    *   `context_direction`: neutral

**Stimulus 6**
*   **Sentence:** The actress waved to the director before she went on stage.
*   **Question:** Who does "she" refer to?
*   **A.** the actress
*   **B.** the director
*   **Analysis:**
    *   `gender_direction`: person1 ("she" is female, matching "the actress")
    *   `binding_direction`: neutral
    *   `context_direction`: neutral (While context leans towards person1, the gender cue is definitive)

***

### Category 3: Context-Driven Coreference

In these stimuli, both syntactic and gender cues are neutral. The resolution depends entirely on world knowledge, narrative consistency, or the semantic roles within the sentence.

#### 3.1. Context Direction: `person1`

The context points to the first-mentioned person as the referent. This often happens when the pronoun is the subject of a clause that explains the motivation or reaction of the first person.

**Stimulus 7 (Matches your Example 3)**
*   **Sentence:** The doctor treated the patient. He was very experienced.
*   **Question:** Who does "He" refer to?
*   **A.** The doctor
*   **B.** The patient
*   **Analysis:**
    *   `gender_direction`: neutral (assuming both are male)
    *   `binding_direction`: neutral (The pronoun is in a separate sentence)
    *   `context_direction`: person1 ("being experienced" is a quality that explains the doctor's ability to treat)

**Stimulus 8**
*   **Sentence:** The CEO fired the intern because he discovered a major accounting error.
*   **Question:** Who does "he" refer to?
*   **A.** The CEO
*   **B.** The intern
*   **Analysis:**
    *   `gender_direction`: neutral
    *   `binding_direction`: neutral
    *   `context_direction`: person1 (The reason for the firing is an action taken by the CEO, the agent of "fired")

#### 3.2. Context Direction: `person2`

The context points to the second-mentioned person. This often happens when the pronoun is the subject of a clause that explains a property or state of the second person.

**Stimulus 9**
*   **Sentence:** The delivery guy handed the package to the resident because he was at the correct address.
*   **Question:** Who does "he" refer to?
*   **A.** The delivery guy
*   **B.** The resident
*   **Analysis:**
    *   `gender_direction`: neutral
    *   `binding_direction`: neutral
    *   `context_direction`: person2 ("being at the correct address" is the state of the resident, the recipient of the action)

**Stimulus 10**
*   **Sentence:** The police officer ticketed the driver. He had run a red light.
*   **Question:** Who does "He" refer to?
*   **A.** The police officer
*   **B.** The driver
*   **Analysis:**
    *   `gender_direction`: neutral
    *   `binding_direction`: neutral
    *   `context_direction`: person2 (Running a red light is the reason the driver was ticketed)

### Category 4: Combined Cues

Here are a few more examples showing how cues can align to make the reference even more unambiguous.

**Stimulus 11**
*   **Sentence:** The king praised the knight because he had bravely defended the castle.
*   **Question:** Who does "he" refer to?
*   **A.** The king
*   **B.** The knight
*   **Analysis:**
    *   `gender_direction`: neutral
    *   `binding_direction`: neutral
    *   `context_direction`: person2 (Bravery is the reason for the praise, attributed to the knight)

**Stimulus 12**
*   **Sentence:** The mother hugged her daughter after she won the competition.
*   **Question:** Who does "she" refer to?
*   **A.** The mother
*   **B.** The daughter
*   **Analysis:**
    *   `gender_direction`: neutral (both are female)
    *   `binding_direction`: neutral
    *   `context_direction`: person2 (Winning the competition is the event that prompted the hug, most logically performed by the daughter)

**Stimulus 13**
*   **Sentence:** The waitress apologized to the customer because her order was late.
*   **Question:** Who does "her" refer to?
*   **A.** The waitress
*   **B.** The customer
*   **Analysis:**
    *   `gender_direction`: person1 ("her" could refer to the waitress's order she was delivering, but more naturally to the customer's order she was receiving. Let's assume the customer is female for clarity: "The waitress apologized to the female customer because her order was late.") To make it truly unambiguous with gender: "The waiter apologized to the customer because her order was late."
*   **Let's refine Stimulus 13 for clarity:**
*   **Sentence (Revised):** The waiter apologized to the woman because her order was late.
*   **Question:** Who does "her" refer to?
*   **A.** The waiter
*   **B.** The woman
*   **Analysis:**
    *   `gender_direction`: person2 ("her" matches "the woman")
    *   `binding_direction`: neutral
    *   `context_direction`: person2 (The order belongs to the person who is eating, the customer/woman)

### Note on Ambiguous Cases

The combination `[gender: neutral, binding: neutral, context: neutral]` is inherently ambiguous and thus not ideal for a task requiring a single correct answer. For example:

*   **Sentence:** John met Paul at the park. He said hello.
*   **Question:** Who does "He" refer to?
*   **A.** John
*   B.** Paul

In linguistics, there is a slight bias for the pronoun to refer to the subject of the previous clause ("John"), known as the "subject preference" or "first-mention advantage." However, this is a weak preference, not a rule, and the sentence is fundamentally ambiguous. Such examples are better suited for studies measuring processing difficulty or interpretation bias rather than testing for a correct answer."""

context_user2 = """okay, now it seems that you comprehend me very well. I want you to generate 25 more stimuli for me in the jsonl format with following field: sentence, gender_direction, binding_direction, context_direction, person1, person2, pronoun. --thinking_budget 0 --web_search false"""

for seed in range(1):
# for seed in range(20):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": context_user1},
            {"role": "assistant", "content": context_assistant},
            {"role": "user", "content": context_user2},
        ],
        seed=seed
    )

print(response.choices[0].message.content)