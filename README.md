
# Assessing the Factuality of System-generated Feedback (Data Augmentation)

Welcome to the Gears 2023 Internship Program! This repository focuses on addressing the challenge of assessing the factuality of system-generated feedback.

## Problem Statement

The main objective of this project is to develop a solution for evaluating the accuracy and factuality of feedback generated by automated systems. One promising approach is to build a "filter" that can automatically discern the factual correctness of generated statements. This filter will ensure that only accurate and pertinent feedback is delivered to students. We intend to leverage Natural Language Inference (NLI) models, such as BART, to perform this filtering task. In NLI, the goal is to determine whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise."

## Tasks

This project comprises two main tasks:

### Task A: Data Augmentation

Task A focuses on data augmentation techniques. It involves enhancing the dataset used for training the NLI model. 

### Task B: Query ChatGPT

Task B is centered around querying ChatGPT.  We work on creating queries and analyzing responses from ChatGPT to further enhance the capabilities of the system.

## Contents
- [Task A](#task-a)
- [Task B](#task-b)
- [TODO List](#todo-list)
- [Authors](#authors)

## Task A

## Task A

**Task details：** Task A aims to increase the diversity of feedback text through multiple text enhancement techniques, thereby improving the quality and diversity of system-generated feedback. In this task, we use a series of text processing techniques, including negation processing, noun replacement, adjective antonym replacement, Unigram Noising, Instance Crossover Augmentation, voice transformation, etc.

**Code example：**

```python
# Negative Processing
if TextTransformation.negate(feedback[i]):
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    new_feedback = TextTransformation.negate(feedback[i])
    feedback_new.append(new_feedback)
    if label[i] == 0:
        label_new.append(1)
    elif label[i] == 1:
        label_new.append(0)

# Noun Swap
if TextTransformation.noun_swap(feedback[i]):
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    new_feedback = TextTransformation.noun_swap(feedback[i])
    feedback_new.append(new_feedback)
    label_new.append(label[i])

# Replace Adjectives
if TextTransformation.replace_adjectives(feedback[i]):
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    new_feedback = TextTransformation.replace_adjectives(feedback[i])
    feedback_new.append(new_feedback)
    if label[i] == 0:
        label_new.append(1)
    elif label[i] == 1:
        label_new.append(0)

# Unigram Noising
if random.random() < 0.5:
    if feedback[i]:
        noised_feedback = TextTransformation.unigram_noising(feedback[i], noise_prob=0.1)
        pid_new.append(pid[i])
        doc_new.append(doc[i])
        feedback_new.append(noised_feedback)
        label_new.append(label[i])

# Instance Crossover Augmentation
if random.random() < 0.5:
    random_index = random.randint(0, l - 1)
    mixed_feedback = TextTransformation.InstanceCrossoverAugmentation(feedback[i], feedback[random_index], mixup_alpha=0.5)
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    feedback_new.append(mixed_feedback)
    mixed_label = (label[i] + label[random_index]) / 2
    if mixed_label < 0.5:
        mixed_label = 0
    else:
        mixed_label = 1
    label_new.append(mixed_label)

# Sentence Voice Transformation
passive_sentence = TextTransformation.convert_to_passive(feedback[i])
if passive_sentence:
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    feedback_new.append(passive_sentence)
    label_new.append(label[i])

active_sentence = TextTransformation.convert_to_active(passive_sentence)
if active_sentence:
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    feedback_new.append(active_sentence)
    label_new.append(label[i])
```
**Progress report：** The goal of Task A is to improve the quality of system-generated feedback by diversifying the feedback text. We have successfully implemented several text enhancement techniques, including negation, noun replacement, adjective antonym replacement, Unigram Noising, Instance Crossover Augmentation, and voice transformation. These techniques have successfully enhanced feedback text and in some cases changed the tone and grammatical structure of the text. This will help generate more diverse feedback and improve the performance of the system.****

       


## Task B

## Task B

In Task B, we utilize OpenAI's GPT-3.5 Turbo model to generate feedback evaluations for student documentation. This process involves the following steps:

1. **API Key Setup**: We start by setting up the OpenAI API key for authentication.

    ```python
    api_key = " sk-********************"
    openai.api_key = api_key
    ```

2. **Feedback Generation Function**: We define a function `get_completion` to generate feedback for a given prompt using the GPT-3.5 Turbo model.

    ```python
    def get_completion(prompt, model="gpt-3.5-turbo"):
        # ...
    ```

3. **Processing Output**: We process the model's response to extract feedback sentences and their corresponding labels (0 or 1) using the `process_output` function.

    ```python
    def process_output(response):
        # ...
    ```

4. **Saving to CSV**: The feedback sentences and labels are saved to a CSV file using the `save_feedback_and_labels_to_csv` function.

    ```python
    def save_feedback_and_labels_to_csv(feedbacks, labels, output_file):
        # ...
    ```

5. **Execution**: Finally, we execute the code by providing a documentation (`doc`) and a sample feedback (`feedback`) for evaluation. The GPT-3.5 Turbo model generates feedback sentences with associated labels. The results are then saved to a CSV file.

    ```python
    doc = """
    # Documentation content goes here...
    """

    feedback = """
    # Sample feedback sentences go here...
    """

    prompt = """
    # Prompt for evaluation...
    """

    response = get_completion(prompt)
    feedbacks, labels = process_output(response)
    output_csv_file = "Your_file_path"
    save_feedback_and_labels_to_csv(feedbacks, labels, output_csv_file)
    ```

This code allows us to leverage AI-driven feedback generation to evaluate the relevance of feedback to student documentation. The generated feedback is saved to a CSV file for further analysis.


## TODO List

- [ ]  Problem of uneven class: Retrofit loss function, like using Focal Loss, DR Loss
- [ ] Syntax tree: Use a defined rule grammar file or train a model
- [ ]  UDA: We can try to only do data expansion without determining the label, and use UDA to label the newly generated data
- [ ] Refinement of Prompt Text
- [ ] Add reasoning steps: Add an inference step before the final answer. Giving the model some extra time and space to think out loud can improve its chances of getting the correct answer.
- [ ] Prepare the project paper

## Authors

- Hongji Li
- Qinjin Jia
- Guanyu Jia
