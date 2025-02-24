import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def display_bar_chart(
        base_accuracy, ptuning_accuracy, lora_accuracy, split):
    plt.figure(figsize=(4, 4))
    _, ax = plt.subplots()
    bars = ['Base', 'P-tuning', 'LoRA']
    values = [
        base_accuracy[split], ptuning_accuracy[split], lora_accuracy[split]]
    ax.bar(bars, values, color=['lightskyblue', 'palegreen', 'sandybrown'])
    ax.set_ylabel('Accuracy')
    ax.set_title(
        f'Accuracy of Medi on {split} split of PubMedQA-labeled dataset')
    plt.show()


def display_table(base_accuracy, ptuning_accuracy, lora_accuracy):
    base_acc = {
        'Train Accuracy': base_accuracy['train'],
        'Test Accuracy': base_accuracy['test']
    }
    ptuning_acc = {
        'Train Accuracy': ptuning_accuracy['train'],
        'Test Accuracy': ptuning_accuracy['test']
    }
    lora_acc = {
        'Train Accuracy': lora_accuracy['train'],
        'Test Accuracy': lora_accuracy['test']
    }
    df = pd.DataFrame(
        [base_acc, ptuning_acc, lora_acc], index=['Base', 'P-tuning', 'LoRA'])
    display(df)


def print_question_and_answers(question, ground_truth_answer, answer=None):
    s = f'Question:\n{question}\n\nGround Truth Answer:\n{ground_truth_answer}'
    if answer is not None:
        s += f'\n\nAnswer:\n{answer}'
    print(s)
