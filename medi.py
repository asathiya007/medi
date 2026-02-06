import shutil
import os
import torch
from tqdm import tqdm
from copy import deepcopy
from datasets import Dataset, load_dataset
import transformers
from transformers import TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoTokenizer, \
    AutoModelForCausalLM
from peft import PeftModel, PeftConfig, PromptEncoderConfig, \
    get_peft_model, LoraConfig
import logging
import re


# set environment variables and seed
os.environ['WANDB_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
SEED = 0


class Medi:
    def __init__(self, output_dir, device=None):
        self.logger = logging.getLogger('Medi_Logger')
        self.logger.setLevel(logging.INFO)
        self.output_dir = output_dir
        self.system_context = 'You are a helpful assistant that answers ' \
            + 'medical research questions, based on the given context. ' \
            + "Your answer must be one word: 'Yes', 'No', or 'Maybe'."
        self.model_name = 'microsoft/Phi-3.5-mini-instruct'
        self.assistant_token = '<|assistant|>'
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def get_chat_template_msgs(self, examples):
        input_chats = list(map(
            lambda i: [
                {'role': 'system', 'content': self.system_context},
                {'role': 'user', 'content': examples['question'][i]}
            ], range(len(examples['question']))))
        output_chats = list(map(
            lambda i: input_chats[i] + [
                {'role': 'assistant', 'content': examples['answer'][i]}
            ], range(len(examples['question']))))
        input_chat_msgs = self.tokenizer.apply_chat_template(
            input_chats, tokenize=False, add_generation_prompt=True)
        output_chat_msgs = self.tokenizer.apply_chat_template(
            output_chats, tokenize=False, add_generation_prompt=False)
        return input_chat_msgs, output_chat_msgs

    def get_max_tok_lens(self, batch_size):
        max_input_tok_len = 0
        max_output_tok_len = 0
        for examples in self.dataset.iter(batch_size=batch_size):
            input_chat_msgs, output_chat_msgs = self.get_chat_template_msgs(
                examples)
            max_input_tok_len = max(
                len(self.tokenizer(
                    input_chat_msgs, return_tensors='pt',
                    padding='longest')['input_ids'][0]),
                max_input_tok_len)
            max_output_tok_len = max(
                len(self.tokenizer(
                    output_chat_msgs, return_tensors='pt',
                    padding='longest')['input_ids'][0]),
                max_output_tok_len)
        self.max_input_tok_len = max_input_tok_len
        self.max_output_tok_len = max_output_tok_len

    def tokenize_input_and_output(self, examples):
        input_chat_msgs, output_chat_msgs = self.get_chat_template_msgs(
            examples)
        if self.pad_training_data:
            tok_inputs = self.tokenizer(
                input_chat_msgs, return_tensors='pt',
                max_length=self.max_input_tok_len,
                padding='max_length')
            tok_input_ids = tok_inputs['input_ids']
            tok_input_attn_mask = tok_inputs['attention_mask']
            tok_outputs = self.tokenizer(
                output_chat_msgs, return_tensors='pt',
                max_length=self.max_output_tok_len,
                padding='max_length')['input_ids']
        else:
            tok_inputs = self.tokenizer(
                input_chat_msgs, return_tensors='pt')
            tok_input_ids = tok_inputs['input_ids']
            tok_input_attn_mask = tok_inputs['attention_mask']
            tok_outputs = self.tokenizer(
                output_chat_msgs, return_tensors='pt')['input_ids']
        return {
            'input_ids': tok_input_ids,
            'attention_mask': tok_input_attn_mask,
            'labels': tok_outputs
        }

    def dataset_generator(self):
        dataset = load_dataset(
            'qiaojin/PubMedQA', 'pqa_labeled', split='train')
        for example in dataset:
            context = example['context']['contexts']
            context_labels = example['context']['labels']
            context_strs = list(map(
                lambda i: f'{context_labels[i].upper()}: {context[i]}.',
                range(len(context))))
            context_strs.append(f'QUESTION: {example["question"]}')
            question = '\n'.join(context_strs)
            answer = (example['final_decision'][0].upper() +
                      example['final_decision'][1:]).strip()
            yield {
                'question': question,
                'answer': answer
            }

    def get_dataset(self, batch_size, test_size):
        self.dataset = Dataset.from_generator(self.dataset_generator)
        self.get_max_tok_lens(batch_size=batch_size)
        # not shuffling dataset so split is the same (fair comparison of base
        # model performance versus parameter-efficient fine-tuned model
        # performance)
        self.dataset = self.dataset.train_test_split(
            test_size=test_size, shuffle=False)
        self.logger.info(
            'Obtained and processed PubMedQA-labeled dataset for training')

    def get_base_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.logger.info(
            f'Obtained base model ({self.model_name}) and tokenizer')

    def train(self, peft_type=None, num_epochs=5, test_size=0.2,
              batch_size=4, grad_accumulation_steps=1):
        transformers.enable_full_determinism(SEED)

        # remove previous output directory (new one will be created when saving
        # PEFT weights and tokenizer)
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

        # free up GPU memory
        torch.cuda.empty_cache()

        # get model, tokenizer, and dataset
        self.pad_training_data = batch_size > 1
        self.get_base_model_and_tokenizer()
        self.get_dataset(batch_size=batch_size, test_size=test_size)
        self.tokenized_dataset = deepcopy(self.dataset)
        for split in self.tokenized_dataset.keys():
            self.tokenized_dataset[split] = self.tokenized_dataset[split].map(
                self.tokenize_input_and_output,
                remove_columns=['question', 'answer'],
                batched=True, batch_size=batch_size)

        # apply PEFT config
        if peft_type == 'P_TUNING':
            peft_config = PromptEncoderConfig(
                peft_type=peft_type,
                task_type='CAUSAL_LM',
                num_virtual_tokens=8,
                token_dim=3072,
                encoder_reparameterization_type='MLP',
                encoder_hidden_size=256)
        elif peft_type == 'LORA':
            peft_config = LoraConfig(
                r=14,
                lora_alpha=2,
                init_lora_weights='gaussian',
                target_modules=['qkv_proj', 'down_proj'],
                task_type='CAUSAL_LM')
        else:
            raise Exception(
                'Invalid PEFT type. Expected one of: P_TUNING, LORA')
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # train and save model
        self.logger.info(f'Fine-tuning model (PEFT type: {peft_type})...')
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            seed=SEED,
            # training requires more GPU memory than evaluation (for storing
            # gradients and optimizer states), so a smaller batch size with 
            # gradient accumulation is used for training and a larger batch size
            # is used for evaluation
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * grad_accumulation_steps,
            gradient_accumulation_steps=grad_accumulation_steps,
            num_train_epochs=num_epochs,
            eval_strategy='epoch',
            save_strategy='epoch',
            # only save the latest checkpoint, to save space on disk
            save_total_limit=1,
            use_cpu=(self.device == 'cpu'),
            # returns only the loss in training, not logits and labels
            # (saves memory)
            prediction_loss_only=True
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            data_collator=data_collator
        )
        trainer.train()

        # save PEFT weights, remove checkpoint directories with redundant files
        # (to save space on disk)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.logger.info(
            'Saved PEFT weights and tokenizer to '
            + f'{os.path.abspath(self.output_dir)}')
        regex = re.compile(r'^checkpoint-.*')
        for dir_name in os.listdir(self.output_dir):
            dir_path = os.path.join(self.output_dir, dir_name)
            if os.path.isdir(dir_path) and regex.match(dir_name):
                shutil.rmtree(dir_path)

        # if trained model has PEFT type LORA, merge weights for faster
        # inference
        if peft_type == 'LORA':
            self.model = self.model.merge_and_unload()

    def inference(self, questions):
        transformers.enable_full_determinism(SEED)
        input_chats = list(map(
            lambda q: [
                {'role': 'system', 'content': self.system_context},
                {'role': 'user', 'content': q}
            ], questions))
        prompts = self.tokenizer.apply_chat_template(
            input_chats, tokenize=False, add_generation_prompt=True)
        tokenized_messages = self.tokenizer(
            prompts, return_tensors='pt', padding='longest')
        outputs = self.model.generate(
            input_ids=tokenized_messages['input_ids'].to(self.model.device),
            attention_mask=tokenized_messages['attention_mask'].to(
                self.model.device),
            max_new_tokens=1)
        outputs = self.tokenizer.batch_decode(outputs.to('cpu'))

        # extract responses
        responses = []
        for output in outputs:
            start_idx = output.rfind(self.assistant_token) + len(
                self.assistant_token)
            response = output[start_idx:].strip()
            responses.append(response)
        return responses

    def load(self):
        peft_config = PeftConfig.from_pretrained(self.output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(model, self.output_dir)
        self.model = self.model.to(self.device)
        self.logger.info(
            'Loaded PEFT weights and tokenizer from '
            + f'{os.path.abspath(self.output_dir)}')

        # if trained model has PEFT type LORA, merge weights for faster
        # inference
        if peft_config.peft_type == 'LORA':
            self.model = self.model.merge_and_unload()

    def evaluate(self, batch_size=4, test_size=0.2, use_base_model=False):
        # free up GPU memory
        torch.cuda.empty_cache()

        # load model, tokenizer, and dataset
        self.accuracy = dict()
        if use_base_model:
            self.get_base_model_and_tokenizer()
        elif not hasattr(self, 'model'):
            self.load()
        if not hasattr(self, 'dataset'):
            self.get_dataset(batch_size=batch_size, test_size=test_size)

        # evaluate on each split of the dataset
        for split in self.dataset.keys():
            correct_ct = 0
            self.logger.info(f'Evaluating on {split} split of dataset...')
            for i in tqdm(range(0, len(self.dataset[split]), batch_size)):
                examples = self.dataset[split][i:i+batch_size]
                questions = examples['question']
                labels = list(map(lambda x: x.lower(), examples['answer']))
                responses = self.inference(questions)
                preds = list(map(lambda x: x.lower(), responses))
                for i in range(len(labels)):
                    label = labels[i]
                    pred = preds[i]
                    if label == pred:
                        correct_ct += 1
            self.accuracy[split] = correct_ct / len(self.dataset[split])
            self.logger.info(
                f'Accuracy on {split} split of dataset: '
                + f'{self.accuracy[split]}')
