# Medi

Medi is a system powered by the Phi-3.5-mini-instruct language model that answers medical research questions. The underlying language model of Medi is parameter-efficient fine-tuned and evaluated on the labeled split of the PubMedQA dataset.

The specific parameter-efficient fine-tuning (PEFT) techniques used in this project are p-tuning and LoRA. In both techniques, a relatively small number of auxiliary parameters are trained and used alongside the base model. With these PEFT techniques, language models can be customized to specific use cases with relatively less time and computational resources (compared to fine-tuning the base model by updating its weights directly). 

As shown in the demo notebook, training a relatively small number of extra parameters can noticeably improve the accuracy on a particular task (in this case, answering medical research questions). 

To see the source code of this project, see the `medi.py` file. To see a demo, see the `demo.ipynb` notebook.
