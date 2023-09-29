# Teaching Large Language Models (LLMs) Predicate Logic

My master thesis project is about teaching Large Language Models to reason in first-order (predicate) logic.


Abstract

Large language models (LLMs), such as GPT-3 and more recently Llama-2, have achieved ever more impressive results on many natural language understanding tasks. Benchmarks such as BIG Bench hard needed to exclude certain tasks, because LLMs have managed to perform so well on them. Finding ever more challenging reasoning tasks for LLMs has been of much interest. On the other hand, LLMs still make silly reasoning mistakes or hallucinate, that is, make false claims as if they were true. Alleviating these mistakes and halluci- nations has equally been of much interest. That is why, in this thesis, we aim to teach LLMs to emulate reasoning in first-order (predicate) logic.
In order to address this challenge, we design and build a first-of-its-kind synthetic dataset that we call the "Synthetic Predicate Logic Corpus" (or SPLC), which includes three tasks in reasoning using both natural language and the artificial language of predicate logic. By making use of a model checker, we can automatically generate the labels; and by building (or modifying) semantic parsers, we can map between natural language and the language of logic. Besides automatic labeling, the big advantage of our dataset is that we can also adjust its difficulty. We produce a baseline that we compare our models’ performance to.
In over 150 experiments, we show the first empirical demonstration that the Falcon, Llama, Orca, and Wizard LLMs can emulate logical reasoning in first-order logic when using LoRA adapters. We find that they are only able to generalize to more difficult tasks to a small extent, although scaling is not robust.


The general pipeline is:
(1) Dataset creation
(2) Possibly: Finetuning 
(3) Evaluation
(4) Comparison to Baseline

Individual files:
- Parser.py: is used in evaluation to parse model outputs
- evaluate_tasks.py: is used in (2) evaluation to postprocess, parse and evaluate parsed outputs
- Predicate_Logic_Dataset.py: is used in (1) dataset creation

Folders:
- all files in finetuning: perform Supervised finetuning of the LLMs on our 3 tasks
- all files in the evaluation folder: evaluate LLMs via zeroshot learning, fewshot learning (with 1, 2 or 4 examples), and after finetuning

Notebooks:
- Baseline: those perform experiments to create a random baseline to compare the LLM performance to (used in (4))
- Dataset creation: those create the datasets / tasks for the LLMs in predicate logic (used in (1))
- Descriptive Statistics: performs descriptive statistics of the datasets (used in (3))
- Fewshot evaluation: evaluates the accuracy for fewshot learning (used in (3))
- Fewshot creation: dataset creation for fewshot learning (used in (1))
- Qualitative Analyses: looks at 30 random datapoints to judge the quality of the outputs (used in (3))
- Quantitative Analyses: looks at which variables from the dataset make the tasks are harder (used in (3))
- Test Parser: tests the parser
- Training evaluation: evaluates the accuracy after training / finetuning (used in (3))
- TrainTestSetCreation: build datasets for finetuning (used in (2))
- Zeroshot evaluation: evaluates the accuracy for zeroshot learning (used in (3))


Please find a more thorough description of the methodology, the hyperparameter settings, the results of the performed experiments as well as possibilities for future work in my thesis. Note that the finetuning and generation was performed on Nvidia A100 80GB GPUs, thanks to BW HPC for their support.