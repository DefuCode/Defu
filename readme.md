# Defu: Optimizing Semantic Code Clone Detection through Execution Path Decomposition and Embedding Fusion

## Task Definition

Given one code pair as the input, the task is to do binary classification (0/1), where 1 stands for clone code and 0 for non-clone code. Models are evaluated by F1 score.

## Directory Structure

- src: source code of Defu and scripts to run experiments

- dataset: our dataset

- models: the pre-trained CodeBERT model and the pre-trained GraphCodeBERT model, both of which can be obtained from hugging face
- parserTool: the tool of Tree-sitter

## Prepare Requirements

- Python: 3.8.18
- Pytorch: 2.1.2
- networkx: 3.1
- tree-sitter: 0.20.2

## Tree-sitter (optional)

Before running the program, if the built file "parserTool/my-languages.so" doesn't work for you, please rebuild as the following command and modify the file path in the parse.py file:

```shell
cd parserTool
bash build.sh
cd ..
```

## Dataset

In our study, we carry out experiments on two primary datasets: BigCloneBench (BCB) and Google Code Jam (GCJ) . 

The BCB dataset is a well-known and extensive code clone benchmark that includes more than eight million labeled clone pairs derived from 25,000 different systems. We choose the BCB dataset because its function-level code granularity matches the detection granularity of Defu. Importantly, the clone pairs within the BCB dataset are categorized into various types, facilitating our evaluation across different clone types. Given that the BCB dataset includes only 278,838 non-clone pairs, we randomly sampled 270,000 pairs each from the clone and non-clone sets to maintain dataset balance. Our selection from the clone set comprises 48,116 Type-1 (T1) clone pairs, 4,234 Type-2 (T2) clone pairs, 21,395 ST3 clone pairs, 86,341 MT3 clone pairs, and 109,914 WT3/T4 clone pairs.We divided all clone pairs and non-clone pairs into training, evaluation, and test subsets with an 8:1:1 ratio. Subsequently, we synthesized the subsets into the training set, evaluation set, and test set, respectively.

The second dataset, known as GCJ, includes 1,669 projects obtained from an online programming competition organized by Google. These projects provide solutions to 12 distinct competition problems and are created by various programmers. Although projects solving the same problem may show syntactical differences, they possess semantic similarities and are therefore treated as clone pairs. In contrast, projects solving different problems are considered dissimilar and categorized as non-clone pairs. Our dataset contains 275,570 semantic clone pairs and 1,116,376 non-clone pairs. To maintain balance within the dataset, we randomly select 270,000 pairs from the clone set and an equal number from the non-clone set.
For the GCJ database, the treatment is the same as for the BCB dataset.

### Data Format

train.jsonl/valid.jsonl/test.jsonl are stored in jsonlines format. Each line in the uncompressed file represents one function. One row is illustrated below.

- **idx1:** index of the function 1
- **idx2:** index of the function 2
- **label:** the label of the pair

## Train

```shell
python run_loss.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_train --train_data_file=datasets/BCB/train.jsonl --eval_data_file=datasets/BCB/eval.jsonl --test_data_file=datasets/BCB/test.jsonl --epoch 2 --block_size 512 --train_batch_size 36 --eval_batch_size 36 --learning_rate 1e-4 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 --pkl_file=datasets/BCB/preprocess/path_embeddings_3_v2.pk
```



## Evaluation 

```shell
python run_loss.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/train.jsonl --eval_data_file=datasets/BCB/eval.jsonl --test_data_file=datasets/BCB/test.jsonl --epoch 2 --block_size 512 --train_batch_size 36 --eval_batch_size 36 --learning_rate 1e-4 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 --pkl_file=datasets/BCB/preprocess/path_embeddings_3_v2.pk
```

