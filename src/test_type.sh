echo "codebert 1"
python run_xglue.py --output_dir=./saved_models_2 --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_1_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_source.pkl
echo "codebert 2"
python run_xglue.py --output_dir=./saved_models_2 --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_2_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_source.pkl
echo "codebert 3"
python run_xglue.py --output_dir=./saved_models_2 --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_3_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_source.pkl
echo "codebert 4"
python run_xglue.py --output_dir=./saved_models_2 --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_4_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_source.pkl
echo "codebert 5"
python run_xglue.py --output_dir=./saved_models_2 --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_5_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_source.pkl
echo "graphcodebert 1"
python run_xglue.py --output_dir=./saved_models_3 --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_1_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_graph_source.pkl
echo "graphcodebert 2"
python run_xglue.py --output_dir=./saved_models_3 --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_2_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_graph_source.pkl
echo "graphcodebert 3"
python run_xglue.py --output_dir=./saved_models_3 --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_3_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_graph_source.pkl
echo "graphcodebert 4"
python run_xglue.py --output_dir=./saved_models_3 --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_4_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_graph_source.pkl
echo "graphcodebert 5"
python run_xglue.py --output_dir=./saved_models_3 --model_type=roberta --tokenizer_name=../models/graphcodebert --model_name_or_path=../models/graphcodebert --do_eval --train_data_file=datasets/BCB/type/type_train_xglue.jsonl --eval_data_file=datasets/BCB/type/type_5_eval.jsonl --test_data_file=datasets/BCB/type/type_5_eval.jsonl --epoch 2 --block_size 512 --train_batch_size 16 --eval_batch_size 32 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --pkl_file=datasets/BCB/preprocess/path_embeddings_graph_source.pkl