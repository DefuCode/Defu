import os
import sys
# sys.path.append('/home/EPVD/')
sys.path.append('..')
import parserTool.parse as ps
from c_cfg_3 import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import json
import pickle
import logging
import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

# MODEL_CLASSES = {
#     'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
# }
# config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
# tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base", do_lower_case=True)
#
# logger = logging.getLogger(__name__)

def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out
    
def main():
    output = open('datasets/test1.pkl', 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0
    sum_ratio = 0
    num_path_dict = {}
    code_path = '../test_dataset/id2sourcecode'
    file_list = os.listdir(code_path)

    for file in file_list:
        code = open(f"{code_path}/{file}", encoding='UTF-8').read()
        code = code.strip()
        num_id += 1
        if num_id%100==0:
            print(num_id, flush=True)
        print(file)
        clean_code, code_dict = remove_comments_and_docstrings(code, 'java')
        g = C_CFG()
        code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        s_ast = g.parse_ast_file(code_ast.root_node)
        num_path, cfg_allpath, _, ratio = g.get_allpath()
        sum_ratio += ratio
        path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
        path_dict[file[:-5]] = path_tokens1, cfg_allpath
        #print("num_paths:", num_path)
    print("test file finish...", flush=True)
    print(sum_ratio/num_id, flush=True)
    # Pickle dictionary using protocol 0.
    pickle.dump(path_dict, output)
    output.close()

if __name__=="__main__":
    main()
