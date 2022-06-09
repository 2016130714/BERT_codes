# -*- coding: utf-8 -*-

############################### TODO ##########################################
# 필요한 모듈 불러오기
###############################################################################

import argparse
import os
import pickle
import tensorflow as tf


# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
type_ = args.type

# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)

if type_ == 'bert':
############################### TODO 경로 고치기 ##########################################
    bert_model_hub_path = '/content/drive/MyDrive/bert-module'
###########################################################################################
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


############################### TODO ##########################################
# 모델과 벡터라이저 불러오기
###############################################################################


while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()
    except:
        continue
        
    if input_text == 'quit':
        break

############################### TODO ##########################################
# 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
###############################################################################

tf.compat.v1.reset_default_graph()

