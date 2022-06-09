# -*- coding: utf-8 -*-

from utils import Reader
from to_array.bert_to_array import BERTToArray
from to_array.tags_to_array import TagsToArray
from models.bert_slot_model import BertSlotModel

import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf


# read command-line parameters
parser = argparse.ArgumentParser('Training the BERT NLU model')
parser.add_argument('--train', '-t', help = 'Path to training set', type = str, required = True)
parser.add_argument('--val', '-v', help = 'Path to validation set', type = str, default = "", required = False)
parser.add_argument('--save', '-s', help = 'Folder path to save the trained model', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs', type = int, default = 5, required = False)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 64, required = False)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
type_ = args.type

# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.compat.v1.random.set_random_seed(7)

config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)

if type_ == 'bert':

############################### TODO 경로 고치기 ##########################################
    bert_model_hub_path = "/content/drive/MyDrive/bert-module"
###########################################################################################
    bert_vocab_path = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = '' # fill out
    bert_vocab_path = '' # fill out
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))

print('read data ...')
train_text_arr, train_tags_arr = Reader.read(train_data_folder_path)

print('train_text_arr[0:2] :', train_text_arr[0:2])
print('train_tags_arr[0:2] :', train_tags_arr[0:2])

bert_to_array = BERTToArray(is_bert, bert_vocab_path) 
# bert_to_array MUST NOT tokenize input !!!

print('bert toarray started ...')
train_input_ids, train_input_mask, train_segment_ids = bert_to_array.transform(train_text_arr)

print('vectorize tags ...')
tags_to_array = TagsToArray()
tags_to_array.fit(train_tags_arr)
train_tags = tags_to_array.transform(train_tags_arr, train_input_ids)
print('train_tags :', train_tags[0:2])
slots_num = len(tags_to_array.label_encoder.classes_)
print('slot num :', slots_num, tags_to_array.label_encoder.classes_)


model = BertSlotModel(slots_num, bert_model_hub_path, sess, 
                    num_bert_fine_tune_layers=10, is_bert=is_bert)


print('train input shape :', train_input_ids.shape, train_input_ids[0:2])
print('train_input_mask :', train_input_mask.shape, train_input_mask[0:2])
print('train_segment_ids :', train_segment_ids.shape, train_segment_ids[0:2])
print('train_tags :', train_tags.shape, train_tags[0:2])


if val_data_folder_path:
    print('preparing validation data')
    val_text_arr, val_tags_arr = Reader.read(val_data_folder_path)
    val_input_ids, val_input_mask, val_segment_ids = bert_to_array.transform(val_text_arr)
    val_tags = tags_to_array.transform(val_tags_arr, val_input_ids)

    print('training model ...')
    model.fit([train_input_ids, train_input_mask, train_segment_ids], train_tags,
        validation_data=([val_input_ids, val_input_mask, val_segment_ids], val_tags),
        epochs=epochs, batch_size=batch_size)    
else:
    print('training model ...')
    model.fit([train_input_ids, train_input_mask, train_segment_ids], train_tags,
        validation_data=None, 
        epochs=epochs, batch_size=batch_size)    

### saving
print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_to_array.pkl'), 'wb') as handle:
    pickle.dump(tags_to_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()
