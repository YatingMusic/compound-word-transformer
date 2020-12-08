from model import  TransformerXL
import pickle
import random
import os
import time
import torch
import random
import yaml
import json

import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main():
    cfg = yaml.full_load(open("config.yml", 'r')) 
    inferenceConfig = cfg['INFERENCE']
    
    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['gpuID']

    print('='*2, 'Inferenc configs', '='*5)
    print(json.dumps(inferenceConfig, indent=1, sort_keys=True))

    # checkpoint information
    CHECKPOINT_FOLDER = inferenceConfig['experiment_dir']
    midi_folder = inferenceConfig["generated_dir"]

    checkpoint_type = inferenceConfig['checkpoint_type']
    if checkpoint_type == 'best_train':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best.pth.tar')
        output_prefix = 'best_train_'
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
        output_prefix = 'best_val_'
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inferenceConfig['model_epoch'])))
        output_prefix = str(inferenceConfig['model_epoch'])+ '_'

    pretrainCfg = yaml.full_load(open(os.path.join(CHECKPOINT_FOLDER,"config.yml"), 'r')) 
    modelConfig = pretrainCfg['MODEL']

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)

    # load dictionary
    event2word, word2event = pickle.load(open(inferenceConfig['dictionary_path'], 'rb'))

    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model =  TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inferenceConfig["num_sample"]
    for idx in range(num_samples):
        print(f'==={idx}/{num_samples}===')
        print(midi_folder, output_prefix + str(idx))
        song_time, word_len = model.inference(
            model_path = model_path,
            token_lim=7680,
            strategies=['temperature', 'nucleus'],
            params={'t': 1.2, 'p': 0.9},
            bpm=120,
            output_path='{}/{}.mid'.format(midi_folder, output_prefix + str(idx)))

        print('song time:',  song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
    

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }
    

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)

if __name__ == '__main__':
    main()