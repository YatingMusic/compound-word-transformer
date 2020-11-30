import sys
import os


import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import saver

# ------------------------------ #
## Config 1 nase, gpu2, p1
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8    
path_exp = 'exp_base_test'

batch_size = 3
gid = 0
init_lr = 0.0001
# ------------------------------ #
## Config 1 large, titan, p0
# D_MODEL = 512
# N_LAYER = 18
# N_HEAD = 16
# path_exp = 'exp_large'
# init_lr = 0.0001
# batch_size = 5
# gid = 1
# ------------------------------ #

os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def write_midi(words, path_outfile, word2event):
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0
    cur_track = None

    track_notes = {
        'piano': [],
        'melody': [],
    }

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
#         print(vals)

        if vals[4] == 'Metrical':
            if vals[3] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[3]:
                beat_pos = int(vals[3].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if cur_track == 'melody':
                    if vals[2] != 'CONTI' and vals[2] != 0:
                        midi_obj.markers.append(
                            Marker(text=str(vals[2]), time=cur_pos))
                elif cur_track == 'piano':
                    try:
                        if vals[1] != 'CONTI' and vals[1] != 0:
                            tempo = int(vals[1].split('_')[-1])
                            midi_obj.tempo_changes.append(
                                TempoChange(tempo=tempo, time=cur_pos))
                    except:
                        cnt_error += 1
                        continue
                else:
                    pass
            else:
                pass
        elif vals[4] == 'Track':
            if vals[0] == 'track:LeadSheet':
                cur_track = 'melody'
            elif vals[0] == 'track:Piano':
                cur_track = 'piano'
            else:
                pass
        elif vals[4] == 'Note':
            if cur_track == 'melody':
                pitch = vals[5].split('_')[-1]
                duration = vals[6].split('_')[-1]
                velocity = 80
            elif cur_track == 'piano':
                try:
                    pitch = vals[5].split('_')[-1]
                    duration = vals[6].split('_')[-1]
                    velocity = vals[7].split('_')[-1]
                except:
                    cnt_error += 1
                    continue
            else:
                pass
            if int(duration) == 0:
                duration = 60
            end = cur_pos + int(duration)
            
            track_notes[cur_track].append(
                Note(
                    pitch=int(pitch), 
                    start=cur_pos, 
                    end=end, 
                    velocity=int(velocity)))
        else:
            pass

    piano_track = Instrument(0, is_drum=False, name='piano')
    melody_track = Instrument(0, is_drum=False, name='melody')
    piano_track.notes = track_notes['piano']
    melody_track.notes = track_notes['melody']

    midi_obj.instruments = [piano_track, melody_track]
    midi_obj.dump(path_outfile)


########################################
# search strategy: sampling (re-shape)
########################################
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

########################################
# search strategy: nucleus (truncate)
########################################
def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)
    
    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word

########################################
def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # print('pe:', pe.shape)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(TransformerModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token   
        self.d_model = D_MODEL 
        self.n_layer = N_LAYER #
        self.dropout = 0.1
        self.n_head = N_HEAD #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [32, 128, 256, 64, 32, 512, 128, 128]

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.word_emb_track     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_tempo     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_chord     = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_barbeat   = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_type      = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_pitch     = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_duration  = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.word_emb_velocity  = Embeddings(self.n_token[7], self.emb_sizes[7])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

         # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (validation)
            print(' [o] using RNN backend.')
            self.transformer_encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

        # output layer for softmax
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        self.proj_track    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[1])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[2])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[3])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[4])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[5])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[6])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[7])


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, target, loss_mask):
        h, y_type  = self.forward_hidden(x)
        y_track, y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)
        
        
        # reshape (b, s, f) -> (b, f, s)
        y_track     = y_track[:, ...].permute(0, 2, 1)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        
        # loss
        loss_track = self.compute_loss(
                y_track, target[..., 0], loss_mask)
        loss_tempo = self.compute_loss(
                y_tempo, target[..., 1], loss_mask)
        loss_chord = self.compute_loss(
                y_chord, target[..., 2], loss_mask)
        loss_barbeat = self.compute_loss(
                y_barbeat, target[..., 3], loss_mask)
        loss_type = self.compute_loss(
                y_type,  target[..., 4], loss_mask)
        loss_pitch = self.compute_loss(
                y_pitch, target[..., 5], loss_mask)
        loss_duration = self.compute_loss(
                y_duration, target[..., 6], loss_mask)
        loss_velocity = self.compute_loss(
                y_velocity, target[..., 7], loss_mask)

        return loss_track, loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity

    def forward_hidden(self, x, memory=None, is_training=True):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
    
        # embeddings
        emb_track =    self.word_emb_track(x[..., 0])
        emb_tempo =    self.word_emb_tempo(x[..., 1])
        emb_chord =    self.word_emb_chord(x[..., 2])
        emb_barbeat =  self.word_emb_barbeat(x[..., 3])
        emb_type =     self.word_emb_type(x[..., 4])
        emb_pitch =    self.word_emb_pitch(x[..., 5])
        emb_duration = self.word_emb_duration(x[..., 6])
        emb_velocity = self.word_emb_velocity(x[..., 7])

        embs = torch.cat(
            [
                emb_track,
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        # assert False
    
        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(pos_emb, attn_mask) # y: b x s x d_model

            # project type
            y_type = self.proj_type(h)
            return h, y_type
        else:
            pos_emb = pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory) # y: s x d_model
            
            # project type
            y_type = self.proj_type(h)
            return h, y_type, memory

    def forward_output(self, h, y):
        tf_skip_type = self.word_emb_type(y[..., 4])

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        y_track    = self.proj_track(y_)
        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)

        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        return  y_track, y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity

    def froward_output_sampling(self, h, y_type):
        '''
        type用nucleus效果好像很差
        t=10 會成塊狀
        t<1 groove會變爛
        '''
        # sample type
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, t=0.99)

        type_word_t = torch.from_numpy(
                    np.array([cur_word_type])).long().cuda().unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        # project other
        y_track    = self.proj_track(y_)
        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)

        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        
        # sampling gen_cond
        cur_word_track =    sampling(y_track, p=0.9)
        cur_word_tempo =    sampling(y_tempo, p=0.9, t=1.2)
        cur_word_chord =    sampling(y_chord, p=0.9)
        cur_word_barbeat =  sampling(y_barbeat,t=1.2)
        cur_word_pitch =    sampling(y_pitch, p=0.9, t=1.2)
        cur_word_duration = sampling(y_duration, p=0.9, t=2)
        cur_word_velocity = sampling(y_velocity, t=5)

        # cur_word_track =    sampling(y_track, p=0.8)
        # cur_word_track = torch.argmax(y_track).cpu().numpy()
        # cur_word_track =    sampling(y_track, t=1.2)
        # cur_word_tempo =    sampling(y_tempo, t=1.2)
        # cur_word_chord =    sampling(y_chord, t=1.2)
        # cur_word_barbeat =  sampling(y_barbeat,t=1.2)
        # cur_word_pitch =    sampling(y_pitch, t=1.2)
        # cur_word_duration = sampling(y_duration, t=1.2)
        # cur_word_velocity = sampling(y_velocity, t=1.2)
        
        # collect
        next_arr = np.array([
            cur_word_track,
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_type,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
            ])        
        return next_arr

    def inference_from_scratch(self, dictionary):
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0], # bar
            [1, 0, 0, 0, 3, 0, 0, 0]  # track:leadsheet
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None
            
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])

                h, y_type, memory = self.forward_hidden(
                        input_, memory, is_training=False)

            print('------ generation ------')
            while(True):
                # sample others
                next_arr = self.froward_output_sampling(h, y_type)
                final_res.append(next_arr[None, ...])
                print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_  = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

                # end of sequence
                if word2event['type'][next_arr[4]] == 'EOS':
                    break

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res

                
    def inference_condition(self, dictionary, cond_words):
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        print(' > num conditions blocks:', len(cond_words))

        with torch.no_grad():
            final_res = []
            memory = None
            h = None
            for block_idx in range(len(cond_words)):
                conds = cond_words[block_idx]
                conds_t = torch.from_numpy(conds).long().cuda()
                
                print('------ condition ------')
                for step in range(conds.shape[0]):
                    print_word_cp(conds[step, :])
                    input_ = conds_t[step, :].unsqueeze(0).unsqueeze(0)
                    final_res.append(conds[step, :][None, ...])

                    h, y_type, memory = self.forward_hidden(
                        input_, memory, is_training=False)

                print('------ generation ------')
                while(True):
                    # sample others
                    next_arr = self.froward_output_sampling(h, y_type)
                    final_res.append(next_arr[None, ...])
                    print_word_cp(next_arr)

                    # forward
                    input_ = torch.from_numpy(next_arr).long().cuda()
                    input_  = input_.unsqueeze(0).unsqueeze(0)
                    h, y_type, memory = self.forward_hidden(
                        input_, memory, is_training=False)

                    # end of gen generation
                    if word2event['bar-beat'][next_arr[3]] == 'Bar':
                        break
        
        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res
        

##########################################################################################################################
# Script
##########################################################################################################################

def get_conds(song, word2event):
    classes = word2event.keys()
    def print_word_cp(cp):
        result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
        print(result)

    cond_words = []
    tmp = []
    is_ls = True

    for i in range(song.shape[0]):
        word = song[i]
        cur_event = word2event['track'][word[0]]
        
        if cur_event == 'track:LeadSheet':
            is_ls = True
        elif cur_event == 'track:Piano':
            tmp.append(word)
            cond_words.append(np.array(tmp))
            tmp = []
            is_ls = False
        
        # print(int(is_ls), end=' ')
        # print_word_cp(word)

        if is_ls:
            tmp.append(word)
    return cond_words

def generate(path_ckpt, name, path_outdir, mode='condition'):
    # load
    dictionary = pickle.load(open('../datasets/cp_v1_linear_m1/dictionary.pkl', 'rb'))
    event2word, word2event = dictionary
    test_data = np.load('../datasets/cp_v1_linear_m1/test_data_linear.npz')

    # ckpt
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
    print('path_saved_ckpt:', path_saved_ckpt)

    # outdir
    os.makedirs(path_outdir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    # song_idxs = [1, 3, 5, 7, 9, -1, -3, -5, -7, -9]
    song_idxs = [5]
    cnt_tokens_all = 0 

    song_time_list = []
    words_len_list = []

    for sidx in range(50):
    # for sidx in  song_idxs:
        start_time = time.time()
        print('current idx:', sidx)
        path_outfile = os.path.join(path_outdir, 'get_{}.mid'.format(str(sidx)))

        song = test_data['x'][sidx]
        cond_words = get_conds(song, word2event)

        if mode == 'condition':
            res = net.inference_condition(dictionary, cond_words)
        elif mode == 'from-scratch':
            res = net.inference_from_scratch(dictionary)
        else:
            raise ValueError(' [!] Unknown mode: {}'.format(mode))
        # np.save(path_outfile, res)

        write_midi(res, path_outfile, word2event)

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
  
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        ' words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats_2.json', 'w') as f:
        json.dump(runtime_result, f)


    # print('Elapsed Time:', str(datetime.timedelta(seconds=runtime)))

    # # to midi
    # npylist = glob.glob(os.path.join(path_outdir, '*.npy'))
    # for path_infile in npylist:
    #     print(' >>> to midi:', path_infile)
    #     write_midi(path_infile, path_infile+'.mid', word2event)


def train():
    # hyper params

    n_epoch = 4000
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open('../datasets/cp_v1_linear/dictionary.pkl', 'rb'))
    event2word, word2event = dictionary
    train_data = np.load('../datasets/cp_v1_linear/train_data_linear.npz')

    # create saver
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)
   
    # init
    net = TransformerModel(n_class)
    net.cuda()
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack
    train_x = train_data['x']
    train_y = train_data['y']
    train_mask = train_data['mask']
    num_batch = len(train_x) // batch_size
    
    print('     num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)
    print('    train_mask:', train_mask.shape)

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(8)
        for bidx in range(num_batch): # num_batch 
            saver_agent.global_step_increment()

            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).long().cuda()
            batch_y = torch.from_numpy(batch_y).long().cuda()
            batch_mask = torch.from_numpy(batch_mask).float().cuda()

            # run
            losses = net.train_step(batch_x, batch_y, batch_mask)
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6] + losses[7]) / 8

            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6], losses[7]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())
        
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('---')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
              acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5], acc_losses[6], acc_losses[7])
        print('    >', each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model
        loss = epoch_loss
        if 0.3 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif 0.05 < loss <= 0.30:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif loss <= 0.05:
            print('Finished')
            return  
        else:
            saver_agent.save_model(net, name='loss_high')

if __name__ == '__main__':
    # train()
    path_ckpt = 'exp_base'

    gen_settings = [
        # (22, 'gen_m1_cond_all_final_sampling3', 'condition'),
        (27, 'gen_aaai_27', 'condition'),
        # (23, 'gen_m1/from-scratch', 'from-scratch'),
    ]
    for loss, path_outdir, mode in gen_settings:
        generate(path_ckpt, 'loss_' + str(loss), path_outdir, mode)