import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

# utilities
def plot_hist(data, path_outfile):
    print('[Fig] >> {}'.format(path_outfile))
    data_mean = np.mean(data)
    data_std = np.std(data)

    print('mean:', data_mean)
    print(' std:', data_std)

    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title('mean: {:.3f}_std: {:.3f}'.format(data_mean, data_std))
    plt.savefig(path_outfile)
    plt.close()


def traverse_dir(
        root_dir,
        extension=('mid', 'MID'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


# define event
def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event

# core functions
def corpus2event_remi_v2(path_infile, path_outfile):
    '''
    <<< REMI v2 >>>
    task: 2 track 
        1: piano      (note + tempo + chord)
    ---
    remove duplicate position tokens
    '''
    data = pickle.load(open(path_infile, 'rb'))

    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_event('Bar', None))

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing] # piano track

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                pos_events.append(create_event('Chord', root+'_'+quality))

            # tempo
            if len(t_tempos):
                pos_events.append(create_event('Tempo', t_tempos[0].tempo))

            # note 
            if len(t_notes):
                for note in t_notes:
                    pos_events.extend([
                        create_event('Note_Pitch', note.pitch),
                        create_event('Note_Velocity', note.velocity),
                        create_event('Note_Duration', note.duration),
                    ])

            # collect & beat
            if len(pos_events):
                final_sequence.append(
                    create_event('Beat', (timing-bar_step)//TICK_RESOL))
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_event('Bar', None))   

    # EOS
    final_sequence.append(create_event('EOS', None))   

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, 'wb'))
    return len(final_sequence)


if __name__ == '__main__':
    # paths
    path_indir = '../../../corpus'
    path_outdir = 'remi-v2_single/events'
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        extension=('pkl'),
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num fiels:', n_files)

    # run all
    len_list = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)

        # proc
        num_tokens = corpus2event_remi_v2(path_infile, path_outfile)
        print(' > num_token:', num_tokens)
        len_list.append(num_tokens)

    plot_hist(len_list, 'num_tokens.png')