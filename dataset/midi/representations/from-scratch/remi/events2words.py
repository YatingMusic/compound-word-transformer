import os
import pickle
import numpy as np

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


if __name__ == '__main__':
    # paths
    path_root = 'remi-v2_single'
    path_indir = 'remi-v2_single/events'
    path_outdir = 'remi-v2_single/words'
    path_dictionary = os.path.join(path_root, 'dictionary.pkl')
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    eventfiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True,
        extension=('pkl'))
    n_files = len(eventfiles)
    print('num fiels:', n_files)

    # --- generate dictionary --- #
    print(' [*] generating dictionary')
    all_events = []
    for file in eventfiles:
        for event in pickle.load(open(os.path.join(path_indir, file), 'rb')):
            all_events.append('{}_{}'.format(event['name'], event['value']))

    # build
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    print(' > num classes:', len(word2event))

    # save
    pickle.dump((event2word, word2event), open(path_dictionary, 'wb'))

    # --- converts to word --- #
    event2word, word2event = pickle.load(open(path_dictionary, 'rb'))
    for fidx, file in enumerate(eventfiles):
        print('{}/{}'.format(fidx, n_files))

        # events to words
        path_infile = os.path.join(path_indir, file)
        events = pickle.load(open(path_infile, 'rb'))
        words = []
        for event in events:
            word = event2word['{}_{}'.format(event['name'], event['value'])]
            words.append(word)

        # save
        path_outfile = os.path.join(path_outdir, file + '.npy')
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
        np.save(path_outfile, words)
