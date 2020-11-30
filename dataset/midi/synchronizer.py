import os
import glob
import copy
import librosa
import numpy as np
import multiprocessing as mp
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from miditoolkit.midi import parser
from miditoolkit.midi.containers import TimeSignature, TempoChange


def get_instruments_abs_timing(instruments, tick_to_time):
    return convert_instruments_timing_from_sym_to_abs(instruments, tick_to_time)


def convert_instruments_timing_from_sym_to_abs(instruments, tick_to_time):
    proc_instrs = copy.deepcopy(instruments)
    for instr in proc_instrs:
        for note in instr.notes:
            note.start = float(tick_to_time[note.start])
            note.end = float(tick_to_time[note.end])
    return proc_instrs


def convert_instruments_timing_from_abs_to_sym(instruments, time_to_tick):
    proc_instrs = copy.deepcopy(instruments)
    for instr in proc_instrs:
        for note in instr.notes:
            # find nearest
            note.start = find_nearest_np(time_to_tick, note.start)
            note.end = find_nearest_np(time_to_tick, note.end)
    return proc_instrs


def find_nearest_np(array, value):
    return (np.abs(array - value)).argmin()


def find_first_downbeat(proc_res):
    rythm = np.where(proc_res[:, 1] == 1)[0]
    pos = proc_res[rythm[0], 0]
    return pos


def interp_linear(src, target, num, tail=False):
    src = float(src)
    target = float(target)
    step = (target - src) / float(num)
    middles = [src + step * i for i in range(1, num)]
    res = [src] + middles
    if tail:
        res += [target]
    return res


def estimate_beat(path_audio):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(path_audio)
    proc_res = proc(act) 
    return proc_res


def export_audio_with_click(proc_res, path_audio, path_output, sr=44100):
    # extract time
    times_beat = proc_res[np.where(proc_res[:, 1]!=1)][:, 0]
    times_downbeat = proc_res[np.where(proc_res[:, 1]==1)][:, 0]

    # load
    y, _ = librosa.core.load(path_audio, sr=sr) 

    # click audio
    y_beat = librosa.clicks(times=times_beat, sr=sr, click_freq=1200, click_duration=0.5) * 0.6
    y_downbeat = librosa.clicks(times=times_downbeat, sr=sr, click_freq=600, click_duration=0.5)

    # merge
    max_len = max(len(y), len(y_beat), len(y_downbeat))
    y_integrate = np.zeros(max_len)
    y_integrate[:len(y_beat)] += y_beat
    y_integrate[:len(y_downbeat)] += y_downbeat
    y_integrate[:len(y)] += y

    librosa.output.write_wav(path_output, y_integrate, sr)


def align_midi(proc_res, path_midi_input, path_midi_output, ticks_per_beat=480):
    midi_data = parser.MidiFile(path_midi_input)

    # compute tempo
    beats = np.array([0.0] + list(proc_res[:, 0]))
    intervals = np.diff(beats)
    bpms = 60 / intervals
    tempo_info = list(zip(beats[:-1], bpms))
    
    # get absolute timing of instruments
    tick_to_time = midi_data.get_tick_to_time_mapping()
    abs_instr = get_instruments_abs_timing(midi_data.instruments, tick_to_time)

    # get end time of file
    end_time = midi_data.get_tick_to_time_mapping()[-1]

    # compute time to tick mapping
    resample_timing = []
    for i in range(len(beats)-1):
        start_beat = beats[i]
        end_beat = beats[i + 1]
        resample_timing += interp_linear(start_beat, end_beat, ticks_per_beat)
        
    # fill the empty in the tail (using last tick interval)
    last_tick_interval = resample_timing[-1] - resample_timing[-2]
    cur_time = resample_timing[-1]
    while cur_time < end_time:
        cur_time += last_tick_interval
        resample_timing.append(cur_time)
    resample_timing = np.array(resample_timing)
        
    # new a midifile obj
    midi_res = parser.MidiFile()

    # convert abs to sym
    sym_instr = convert_instruments_timing_from_abs_to_sym(abs_instr, resample_timing)
    
    # time signature
    first_db_sec = find_first_downbeat(proc_res)
    first_db_tick = find_nearest_np(resample_timing, first_db_sec)
    time_signature_changes = [TimeSignature(numerator=4, denominator=4, time=int(first_db_tick))]
    
    # tempo
    tempo_changes = [] 
    for pos, bpm in tempo_info:
        pos_tick = find_nearest_np(resample_timing, pos)
        tempo_changes.append(TempoChange(tempo=float(bpm), time=int(pos_tick)))
    
    # shift (pickup at the beginning)
    shift_align = ticks_per_beat * 4 - first_db_tick
    
    # apply shift to tempo
    for msg in tempo_changes:
        msg.time += shift_align
        
    # apply shift to notes
    for instr in sym_instr:
        for note in instr.notes:
            note.start += shift_align
            note.end += shift_align
            
    # set attributes
    midi_res.ticks_per_beat = ticks_per_beat
    midi_res.tempo_changes = tempo_changes 
    midi_res.time_signature_changes = time_signature_changes 
    midi_res.instruments = sym_instr
    
    # saving
    midi_res.dump(filename=path_midi_output)

    
def analyze(path_midi_input, path_audio_input, path_midi_output, path_audio_output=None):
    print(path_midi_input)
    # beat tracking
    proc_res = estimate_beat(path_audio_input)
    # export audio with click
    if path_audio_output is not None:
        export_audio_with_click(proc_res, path_audio_input, path_audio_output)
    # export midi file
    align_midi(proc_res, path_midi_input, path_midi_output)

if __name__ == '__main__':
    files = sorted(glob.glob('mp3/MusicBand-Guide/*.mp3'))

    data = []
    for i in range(len(files)):
        folder = files[i].split('/')[-2]
        name = files[i].split('/')[-1][:-4]
        path_midi_input = 'midi/{}/{}.wav.midi'.format(folder, name)
        path_audio_input = files[i]
        path_midi_output = 'aligned-midi/{}/{}.mid'.format(folder, name)
        data.append([path_midi_input, path_audio_input, path_midi_output, None])

    pool = mp.Pool()
    pool.starmap(analyze, data)

    # for i in range(len(data)):
    #     analyze(path_midi_input=data[i][0], path_audio_input=data[i][1],
    #         path_midi_output=data[i][2], path_audio_output=None)
    #     break