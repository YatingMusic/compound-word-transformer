import miditoolkit
import numpy as np
import pandas as pd
import copy
from functools import wraps
import time
import os

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, value):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, value={})'.format(
            self.name, self.start, self.end, self.velocity, self.value)

# read midi
def read_midi(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note',
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            value=note.pitch))
    note_items.sort(key=lambda x: x.start)
    return note_items, midi_obj.ticks_per_beat

# quantize items
def quantize_items(items, resolution):
    grids = np.arange(0, items[-1].start, resolution, dtype=int)
    output = []
    for i in range(len(items)):
        temp = copy.deepcopy(items[i])
        index = np.argmin(abs(grids-temp.start))
        shift = grids[index] - temp.start
        temp.start += shift
        if temp.end:
            temp.end += shift
        output.append(temp)
    return output

# group items
def group_items(note_items, resolution):
    flags = np.arange(0, note_items[-1].end+1, resolution)
    items = note_items
    groups = []
    for i in range(len(flags)-1):
        insiders = []
        for j in range(len(items)):
            if (items[j].start >= flags[i]) and (items[j].start < flags[i+1]):
                insiders.append(items[j])
        overall = [flags[i]] + insiders + [flags[i+1]]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups, ticks_per_beat):
    DEFAULT_DURATION_BINS = np.arange(
        ticks_per_beat/8, ticks_per_beat*8+1, ticks_per_beat/8)
    events = []
    n_bar = 0
    for group in groups:
        n_bar += 1
        if len(group) == 2: # without any notes
            continue
        else:
            # bar event
            bar_st, bar_et = group[0], group[-1]
            events.append(Event(
                name='Bar',
                time=None, 
                value=None,
                text='{}'.format(n_bar)))
            # note event
            for item in group[1:-1]:
                # position
                flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
                index = np.argmin(abs(flags-item.start))
                events.append(Event(
                    name='Position', 
                    time=item.start,
                    value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.value,
                    text='{}'.format(item.value)))
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
    return events

# convert word to event
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

# write midi
def write_midi(words, ticks_per_beat, bpm, word2event, output_path):
    DEFAULT_DURATION_BINS = np.arange(
        DEFAULT_RESOLUTION/8, DEFAULT_RESOLUTION*8+1, DEFAULT_RESOLUTION/8)
    events = word_to_event(words, word2event)
    # get bar and note (no time)
    temp_notes = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note On' and \
            events[i+2].name == 'Note Velocity' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # pitch
            pitch = int(events[i+1].value)
            # velocity
            index = int(events[i+2].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
    # get specific time for notes
    ticks_per_bar = DEFAULT_RESOLUTION * 4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, int(st), int(et)))
    # write
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    midi.tempo_changes = [miditoolkit.midi.containers.TempoChange(bpm, 0)]
    # write
    midi.dump(output_path)


def record_song_time(song_info):
    PATH = './songTime.csv'
    
    df = pd.DataFrame(song_info)
    df.to_csv(PATH)
    print('Time result saved to ', PATH)



def record_time(times, output_path):
    _df = pd.DataFrame().from_dict(result_dict)
    df.to_csv(output_path[:-3]+'time.csv')


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running [%s]: %s seconds" % (function.__name__, str(t1-t0)))
        return result
    return function_timer
