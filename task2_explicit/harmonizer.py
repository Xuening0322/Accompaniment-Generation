import os
import numpy as np
from music21 import *
import pretty_midi
import math
from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo, tempo2bpm
from loader import get_filenames, convert_files
from model import build_model
from config import *
from tqdm import trange

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

chord_dictionary = ['R',
                    'Cm', 'C',
                    'C#m', 'C#',
                    'Dm', 'D',
                    'D#m', 'D#',
                    'Em', 'E',
                    'Fm', 'F',
                    'F#m', 'F#',
                    'Gm', 'G',
                    'G#m', 'G#',
                    'Am', 'A',
                    'A#m', 'A#',
                    'Bm', 'B']


def predict(song, model):
    chord_list = []

    # Traverse the melody sequence
    for idx in range(int(len(song) / 4)):

        # Create input data
        melody = [song[idx * 4], song[idx * 4 + 1], song[idx * 4 + 2], song[idx * 4 + 3]]
        melody = np.array([np.array(seg) for seg in melody])[np.newaxis, ...]

        # Predict the next four chords
        net_output = model.predict(melody)[0]

        for chord_idx in net_output.argmax(axis=1):
            chord_list.append(chord_dictionary[chord_idx])

    # Create input data
    melody = [song[-4], song[-3], song[-2], song[-1]]
    melody = np.array([np.array(seg) for seg in melody])[np.newaxis, ...]

    # Predict the last four chords
    net_output = model.predict(melody)[0]

    for idx in range(-1 * (len(song) % 4), 0):
        chord_list.append(chord_dictionary[net_output[idx].argmax(axis=0)])

    return chord_list


def clamp_midi_data(input_midi_data):
    tempo = input_midi_data.get_tempo_changes()[1][0]

    original_end_time = input_midi_data.instruments[0].notes[-1].end
    # end_time -> beats
    original_end_beat = (original_end_time / 60) * tempo

    end_time = math.ceil((60 / tempo) * original_end_beat / 2) * 2
    # end_time = input_midi_data.tick_to_time(last_tick)
    # extends first and last note to the end
    input_midi_data.instruments[0].notes[0].start = 0
    input_midi_data.instruments[0].notes[-1].end = end_time


def quantize_midi_data(input_midi_data: pretty_midi.PrettyMIDI):
    starts = input_midi_data.get_onsets()
    tempo = input_midi_data.get_tempo_changes()[1][0]

    def quantize(time):

        beat = round((time / 60) * tempo * 8) / 8

        new_time = ((beat / tempo) * 60)

        return new_time

    # get ends:
    ends = [note.end for note in input_midi_data.instruments[0].notes]
    delete_notes_index = []

    for i, note in enumerate(input_midi_data.instruments[0].notes):
        note.start = quantize(note.start)
        note.end = quantize(note.end)
        if i != 0:
            if note.start == input_midi_data.instruments[0].notes[i - 1].start:
                # choose longest => choose last end
                if note.end > input_midi_data.instruments[0].notes[i - 1].end:
                    delete_notes_index.append(i - 1)
                else:
                    delete_notes_index.append(i)
            elif note.start >= input_midi_data.instruments[0].notes[i - 1].start and note.start < \
                    input_midi_data.instruments[0].notes[i - 1].end:
                input_midi_data.instruments[0].notes[i - 1].end = note.start

    input_midi_data.instruments[0].notes = [
        note for i, note in enumerate(input_midi_data.instruments[0].notes) if i not in delete_notes_index
    ]


def _print_notes(input_midi_data):
    for note in input_midi_data.instruments[0].notes:
        print(note)


def preprocess(input_midi_path):
    input_midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    # _print_notes(input_midi_data)
    clamp_midi_data(input_midi_data)
    quantize_midi_data(input_midi_data)
    # _print_notes(input_midi_data)
    # write midi data
    # output_midi_path = ''.join(input_midi_path.split('.')[:-1]) + '_preprocessed.mid'
    # print(output_midi_path)
    output_midi_path = input_midi_path
    input_midi_data.write(output_midi_path)

    return output_midi_path


def adjust_chord(input_path, output_path):

    mid = pretty_midi.PrettyMIDI(input_path)
    nn = []
    tempo = int(mid.get_tempo_changes()[1])
    expected_end_time = 60 / tempo * 16 * 4
    mid.instruments[1].program = 0

    for note in mid.instruments[1].notes:
        nn.append([note.start, note.end, note.pitch, note.velocity])
    nn.sort(key=lambda x: (x[0], x[1]), reverse=False)
    duration = set([])
    for i in range(len(nn)):
        duration.add(nn[i][1] - nn[i][0])
    min_duration = min(duration)

    for i in range(len(nn)):
        if nn[i][1] - nn[i][0] != min_duration:
            nn[i][1] = nn[i][0] + min_duration

    time_d = {}
    for i in nn:
        if i[0] not in time_d:
            time_d[i[0]] = 1
        else:
            time_d[i[0]] += 1

    for i in time_d.keys():
        if time_d[i] == 1:
            for j in range(len(nn)):
                if nn[j][0] == i:
                    # j: index
                    pre = nn[j - 3: j]
                    pitch = []
                    for k in pre:
                        pitch.append(k[2])
                    pitch.remove(nn[j][2])
                    for p in pitch:
                        nn.append([nn[j][0], nn[j][0] + min_duration, p, 90])
        if time_d[i] == 2:
            for j in range(len(nn)):
                if nn[j][0] == i:
                    # j: the first index
                    pre = nn[j - 3: j]
                    pitch = []
                    for k in pre:
                        pitch.append(k[2])
                    pitch.remove(nn[j][2])
                    pitch.remove(nn[j + 1][2])
                    for p in pitch:
                        nn.append([nn[j][0], nn[j][0] + min_duration, p, 90])

    nn.sort(key=lambda x: (x[0], x[1]), reverse=False)

    if nn[-1][1] != expected_end_time:
        nn[-1][1] = expected_end_time
        nn[-2][1] = expected_end_time
        nn[-3][1] = expected_end_time

    new = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    for n in nn:
        piano.notes.append(pretty_midi.Note(start=n[0], end=n[1], pitch=n[2], velocity=n[3]))
    new.instruments.append(mid.instruments[0])
    new.instruments.append(piano)
    new.write(output_path)


def export_music(score, chord_list, gap_list, filename, midi_tempo):
    harmony_list = []
    filepath = filename
    filename = os.path.basename(filename)
    filename = '.'.join(filename.split('.')[:-1])

    for idx in range(len(chord_list)):
        chord = chord_list[idx]
        if chord == 'R':
            harmony_list.append(note.Rest())
            print("chord = R")
        else:
            harmony_list.append(harmony.ChordSymbol(chord).transpose(-1 * gap_list[idx].semitones))

    m_idx = 0
    new_score = []

    for m in score.recurse():
        if isinstance(m, stream.Measure):
            new_m = []
            if not isinstance(harmony_list[m_idx], note.Rest):
                new_m.append(harmony_list[m_idx])
            new_m = stream.Measure(new_m)
            new_m.offset = m.offset
            new_score.append(new_m)
            m_idx += 1

    # Save only chord track as midi
    new_score[-1].rightBarline = bar.Barline('final')
    score = stream.Score(new_score)
    output_filename = OUTPUTS_PATH + '/' + filename + '_chord.mid'
    score.write('midi', fp=output_filename)

    mid = MidiFile(output_filename)
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            msg.tempo = midi_tempo
    mid.save(output_filename)

    # insert the chord track into the original melody file
    mid = MidiFile(filepath)
    new_mid = MidiFile(output_filename, ticks_per_beat=mid.ticks_per_beat)
    new_track = MidiTrack()
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.time != 0:
                msg.time = int(msg.time * 4.65)
            if not msg.is_meta:
                new_track.append(msg)

    new_mid.tracks.insert(1, new_track)
    new_mid_path = OUTPUTS_PATH + '/' + filename + '_all.mid'
    new_mid.save(new_mid_path)

    preprocess(new_mid_path)
    adjust_chord(new_mid_path, OUTPUTS_PATH + '/' + filename + '_all_adjusted.mid')


def generate_chord(midi_path):
    # Build model
    model = build_model(weights_path='weights.hdf5')
    data_corpus = convert_files([midi_path], fromDataset=False)

    mid = MidiFile(midi_path)
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            midi_tempo = msg.tempo
    # Process each melody sequence
    for idx in trange(len(data_corpus)):
        melody_vecs = data_corpus[idx][0]
        gap_list = data_corpus[idx][1]
        score = data_corpus[idx][2]
        filename = data_corpus[idx][3]

        chord_list = predict(melody_vecs, model)
        export_music(score, chord_list, gap_list, filename, midi_tempo)


if __name__ == '__main__':

    generate_chord('/root/chord_generation/demo_20220730_031258_preprocessed.mid')

