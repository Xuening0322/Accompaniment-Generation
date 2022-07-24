import os
import numpy as np
from music21 import *
from mido import MidiFile, MidiTrack, MetaMessage
from .loader import get_filenames, convert_files
from .model import build_model
from .config import *
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


def export_music(score, chord_list, gap_list, filename):
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

    # Todo: reset the chord track tempo, change the parameters
    mid = MidiFile(output_filename)
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            msg.tempo = 750000
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
    new_mid.save(OUTPUTS_PATH + '/' + filename + '_all.mid')
    return OUTPUTS_PATH + '/' + filename + '_all.mid'


def generate_chord(midi_path):
    # Build model
    model = build_model(weights_path='XGeneration/task2_explicit/weights.hdf5')
    data_corpus = convert_files([midi_path], fromDataset=False)
    output_path = ''
    # Process each melody sequence
    for idx in trange(len(data_corpus)):
        melody_vecs = data_corpus[idx][0]
        gap_list = data_corpus[idx][1]
        score = data_corpus[idx][2]
        filename = data_corpus[idx][3]

        chord_list = predict(melody_vecs, model)
        output_path = export_music(score, chord_list, gap_list, filename)

    return output_path


if __name__ == '__main__':

    generate_chord('/home/ld/folder/chord_generation/inputs/demo_20220723_040437.mid')

    # # Build model
    # model = build_model(weights_path='weights.hdf5')
    # # filenames = get_filenames(input_dir=INPUTS_PATH)
    # filenames = ['/home/ld/folder/chord_generation/inputs/demo_20220723_040437.mid']
    # data_corpus = convert_files(filenames, fromDataset=False)
    #
    # # Process each melody sequence
    # for idx in trange(len(data_corpus)):
    #     melody_vecs = data_corpus[idx][0]
    #     gap_list = data_corpus[idx][1]
    #     score = data_corpus[idx][2]
    #     filename = data_corpus[idx][3]
    #
    #     chord_list = predict(melody_vecs, model)
    #     export_music(score, chord_list, gap_list, filename)
