import os.path
import sys
import time
import datetime

import mido
import numpy as np
import pretty_midi
from torch.nn import functional as F
from .SketchVAE.sketchvae import SketchVAE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .SketchNet.sketchnet import SketchNet
from .utils.helpers import *
from .loader.dataloader import MIDI_Loader, MIDI_Render
from mido import Message, MidiFile, MidiTrack, tick2second

current_path = os.path.dirname(__file__)
s_dir = ""  # folder address
save_path = s_dir + "model_backup/"

# initial parameters
zp_dims = 128
zr_dims = 128
pf_dims = 512
gen_dims = 1024
combine_dims = 512
combine_head = 4
combine_num = 4
pf_num = 2
batch_size = 1

# for vae init
vae_hidden_dims = 1024
vae_zp_dims = 128
vae_zr_dims = 128
vae_beta = 0.1
vae_input_dims = 130
vae_pitch_dims = 129
vae_rhythm_dims = 3
vae_seq_len = 6 * 4
vae_beat_num = 4
vae_tick_num = 6

# note extraction
hold_state = 128
rest_state = 129


def process_note_time(input_midi_path):
    input_midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    tempo = int(input_midi_data.get_tempo_changes()[1])
    input_midi_data.instruments[0].notes[0].start = 0
    input_midi_data.instruments[0].notes[-1].end = 60 / tempo * 16
    input_midi_data.write(input_midi_path)


def extract_note(x, pad_token=128):
    d = []
    for i in x:
        if i < 128:
            d.append(i)
    ori_d = len(d)
    d.extend([pad_token] * (len(x) - len(d)))
    return np.array(d), ori_d


def extract_rhythm(x, hold_token=2, rest_token=3):
    d = []
    for i in x:
        if i < 128:
            d.append(1)
        elif i == hold_state:
            d.append(hold_token)
        else:
            d.append(rest_token)
    return np.array(d)


def processed_data_tensor(data):
    print("processed data:")
    gd = []
    px = []
    rx = []
    len_x = []
    nrx = []
    total = 0
    for i, d in enumerate(data):
        gd.append([list(dd[0]) for dd in d])
        px.append([list(dd[1]) for dd in d])
        rx.append([list(dd[2]) for dd in d])
        len_x.append([dd[3] for dd in d])
        if len(gd[-1][-1]) != vae_seq_len:
            gd[-1][-1].extend([128] * (vae_seq_len - len(gd[-1][-1])))
            px[-1][-1].extend([128] * (vae_seq_len - len(px[-1][-1])))
            rx[-1][-1].extend([2] * (vae_seq_len - len(rx[-1][-1])))
    for i, d in enumerate(len_x):
        for j, dd in enumerate(d):
            if len_x[i][j] == 0:
                gd[i][j][0] = 60
                px[i][j][0] = 60
                rx[i][j][0] = 1
                len_x[i][j] = 1
                total += 1
    gd = np.array(gd)
    px = np.array(px)
    rx = np.array(rx)
    len_x = np.array(len_x)
    for d in rx:
        nnrx = []
        for dd in d:
            temp = np.zeros((vae_seq_len, vae_rhythm_dims))
            lins = np.arange(0, len(dd))
            temp[lins, dd - 1] = 1
            nnrx.append(temp)
        nrx.append(nnrx)
    nrx = np.array(nrx)
    gd = torch.from_numpy(gd).long()
    px = torch.from_numpy(px).long()
    rx = torch.from_numpy(rx).float()
    len_x = torch.from_numpy(len_x).long()
    nrx = torch.from_numpy(nrx).float()
    print("processed finish! zeros:", total)
    print(gd.size(), px.size(), rx.size(), len_x.size(), nrx.size())
    return TensorDataset(px, rx, len_x, nrx, gd)


def process_raw_x(raw_x, n_past, n_inpaint, n_future):
    raw_px, raw_rx, raw_len_x, raw_nrx, raw_gd = raw_x
    past_px = raw_px[:, :n_past, :]
    inpaint_px = raw_px[:, n_past:n_past + n_inpaint, :]
    future_px = raw_px[:, n_future:, :]
    past_rx = raw_rx[:, :n_past, :]
    inpaint_rx = raw_rx[:, n_past:n_past + n_inpaint, :]
    future_rx = raw_rx[:, n_future:, :]
    past_len_x = raw_len_x[:, :n_past]
    inpaint_len_x = raw_len_x[:, n_past:n_past + n_inpaint]
    future_len_x = raw_len_x[:, n_future:]
    past_nrx = raw_nrx[:, :n_past, :]
    inpaint_nrx = raw_nrx[:, n_past:n_past + n_inpaint, :]
    future_nrx = raw_nrx[:, n_future:, :]
    past_gd = raw_gd[:, :n_past, :]
    inpaint_gd = raw_gd[:, n_past:n_past + n_inpaint, :]
    future_gd = raw_gd[:, n_future:, :]
    re = [
        past_px, past_rx, past_len_x, past_nrx, past_gd,
        inpaint_px, inpaint_rx, inpaint_len_x, inpaint_nrx, inpaint_gd,
        future_px, future_rx, future_len_x, future_nrx, future_gd,
    ]
    return re


def get_acc(recon, gd):
    recon = recon.cpu().detach().numpy()
    gd = gd.cpu().detach().numpy()
    return np.sum(recon == gd) / recon.size


def preprocessing(data, measure_len):
    for d in data:
        if 'raw' in d.keys():
            del d['raw']
    # process rhythm and pitch tokens
    split_size = 24
    new_data = []
    for i, d in enumerate(data):
        d = np.array(d["notes"])
        ds = np.split(d, list(range(split_size, len(d), split_size)))
        data = []
        for sd in ds:
            if len(sd) != split_size:
                continue
            q, k = extract_note(sd)
            if k == 0:
                continue
            s = extract_rhythm(sd)
            data.append([sd, q, s, k])
        new_data.append(data)
        if i % 1000 == 0:
            print("processed:", i)
    # extract each measure in each song
    length = int(len(new_data[0]) / measure_len)
    res = []
    for i in range(length):
        res.append(np.array(new_data[0][measure_len * i:measure_len * (i + 1)]))
    return res


def build_single_dataset(validate_set):
    validate_loader = DataLoader(
        dataset=processed_data_tensor(validate_set),
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    validate_data = []
    for i, d in enumerate(validate_loader):
        validate_data.append(d)
    return validate_data


def load_vae():
    # load VAE model
    vae_model = SketchVAE(
        vae_input_dims, vae_pitch_dims, vae_rhythm_dims, vae_hidden_dims,
        vae_zp_dims, vae_zr_dims, vae_seq_len, vae_beat_num, vae_tick_num, 4000)
    dic = torch.load("XGeneration/task1_explicit/model_backup/sketchvae.pt")

    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    vae_model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        vae_model.cuda()
    else:
        print('Using: CPU')
    vae_model.eval()
    print(vae_model.training)
    return vae_model


def load_model_a4(vae_model, inpaint_len, total_len):
    # load SketchNet
    model = SketchNet(
        zp_dims, zr_dims,
        pf_dims, gen_dims, combine_dims,
        pf_num, combine_num, combine_head,
        inpaint_len, total_len,
        vae_model, True
    )
    dic = torch.load("XGeneration/task1_explicit/model_backup/sketchNet-stage"
                     "-1_4_measure.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    model.set_stage("sketch")

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')
    return model


def load_model_b4(vae_model, inpaint_len, total_len):
    # load SketchNet
    model = SketchNet(
        zp_dims, zr_dims,
        pf_dims, gen_dims, combine_dims,
        pf_num, combine_num, combine_head,
        inpaint_len, total_len,
        vae_model, True
    )
    dic = torch.load('XGeneration/task1_explicit/model_backup/sketchNet-stage'
                     '-1_12_measure.pt')
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    model.set_stage("sketch")

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')
    return model


def model_eval(model, inference_data, n_past, n_future, n_inpaint):
    output = []
    v_mean_loss = 0.0
    v_mean_acc = 0.0
    total = 0
    val = []
    val.append(inference_data)
    v_raw_x = process_raw_x(inference_data, n_past, n_inpaint, n_future)
    for k in range(len(v_raw_x)):
        v_raw_x[k] = v_raw_x[k].to(device=device, non_blocking=True)
    v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd, \
    v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd, \
    v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd = v_raw_x
    v_inpaint_gd_whole = v_inpaint_gd.contiguous().view(-1)
    v_past_x = [v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd]
    v_inpaint_x = [v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd]
    v_future_x = [v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd]

    model.eval()
    with torch.no_grad():
        v_recon_x, _, _, _ = model(v_past_x, v_future_x, v_inpaint_x)
        v_loss = F.cross_entropy(v_recon_x.view(-1, v_recon_x.size(-1)), v_inpaint_gd_whole, reduction="mean")
        v_acc = get_acc(v_recon_x.view(-1, v_recon_x.size(-1)).argmax(-1), v_inpaint_gd_whole)
        v_mean_loss += v_loss.item()
        v_mean_acc += v_acc
        v_result = v_recon_x.argmax(-1)
    total += 1
    output.append(
        {
            "past": v_past_gd.cpu().detach().numpy(),
            "future": v_future_gd.cpu().detach().numpy(),
            "inpaint": v_result.cpu().detach().numpy(),
            "gd": v_inpaint_gd.cpu().detach().numpy(),
            "acc": v_acc,
            "nll": v_loss.item()
        }
    )
    return output


def append_notes(output, section, res):
    for note in output[0][section]:
        for cc in note:
            for ccc in cc:
                res.append(ccc)


def change_tempo(midi):
    mid = mido.MidiFile(midi)
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            msg.tempo = 750000


def inference(midi_path):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    change_tempo(midi_path)
    process_note_time(midi_path)
    min_step = 1 / 12

    ml = MIDI_Loader("Irish", minStep=min_step)
    ml.load_single_midi(midi_path)
    data = ml.processed_all()
    print(data)
    print(len(data[0]['notes']))

    # load vae model
    sketch_vae_model = load_vae()

    # load SketchNet for a4
    model_a4 = load_model_a4(sketch_vae_model, inpaint_len=2, total_len=4)
    model_a4.set_stage("sketch")

    inference_data = build_single_dataset(preprocessing(data, measure_len=4))
    try:
        output_a = model_eval(model_a4, inference_data[0], n_past=1, n_future=3, n_inpaint=2)
    except IndexError:
        print("Piece too short! Try to include more notes!")
        exit()
    output_a4 = model_eval(model_a4, inference_data[0], n_past=1, n_future=3, n_inpaint=2)
    append_notes(output_a4, "past", res)
    append_notes(output_a4, "inpaint", res)
    append_notes(output_a4, "future", res)

    # load SketchNet for b4
    notes = list(np.tile(np.array(data[0]['notes']), 3))
    new_data = data
    new_data[0]['notes'] = notes

    model_b4 = load_model_b4(sketch_vae_model, inpaint_len=4, total_len=12)
    model_b4.set_stage("sketch")
    inference_data = build_single_dataset(preprocessing(new_data, measure_len=12))
    output_b4 = model_eval(model_b4, inference_data[0], n_past=4, n_future=8, n_inpaint=4)

    append_notes(output_b4, "inpaint", res)

    # render midi
    data = {'notes': res}
    m = MIDI_Render("Irish", minStep=min_step)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = "temporary/inpaint_" + time + ".mid"
    m.data2midi(data, output=output_path)

    msg_list = []
    # inpaint length: 8 bars
    new_mid = MidiFile(output_path)
    for msg in new_mid.tracks[1]:
        if not msg.is_meta:
            msg_list.append(msg)
    # original length: 4 bars before
    original_mid = MidiFile(midi_path)
    for msg in original_mid.tracks[1]:
        if not msg.is_meta:
            msg_list.append(msg)
    # combined length: 16 bars in total
    for msg in msg_list[:-2]:
        original_mid.tracks[1].append(msg)
    save = "temporary/demo_" + time + ".mid"
    original_mid.save(save)
    return save

