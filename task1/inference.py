import numpy as np
from torch.nn import functional as F
from SketchVAE.sketchvae import SketchVAE
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from SketchNet.sketchnet import SketchNet
from utils.helpers import *
from loader.dataloader import MIDI_Loader, MIDI_Render

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
# Todo: change the following parameters to fit in the customized dataset
inpaint_len = 4
seq_len = 16
total_len = 16
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


# load data from Midis, because bpm = 120，so one beat time = 60 / 120 = 0.5
# And in 4/4 we divide 4 beat to 24 step/frames, each will be 0.5 * 4 / 24  = 0.5 / 6 sec
# Todo: change minStep for our dataset

def preprocessing(data):
    for d in data:
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
    length = int(len(new_data[0]) / 16)
    res = []
    for i in range(length):
        res.append(np.array(new_data[0][16 * i:16 * (i + 1)]))
    print(res)

    validate_set = res
    print(validate_set)
    print(validate_set[0][0])
    print(len(validate_set[0][1]))
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
    print(len(validate_data))
    return validate_data


def load_vae():
    # load VAE model
    vae_model = SketchVAE(
        vae_input_dims, vae_pitch_dims, vae_rhythm_dims, vae_hidden_dims,
        vae_zp_dims, vae_zr_dims, vae_seq_len, vae_beat_num, vae_tick_num, 4000)
    dic = torch.load("model_backup/sketchvae-loss_0.04306925100494333_acc_0.9972101456969356_epoch_26_it_174824.pt")

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


def load_model():
    # load SketchNet
    model = SketchNet(
        zp_dims, zr_dims,
        pf_dims, gen_dims, combine_dims,
        pf_num, combine_num, combine_head,
        inpaint_len, total_len,
        vae_model, True
    )
    dic = torch.load(
        save_path + "sketchNet-stage-1loss_0.5944014993628857_acc_0.8705233683628317_epoch_90_it_111870.pt")
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


def model_eval(inference_data):
    # sketch parameters
    n_past = 6
    n_future = 10
    n_inpaint = 4
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
    print(output)
    return output


if __name__ == '__main__':

    # load midi
    # Todo: change the data loader
    midi_path = "../data/IrishFolkSong/session/sessiontune10.mid"
    ml = MIDI_Loader("Irish", minStep=0.5 / 6)
    ml.load_single_midi(midi_path)
    data = ml.processed_all()
    validate_data = preprocessing(data)
    # load vae model
    vae_model = load_vae()
    # load SketchNet
    model = load_model()
    model.set_stage("sketch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(vae_model.training)
    inference_data = validate_data[0]
    output = model_eval(inference_data)

    # render midi
    m = MIDI_Render("Irish", minStep=0.5 / 6)
    print(output[0]["gd"])
    output[0]["notes"] = output[0]["gd"]
    res = []
    for c in output[0]["gd"]:
        print("____")
        for cc in c:
            for ccc in c:
                for cccc in ccc:
                    res.append(cccc)

    data = {'notes': res}
    m.data2midi(data, output="test.mid")
