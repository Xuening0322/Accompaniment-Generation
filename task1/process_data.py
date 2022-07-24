import os
import mido
import shutil
from mido import Message, MidiFile, MidiTrack, MetaMessage


def change_tempo(filename, data_path, target_path):
    mid = mido.MidiFile(data_path + "\\" + filename)
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.type == 'set_tempo':
                print(msg)
                msg.tempo = 750000
                print(msg)

            new_track.append(msg)
    new_mid.save(target_path + "\\" + filename)


def get_filelist(dir, Filelist):
    newDir = dir

    if os.path.isfile(dir):

        # Filelist.append(dir)

        # # 若只是要返回文件文，使用这个

        Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码

            # if s == "xxx":

            # continue

            newDir = os.path.join(dir, s)

            get_filelist(newDir, Filelist)



    return Filelist


if __name__ == '__main__':
    folder = r"D:\Surf\AccoMontage-main\data files\dataset"
    list = get_filelist(r"D:\Surf\AccoMontage-main\data files\dataset", [])
    print(len(list))
    for e in list:
        change_tempo(e, r"D:\Surf\AccoMontage-main\data files\dataset", r"D:\Surf\AccoMontage-main\data files\dataset1")
