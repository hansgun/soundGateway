import os
from glob import glob

data_dir = './data/'

sub_dir = [x[0] for x in sorted(os.walk(data_dir))][1:]
wav_names = []

for sub_d in sub_dir :
    wav_names.extend(glob(sub_d+'/'+'*.wav'))

list_param = [('/'.join(x.split('/')[:-1]), os.path.splitext(x.split('/')[-1])[0]) for x in wav_names]

for x1, x2 in list_param :
    os.system(f"python3.9 snd_dns_cal_new.py {x1} {x2}")
