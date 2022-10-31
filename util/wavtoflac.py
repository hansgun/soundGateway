import os, sys, soundfile, argparse
from logs import paiplog

@paiplog
def convertwavtoflac(wav_path) :
    # read wav file
    path, exp_tail = os.path.splitext(wav_path)
    # raise Exception("test Exception")
    if exp_tail != '.wav' : raise Exception(f"{wav_path} is not a wav file")

    try :
        audio, sr = soundfile.read(wav_path)
    except :
        raise Exception(f"{wav_path} : failted to read wave file")

    # save
    try :
        soundfile.write(path + '.flac', audio, sr, 'PCM_16')
    except :
        raise Exception(f"{wav_path} : failted to save as a flac file")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="A wav file path to convert", type=str)
    parser.add_argument("-f", "--fileName", help="A fileName to convert", type=str)

    args = parser.parse_args()
    try :
        ret = convertwavtoflac(os.path.join(args.path, args.fileName))
        print(ret)
    except Exception as e:
        print(e)
