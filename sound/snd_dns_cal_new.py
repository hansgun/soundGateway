import os, sys
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

HOME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
MODULE_DIR = 'util'
sys.path.append(os.path.join(HOME_PATH, MODULE_DIR)) # setup sys dir
from logs import paiplog

NEIGHBOR_SIZE = 9
SEARCH_SIZE = 128

def normalization(test_sound_2d : np.ndarray ) -> int :
    min_val, max_val = np.min(test_sound_2d).astype(np.int32), np.max(test_sound_2d).astype(np.int32)
    return ((test_sound_2d - min_val) / (max_val - min_val) * 255).astype(int)


def reshape_snd_data(test_sound_2d_norm : np.ndarray ) -> list :
    import math
    DIMEN_N = int(round(math.sqrt(test_sound_2d_norm.shape[1]), 0) - 1)
    DIMEN_N_2 = DIMEN_N ** 2
    ND_LIST = []

    for x in test_sound_2d_norm:
        ND_LIST.append(x[:DIMEN_N_2].reshape(-1, DIMEN_N))
    return ND_LIST


def get_pages(test_sound_2d , search_size : int = SEARCH_SIZE, neighbor_size: int = NEIGHBOR_SIZE) :
    """
    전체 matrix를 계산할 sub-matrix 로 분할하여 그 리스트를 return 하는 함수
    """
    ND_LIST = []

    # calcluate n*n 의 갯수
    size_x, size_y = test_sound_2d[0].shape
    div_ = search_size + neighbor_size - 1
    ind_x, ind_y = size_x // div_, size_y // div_
    # cent_p = (div_ -1, div_ -1)
    result_list = []

    # slicing
    for x in test_sound_2d:
        for i in range(ind_x):
            for j in range(ind_y):
                ND_LIST.append(x[i * div_:(i + 1) * div_, j * div_:(j + 1) * div_])
        result_list.append(ND_LIST)
        ND_LIST = []
    # print(self.ND_LIST)
    return np.asarray(result_list)


@njit
def distance_matrix(mat_x , search_win  , search_size : int , neighbor_size : int ) :
    """
    :param mat_x: neighborhood window maxrix
    :param search_win: search window matrix
    :param search_size:
    :param neighbor_size: size
    :return:
    todo :: 결과 검증 하기
    """
    out = np.empty((search_size, search_size))
    for x_in in range(search_size):
        for y_in in range(search_size):
            out[x_in, y_in] = np.sqrt(
                np.sum((mat_x[x_in:x_in + neighbor_size, y_in:y_in + neighbor_size] - search_win) ** 2))
    return out.reshape(search_size, search_size, 1)


def cal_dns_mat(sliced_array , per_length : int , search_size : int =SEARCH_SIZE, neighbor_size : int =NEIGHBOR_SIZE) :
    """
    matrix array에 대한 dns 계산하여 np array (SEARCH_SIZE X SEARCH_SIZE X len(ND_LIST)) 를 return
    """

    # cent of matrix position
    CENT_P : tuple(int,int) = (search_size // 2 + neighbor_size // 2, search_size // 2 + neighbor_size // 2)

    # result array
    result_mat = np.array([])
    return_result = np.zeros((search_size, search_size, sliced_array.shape[0]))
    # for phase
    for ind_mat, x in enumerate(sliced_array):  # number of nd_array
        # if ind_mat % 100 == 0 : print('{} of {}'.format(ind_mat,len(paged_norm_sliced)))
        for ind_inner, mat_x in enumerate(x):
            search_win = mat_x[CENT_P[0] - (neighbor_size // 2):CENT_P[0] + (neighbor_size // 2) + 1,
                         CENT_P[1] - (neighbor_size // 2):CENT_P[1] + (neighbor_size // 2) + 1].copy()
            if ind_inner == 0:
                result_mat = distance_matrix(mat_x, search_win, search_size, neighbor_size)
            else:
                result_mat = np.concatenate(
                    (result_mat, distance_matrix(mat_x, search_win, search_size, neighbor_size)), axis=2)

        return_result[:, :, ind_mat] = np.mean(result_mat, axis=2)
    # get a mean value of each cell finally
    return return_result

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


if __name__ == "__main__":
    import time
    import sys
    import snd_loader

    HOME_PATH = './'
    # HOME_PATH = '/Users/hansgun/Documents/code/python/sound'
    # DATA_DIR = 'data'
    # MODULE_DIR = 'farms'
    OUTPUT_DIR = 'out'
    sys.path.append(HOME_PATH)


    # check exec time start

    start_time = time.time()

    ''' sys.argvs
    1. directory name of sound file
    2. file name
    3. dns size ... (seconds)
    4. sliding window duration... (seconds)
    '''
    # 1. read wavefile
    #wav_file_str = os.path.join(sys.argv[1] + '/' + sys.argv[2]+'.wav')
    if len(sys.argv) < 2 : raise Exception("Not enough number of params")
    checkExtender = os.path.splitext(sys.argv[2])[1]

    print(f"checkExtender : {checkExtender}")

    if not checkExtender :
        wav_file_str = os.path.join(sys.argv[1] + '/' + sys.argv[2]+'.flac')
    else :
        wav_file_str = os.path.join(sys.argv[1] + '/' + sys.argv[2])
    #print(wav_file_str)

    # 2. generate sound list... from parameters...
    samplerate, sliced = snd_loader.snd_loader(wav_file_str, 1, 1).get_snd_df()

    # 3. normalization
    norm_sliced = normalization(sliced)

    # 4. reshape
    reshaped = reshape_snd_data(np.array(sliced))

    # 5. list divide by neighbor_size & search_size. to pages
    paged_norm_sliced = get_pages(reshaped)

    # 6. calculate dns matrix
    result = cal_dns_mat(np.asarray(paged_norm_sliced), per_length=len(paged_norm_sliced[0]))

    #print('elapsed : ', time.time() - start_time)

    # 7. save result to pickle data format
    # import pickle
    #
    # with open(os.path.join(HOME_PATH,OUTPUT_DIR,'data_real_wave_680.pickle'), 'wb') as f:
    #    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    # 8. ploting
    # plt.imshow(result[:, :, 10], cmap='gray')
    # 8. create directory 
    createFolder(sys.argv[1]+'/result_image/')

    #print(f'result shape : {result.shape}')
    for i in range(result.shape[2]) : 
        fig_name =  sys.argv[1] + '/result_image/' + sys.argv[2] + '_' + str(i) +'.png'
        plt.imsave(fig_name, result[:,:,i], cmap='gray')

        score_temp = str(np.round(np.mean(result[:,:,i])/1000,2))
        print(fig_name + '/' + score_temp)

    # AE
    os.system(f"python3 AE_predict.py --path {sys.argv[1]+'/result_image/'} --fileName {sys.argv[2]}")

