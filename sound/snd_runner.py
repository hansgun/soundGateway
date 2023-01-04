import os, pickle
from glob import glob
import snd_loader
import snd_dns_cal_new
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

data_dir = '../data/ILT/'

sub_dir = [x[0] for x in sorted(os.walk(data_dir))][1:]
wav_names = []

for sub_d in sub_dir :
    wav_names.extend(glob(sub_d+'/'+'*.wav'))

list_param = [('/'.join(x.split('/')[:-1]) + '/', x.split('/')[-1]) for x in wav_names]


df = pd.DataFrame([[None, None]], columns = ['fileName', 'score'])


for i, (x1, x2) in enumerate(list_param) :
    wav_file_str = os.path.join(x1 + '/' + x2)
    if i % 100 == 0 : print(f"i : {i}/{len(list_param)}")
    # if i > 2 : break
    samplerate, sliced = snd_loader.snd_loader(wav_file_str, 1, 1).get_snd_df()

    # 4. reshape
    reshaped = snd_dns_cal_new.reshape_snd_data(np.array(sliced))
    # 5. list divide by neighbor_size & search_size. to pages
    # paged_norm_sliced = get_pages(reshaped)

    # 6. calculate dns matrix
    result = snd_dns_cal_new.cal_dns_mat(np.asarray(reshaped))

    # 7. normalization
    result = snd_dns_cal_new.normalization(result)
    #print('elapsed : ', time.time() - start_time)

    # 7. save result to pickle data format
    # import pickle
    #
    # with open(os.path.join(HOME_PATH,OUTPUT_DIR,'data_real_wave_680.pickle'), 'wb') as f:
    #    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    # 8. ploting
    
    # 8. create directory 
    snd_dns_cal_new.createFolder('../output/ilt_20211108_20211109/')
    #print(f'result shape : {result.shape}')
    for i in range(result.shape[0]) : 
        # fig_name =  sys.argv[1] + '/result_image/' + sys.argv[2] + '_' + str(i) +'.png'
        fig_name =  '../output/ilt_20211108_20211109/' + x2.split('.')[0] + '_' + str(i).zfill(2) +'.jpg'

        plt.imsave(fig_name, result[i,:,:], cmap='gray')

        score_temp = str(np.round(np.mean(result[:,:,i]),2))
        df.loc[len(df)] = [x2.split('.')[0] + '_' + str(i).zfill(2) +'.jpg',score_temp]
        # print(fig_name + '/' + score_temp)


with open('../output/ilt_1108_1109.pickle','wb') as f : 
    pickle.dump(df,f)
