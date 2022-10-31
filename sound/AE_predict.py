import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import os, sys
from glob import glob
from PIL import Image
import fire 
from torch.utils.data import Dataset
import pickle

HOME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
MODULE_DIR = 'util'
sys.path.append(os.path.join(HOME_PATH, MODULE_DIR))
from PyDBconnector import PyDBconnector
from logs import paiplog

## models 
PATH = './models/'
MODEL_NAME = 'model_dns.pt'

learning_rate = 0.01
num_steps = 10000
batch_size = 512
lv_level = 5 # letent vector projection level
display_step = 1000
examples_to_show = 10

num_hidden_1 = 256
num_hidden_2 = 128
num_hidden_3 = 64
num_hidden_4 = 12
num_hidden_5 = 3

num_input = 784

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Sigmoid(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.Sigmoid(),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim4, hidden_dim3),
            nn.Sigmoid(),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Sigmoid(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size())
        return out

    def get_codes(self, x):
        # print(f"len : {len(x.view(x.size(0), -1))}")
        return self.encoder(x)

@paiplog
def _loadModel() :
    try :
        AE = torch.load(PATH + MODEL_NAME)  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
#         AE.load_state_dict(torch.load(PATH + 'model_layer_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

#         checkpoint = torch.load(PATH + 'all_layer.tar')  # dict 불러오기
#         AE.load_state_dict(checkpoint['model'])
        AE_loss = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        AE = AE.to(device)
        AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
        #AE_optimizer.load_state_dict(checkpoint['optimizer'])
    except :
        AE = AutoEncoder(128 * 128, 256, 128, 64, 32)
        AE_loss = nn.MSELoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        AE = AE.to(device)
        AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)

    return AE, AE_loss, AE_optimizer

@paiplog
def predictAE(test_loader: DataLoader) -> list:
    AE = torch.load(PATH + MODEL_NAME,map_location='cpu')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    #AE.load_state_dict(torch.load(PATH + 'model_layer_state_dict.pt'),map_location='cpu')  # state_dict를 불러 온 후, 모델에 저장

    #checkpoint = torch.load(PATH + 'all_layer.tar',map_location='cpu')  # dict 불러오기
    #AE.load_state_dict(checkpoint['model'],map_location='cpu')
    AE_loss = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AE = AE.to(device)
    AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
    #AE_optimizer.load_state_dict(checkpoint['optimizer'],map_location='cpu')

    y_true = []
    AE_y_pred = []

    idx = 0
    z_dim = 3
    cal_sap = lambda x: np.sqrt((x ** 2).sum())

    AE.eval()
    results_val = []
    for test_data in test_loader:
        # if idx > 0 : break
        for inner_idx, X in enumerate(test_data):
            X = X.to(device)
            #print(f"X size : {X.size()}, {X[0].size()}")
            # Forward Pass
            X_ = AE(X)

            lv = AE.get_codes(X.reshape(-1, ))
            #print(f"lv size : {lv.size()}, {lv[0].size()}")
            lv = lv.to(device="cpu").detach().numpy() 
            # recon = nn.MSELoss(X, X_)
            recon = (X_ - X).cpu().detach().numpy()
            recon = cal_sap(recon)
            # print("recon:", recon)
            # print(f"latent vector : {lv}")
            h1 = torch.sigmoid(AE.encoder[0](X.reshape(-1)))
            h2 = torch.sigmoid(AE.encoder[2](h1.reshape(-1)))
            h3 = torch.sigmoid(AE.encoder[4](h2.reshape(-1)))
            h4 = torch.sigmoid(AE.encoder[6](h3.reshape(-1)))
            #h5 = torch.sigmoid(AE.encoder[8](h4.reshape(-1)))

            X_X = AE(X_)

            h1_ = torch.sigmoid(AE.encoder[0](X_.reshape(-1)))
            h2_ = torch.sigmoid(AE.encoder[2](h1_.reshape(-1)))
            h3_ = torch.sigmoid(AE.encoder[4](h2_.reshape(-1)))
            h4_ = torch.sigmoid(AE.encoder[6](h3_.reshape(-1)))
            #h5_ = torch.sigmoid(AE.encoder[8](h4_.reshape(-1)))

            # print((h1-h1_).shape)
            # print((h2-h2_).shape)
            # print((h3-h3_).shape)
            # print((h4-h4_).shape)

            # 차이를 제곱함

            # cal_sap = lambda x : (x**2).sum()

            sap = np.array([cal_sap((h1 - h1_).cpu().detach().numpy()),
                            cal_sap((h2 - h2_).cpu().detach().numpy()),
                            cal_sap((h3 - h3_).cpu().detach().numpy()),
                            cal_sap((h4 - h4_).cpu().detach().numpy())])#,
                            #cal_sap((h5 - h5_).cpu().detach().numpy())])

            # sap = torch.cat([h1-h1_,h2-h2_,h3-h3_,h4-h4_], dim=[-1,0]).detach().numpy()
            # sap = (sap**2).sum()
            # _, _, t_v = np.linalg.svd(sap.reshape(1,-1) - sap.reshape(1,-1).mean(), full_matrices=False)

            # result_vector = np.concatenate([recon, lv.reshape(-1,1), (sap.reshape(1,-1) - sap.reshape(1,-1).mean()).reshape(-1,1)], axis=None)
            _, _, t_v = np.linalg.svd(sap.reshape(1, -1) - sap.reshape(1, -1).mean(), full_matrices=False)
            # print(f"result_vector : {result_vector}")
            # _, _, t_v = np.linalg.svd(result_vector.reshape(1,-1), full_matrices=False)
            # print(f"SAP : {sap}")
            # print(f"svd : {t_v}")
            if _ == 5:
                y_true.append(1)
            else:
                y_true.append(0)

            if AE_loss(X_, X).item() < 0.02:
                AE_y_pred.append(1)
            else:
                AE_y_pred.append(0)
            results_val.append(
                [recon, t_v, lv])
        idx += 1

    return results_val


class Img_Dataset(Dataset) : 
    def __init__(self, root, transform) : 
        self.file_list = fileList = glob(root + "*.png")
        self.transform = transform
        
    def __len__(self) : 
        return len(self.file_list)
    
    def __getitem__(self, index) : 
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        return img_transformed

trans = transforms.Compose([transforms.Resize((128, 128)),
                        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        transforms.Grayscale(),
                        transforms.ToTensor()])

@paiplog
def insertUpdateDatabase(dbConn, target_table='tb_ae_results', data_df='result_df', isInsert=True) :
    try:
        # [['fileName','recon_error','t_v','lv','svmResult','createDate']]
        print('insert results===========================')
        if isInsert : # insert new InDate
            for pdRow in data_df.iterrows():
                insert_string = f"insert into {target_table}(create_time, fileName, recon_error, t_v, lv, svmResult) " \
                                f"values('{pdRow[1]['createDate']}','{pdRow[1]['fileName']}',{pdRow[1]['recon_error']},'{pdRow[1]['t_v']}','{pdRow[1]['lv']}',{pdRow[1]['svmResult']})"
                dbConn.insert_to_db(insert_string)

            dbConn.close()
            return True
        else : 
            pass
    except:
        raise Exception('insert query error! check DB')


def main(path, fileName) : 
    return path, fileName 

if __name__ == '__main__' : 
    # Setup parameters ###############################################################################################################
    # 필요한 arguments. 
    # @param : path 
    # @parma : filename 
    path, fileName = fire.Fire(main)
    completePath = path + os.path.sep + fileName
    AE, AE_loss, AE_optimizer = _loadModel()
    saveToDB = True
    ####################################################################################################################################

    # AE test code ####################################################################################################################

    test_data = Img_Dataset(root=completePath, transform=trans)
    # sampling
    # test_data = Subset(test_data, list(range(0,100000)))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # list[ (recon, t_v, lv) ]
    results_list = predictAE(test_dataloader)
    

    result_df = pd.DataFrame(results_list, columns=['recon_error','t_v','lv'])

    ####################################################################################################################################



    # OC SVM ###########################################################################################################################
    with open('./models/oclf.pickle','br') as f : 
        oclf = pickle.load(f)
    X_test = list(zip(result_df.recon_error,result_df.t_v, result_df.lv))
    X_test = [[x[0], *x[1].flatten(),*x[2].flatten()] for x in X_test]

    y_pred = oclf.predict(X_test)

    result_df['svmResult'] = y_pred
    result_df['createDate'] = [x.split('_')[4][:12] for x in result_df.fileName] ## datetime
    ####################################################################################################################################


    ## insert results
    if saveToDB and len(result_df[result_df['svmResult'] == -1]) > 0 :
        dbConn = PyDBconnector()
        target_table = ''
        # DB Update
        insertUpdateDatabase(dbConn, target_table, result_df[result_df['svmResult'] == -1], isInsert=True)
