import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt
import os

learning_rate = 0.01
num_steps = 10000
batch_size = 512
lv_level = 5 # letent vector projection level
display_step = 1000
examples_to_show = 10

num_hidden_1 = 256
num_hidden_2 = 128
num_hidden_3 = 64
num_hidden_4 = 32
num_hidden_5 = 16

num_input = 784
PATH = './'
trans = transforms.Compose([transforms.Resize((128, 128)),
                            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            transforms.Grayscale(),
                            transforms.ToTensor()])


class AutoEncoder_backup(nn.Module):
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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Sigmoid(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.Sigmoid(),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.Sigmoid(),
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim5, hidden_dim4),
            nn.Sigmoid(),
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


def _loadModel() :
    try :
        AE = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
        AE.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

        checkpoint = torch.load(PATH + 'all.tar')  # dict 불러오기
        AE.load_state_dict(checkpoint['model'])
        AE_loss = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        AE = AE.to(device)
        AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
        AE_optimizer.load_state_dict(checkpoint['optimizer'])
    except :
        AE = AutoEncoder(128 * 128, 256, 128, 64, 32)
        AE_loss = nn.MSELoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        AE = AE.to(device)
        AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)

    return AE, AE_loss, AE_optimizer


def predictAE(test_loader: DataLoader) -> list:
    AE = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    AE.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    checkpoint = torch.load(PATH + 'all.tar')  # dict 불러오기
    AE.load_state_dict(checkpoint['model'])
    AE_loss = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AE = AE.to(device)
    AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
    AE_optimizer.load_state_dict(checkpoint['optimizer'])

    y_true = []
    AE_y_pred = []

    idx = 0
    z_dim = 32
    cal_sap = lambda x: np.sqrt((x ** 2).sum())

    results_val = []
    AE.eval()
    results_val = []
    for test_data, _ in test_loader:
        # if idx > 0 : break
        for inner_idx, X in enumerate(test_data):
            X = X.to(device)
            # print(f"X size : {X.size()}, {X[0].size()}")
            # Forward Pass
            X_ = AE(X)

            lv = AE.get_codes(X.reshape(-1, ))
            lv = torch.sigmoid(nn.Linear(16, z_dim)(lv.to(device="cpu"))).detach().numpy()
            # recon = nn.MSELoss(X, X_)
            recon = (X_ - X).cpu().detach().numpy()
            recon = cal_sap(recon)
            # print("recon:", recon)
            # print(f"latent vector : {lv}")
            h1 = torch.sigmoid(AE.encoder[0](X.reshape(-1)))
            h2 = torch.sigmoid(AE.encoder[2](h1.reshape(-1)))
            h3 = torch.sigmoid(AE.encoder[4](h2.reshape(-1)))
            h4 = torch.sigmoid(AE.encoder[6](h3.reshape(-1)))
            h5 = torch.sigmoid(AE.encoder[8](h4.reshape(-1)))

            X_X = AE(X_)

            h1_ = torch.sigmoid(AE.encoder[0](X_.reshape(-1)))
            h2_ = torch.sigmoid(AE.encoder[2](h1_.reshape(-1)))
            h3_ = torch.sigmoid(AE.encoder[4](h2_.reshape(-1)))
            h4_ = torch.sigmoid(AE.encoder[6](h3_.reshape(-1)))
            h5_ = torch.sigmoid(AE.encoder[8](h4_.reshape(-1)))

            # print((h1-h1_).shape)
            # print((h2-h2_).shape)
            # print((h3-h3_).shape)
            # print((h4-h4_).shape)

            # 차이를 제곱함

            # cal_sap = lambda x : (x**2).sum()

            sap = np.array([cal_sap((h1 - h1_).cpu().detach().numpy()),
                            cal_sap((h2 - h2_).cpu().detach().numpy()),
                            cal_sap((h3 - h3_).cpu().detach().numpy()),
                            cal_sap((h4 - h4_).cpu().detach().numpy()),
                            cal_sap((h5 - h5_).cpu().detach().numpy())])

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
                [os.path.basename(test_loader.dataset.samples[idx * batch_size + inner_idx][0]), recon, t_v, lv])
        idx += 1

    return results_val


def _train(model, Loss, optimizer, num_epochs, train_loader, device, saveModel=True):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.

    for epoch in range(num_epochs):

        epoch_loss = 0.
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()

            # Forward Pass
            model.train()
            outputs = model(batch_X)
            train_loss = Loss(outputs, batch_X)
            epoch_loss += train_loss.data

            # Backward and optimize
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))

        if epoch % 10 == 0:
            model.eval()

            test_loss = 0.

            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)

                # Forward Pass
                outputs = model(batch_X)
                batch_loss = Loss(outputs, batch_X)
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss,
                                                                                      test_loss))
            else:
                early_stop += 1
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss,
                                                                                    test_loss))

        if early_stop >= early_stop_max:
            break

    torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
    # torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
    # torch.save({
    #     'model': model.state_dict(),
    #     'optimizer': model.state_dict()
    # }, PATH + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
    return True

def train(train_dir, num_epochs = 500) :
    train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AE, AE_loss, AE_optimizer = _loadModel()
    returnVal = _train(AE, AE_loss, AE_optimizer, num_epochs, train_loader, device)
    print(f"train model : {returnVal}")



if __name__ == '__main__':

    # train code #######################################################################################################
    # train('/home/gmos/sound/data/dnsfile')
    ####################################################################################################################


    # test code ########################################################################################################
    # #test_data = torchvision.datasets.ImageFolder(root='./sound_train/controlGroup/', transform  = trans)
    test_data = torchvision.datasets.ImageFolder(root='/home/gmos/sound/data/dnsfile', transform=trans, )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # list[ (recon, t_v, lv) ]
    results_list = predictAE(test_loader)

    pd.DataFrame(results_list, columns=['recon_error','t_v','lv']).to_csv('./results.csv')
    ####################################################################################################################