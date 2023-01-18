import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import os, pickle
from glob import glob
from PIL import Image
from sklearn.metrics import roc_auc_score
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:80% !important; }</style>"))

class Deep_SVDD(nn.Module):
    def __init__(self, z_dim=32):
        super(Deep_SVDD, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4096, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# data를 새롭게 representation하기 위한 AutoEncoder
class C_AutoEncoder(nn.Module):
    def __init__(self, z_dim=32):
        super(C_AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4096, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
    def encoder(self, x):
        
        # encoder 구조는 Deep SVDD와 완전히 동일한 구조를 가지고 있음
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
   
    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
    
    def pretrain(self):
        # Deep SVDD에 적용할 가중치 W를 학습하기 위해 autoencoder를 학습함
        ae = C_AutoEncoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        
        # 지정한 step마다 learning rate를 줄여감
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        # AE 학습
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                # 재구축 오차를 최소화하는 방향으로 학습함
                # AE 모델을 통해 그 데이터를 잘 표현할 수 있는 common features를 찾는 것이 목표임
                print(x_hat.shape, x.shape)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        
        # AE의 encoder 구조의 가중치를 Deep SVDD에 초기화하기 위함임
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        # 구조가 맞는 부분만 가중치를 load함
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters.pth')
    

    def set_c(self, model, dataloader, eps=0.1):
        
        # 구의 중심점을 초기화함
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        
        # AE의 학습을 마치고 그 가중치를 적용한 Deep SVDD를 학습함
        net = Deep_SVDD().to(self.device)
        
        if self.args.pretrain==True:
            state_dict = torch.load('weights/pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            # pretrain을 하지 않았을 경우 가중치를 초기화함
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        
        # early_stop
        train_loss_arr = []
        test_loss_arr = []
        
        best_test_loss = 99999999
        early_stop, early_stop_max = 0., 3. 
        
        
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            epoch_loss = 0.
            
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_loss_arr.append(total_loss / len(self.train_loader.dataset))
            
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.net = net
        self.c = c

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return 


class Img_Dataset(Dataset) : 
    def __init__(self, root, transform, labels=0) : 
        self.file_list = fileList = glob(root + '*.jpg')
        self.labels = [0] * len(self.file_list) if labels == 0 else labels
        self.transform = transform
        
    def __len__(self) : 
        return len(self.file_list)
    
    def __getitem__(self, index) : 
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        return img_transformed, self.labels[index]

def data_loader(train_dir, test_dir, batch_size, trans, test_labels) :
#     train_data_total = Img_Dataset(root=train_dir, transform=trans)
#     train_data = Subset(train_data_total, list(range(0,100000)))
    
#     total_size = len(train_data)

#     train_size = int(total_size * 0.8)
#     validation_size = int(total_size * 0.1)
#     test_size = total_size - train_size - validation_size

#     train_dataset, validation_dataset, test_dataset = random_split(train_data, [train_size, validation_size, test_size])
    train_data = Img_Dataset(root=train_dir, transform=trans)
    
    test_data = Img_Dataset(root=test_dir, transform=trans, labels=test_labels)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_dataloader, test_dataloader

def eval(net, c, dataloader, device):
    # ROC AUC score 계산

        scores = []
        labels = []
        net.eval()
        print('Testing...')
        with torch.no_grad():
            for x, y in dataloader:
                x = x.float().to(device)
                z = net(x)
                score = torch.sum((z - c) ** 2, dim=1)

                scores.append(score.detach().cpu())
                labels.append(y.cpu())
        labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
        print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
        return labels, scores

if __name__ == '__main__' : 
    # DNS
    BASE_PATH = '/home/centos/data/sound/ib/dns/normal/' # 실제는 ILT 임.. 2021년 11월 생성 파일
    TEST_PATH = '/home/centos/data/sound/ib/dns/abnormal/'

    # spectrogram
    # BASE_PATH = '/home/centos/data/sound/ib/1st/datas/jpg_1s/' 

    # mnist : 28 × 28 × 1
    y_size, x_size = 128, 128

    trans = transforms.Compose([transforms.Resize((y_size, x_size)),
                        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        transforms.Grayscale(),
                        transforms.ToTensor()])

    label_df = pd.read_csv('/home/centos/data/sound/ib/1st/datas/sound_labeling_1s.csv')
    label_df.columns = ['file_name','divided']
    label_df['divided'].value_counts()

    test_df = pd.DataFrame(glob(TEST_PATH + '/*.jpg'), columns=['file_name_raw'])
    test_df['file_name'] = [os.path.basename(str(x).replace('.wav','')) for x in test_df.file_name_raw]

    # test_df left outer join label_df
    joined = pd.merge(test_df['file_name'], label_df, on='file_name', how='left')
    joined['divided'].fillna(0., inplace=True)
    joined['divided'] = joined['divided'].astype(int)
    
    # 파라미터 지정
    class Args:

        num_epochs=150
        num_epochs_ae=150
        patience=50
        lr=1e-4
        weight_decay=0.5e-6
        weight_decay_ae=0.5e-3
        lr_ae=1e-4
        lr_milestones=[50]
        batch_size=200
        pretrain=False
        latent_dim=32
        normal_class=0
        
        
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data_loader(BASE_PATH, TEST_PATH, args.batch_size, trans, joined['divided'].to_list())

    isTrain=True
    if isTrain : 
        ### train code ###############################################################################
        deep_SVDD = TrainerDeepSVDD(args, data, device)

        # AE pretrain
        if args.pretrain:
            deep_SVDD.pretrain()

        deep_SVDD.train()

        torch.save(deep_SVDD,'torchModel/deep_SVDD_spectrum_v0.1.pth')

        ### train code ###############################################################################
    else : 
        ### load model ###############################################################################
        deep_SVDD = torch.load('torchModel/deep_SVDD_v0.1.pth',map_location={'cuda:0':'cuda:2'})
        labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)

        for x, y in zip(labels,scores) :
           print(f"label : {x}, score : {y}")
        result_df = pd.DataFrame({'labels' : labels, 'scores' : scores})

        with open('./deep_svdd_results.pickle','bw') as f :
            pickle.dump(result_df, f)

        ### load model ###############################################################################



    

