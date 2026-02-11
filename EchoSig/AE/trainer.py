import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import collections
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# from AE.utility import create_activation
from EchoSig.AE.models import AELoss,VAELoss
import numpy as np
from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler

sys.path.append('../')
def dataloader_split(X,test_size,seed,batch_size):
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(torch.tensor(X_train,dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
class LossHistory:
    def __init__(self,loss_name):
        self.history = []
        self.loss_name=loss_name
    def record(self, loss, recon_loss=None, kld_loss=None):
        if self.loss_name == 'AE':
            self.history.append({'loss': loss})
        elif self.loss_name == 'VAE':
            self.history.append({'loss': loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss})
    def average(self,epoch,beta, scaler):
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        for entry in self.history:
            total_loss += entry['loss']
            if 'recon_loss' in entry and 'kld_loss' in entry:
                total_recon_loss += entry['recon_loss']
                total_kld_loss += entry['kld_loss']
        avg_loss = total_loss / scaler
        avg_recon_loss = total_recon_loss  / scaler
        avg_kld_loss = total_kld_loss  / scaler

        # Update self.history with the average values
        self.history = [{'epoch': epoch,
                         'beta': beta,
                         'loss': avg_loss}]
        if total_recon_loss > 0 or total_kld_loss > 0:
            self.history[0]['recon_loss'] = avg_recon_loss
            self.history[0]['kld_loss'] = avg_kld_loss            
    def convert_list(self):

        l={key:[] for key in self.history[0].keys()}
        for itm in self.history:
            for key in self.history[0].keys():
                l[key].append(itm[key])
        return l
def append_loss_histories(history1, history2):
    if isinstance(history1, LossHistory) and isinstance(history2, LossHistory):
        history1.history.extend(history2.history)
    else:
        raise ValueError("Both arguments must be instances of LossHistory")

 


class Trainer(object):
    def __init__(self,
                 model,
                #  loss,
                 X,
                 seed:int=42,
                 test_size:float=0.1,
                 batch_size:int=256,
                 lr:float=1e-3,
                 weight_decay:float=0.,
                 l1_lambda:float=0.,
                 max_epoch = 200,
                #  warmup_epochs=30,
                #  max_beta_epoch=50,
                 beta = 1.0,
                #  mode='linear',
                #  cycle=None,
                 early_stopping=True,
                 tol=0.,  
                 patience=30,
                 device=None,
                 ):
        '''
        Trainer for pretrain model.
        Parameters:
        model:
            A pytorch model defined in "models.py"
            e.g., AutoEncoder()
        X:
            Feature matrix. mxn numpy array.
                a standarized logorithmic data (i.e., zero mean, unit variance)
        test_size:
            fraction of testing/validation data size. default: 0.2
        batch_size:
            batch size.
        lr:
            learning rate.
        weight_decay:
            L2 regularization.
        l1_lambda: 
            L1 regularization        
        max_epoch:
            maximal number of epoches
        seed:
            random seed.

        beta: 
            coefficient of reconstruction errors and KL divergence
        Early Stopping Parameters:
            early_stopping:
                whether to use early stopping
            tol:
                tolerence to determine if validation error is improved
            patience:
                number of epochs to have early stop if validation error is not improved
        '''
        # self.args = args
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        self.model = model
        self.model.to(self.device)
        if model.model_name == 'VAE':
            self.loss = VAELoss()
        elif model.model_name == 'AE':
            self.loss = AELoss()
        # if self.model.decoder_type=='normal':
        self.X=X
        self.train_loader,self.test_loader=\
            dataloader_split(X,test_size,seed,batch_size)
        self.batch_size=batch_size
        self.lr=lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.max_epoch = max_epoch
        # self.warmup_epochs=warmup_epochs
        # self.max_beta_epoch=max_beta_epoch
        # self.max_beta=max_beta
        # self.mode = mode
        # self.cycle = cycle
        # if cycle is not None:
        #     self.cycle=cycle*self.train_loader.__len__() 
        self.seed=seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        # self.beta_scheduler = BetaScheduler(max_beta=self.max_beta,
        #                                     max_epoch=self.max_epoch,
        #                                     warmup_epochs=self.warmup_epochs,
        #                                     max_beta_epoch=self.max_beta_epoch,
        #                                     mode = self.mode,
        #                                     cycle=self.cycle)
        self.beta=beta
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
    def train_step(self,epoch):
        self.model.train()
        # beta = self.beta_scheduler.get_beta(current_epoch=epoch,total_iter=self.total_iter)
        train_loss = 0
        train_history = LossHistory(self.loss.loss_name)
        for batch_idx, (data,) in enumerate(self.train_loader):
            data = data.to(self.device)
            # scale_factor = scale_factor.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data,)
            loss_ = self.loss(data,output)
            # beta = self.beta_scheduler.get_beta(current_epoch=epoch,total_iter=self.total_iter)
            # print(str(epoch)+' ' +str(self.total_iter)+ ' '+str(beta))
            if self.loss.loss_name=='AE':
                loss=loss_
                train_history.record(loss.item())
            elif self.loss.loss_name=='VAE':
                recon_loss, kld_loss=loss_[0],loss_[1]
                loss = recon_loss+self.beta*kld_loss
                train_history.record(loss.item(),recon_loss.item(),kld_loss.item())
            # if epoch>self.warmup_epochs:
            #     self.total_iter+=1
            if self.l1_lambda>0:
                l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
                loss = loss + self.l1_lambda * l1_penalty
            loss.backward()
            train_loss += loss.item()#*data.shape[0]
            self.optimizer.step()
        train_loss=train_loss / len(self.train_loader.dataset)
        train_history.average(epoch,self.beta,len(self.train_loader.dataset))
        return train_loss , train_history
    
    def test(self,epoch):
        self.model.eval()
        test_loss = 0
        test_history = LossHistory(self.loss.loss_name)

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data,)
                loss_ = self.loss(data,output)
                if self.loss.loss_name=='AE':
                    loss=loss_
                    test_history.record(loss.item())
                elif self.loss.loss_name=='VAE':
                    recon_loss, kld_loss=loss_[0],loss_[1]
                    loss = recon_loss+self.beta*kld_loss
                    test_history.record(loss.item(),recon_loss.item(),kld_loss.item())
                test_loss += loss.item()#*data.shape[0]
        test_loss /= len(self.test_loader.dataset)
        test_history.average(epoch,self.beta,len(self.test_loader.dataset))
        return test_loss, test_history
    def train(self,):
        # self.total_iter=0.
        # self.model.train()
        self.history={}
        self.history['train']=LossHistory(self.loss.loss_name)
        self.history['val']=LossHistory(self.loss.loss_name)
        best_val_error = float('inf')
        num_patience_epochs = 0
        for epoch in tqdm(range(self.max_epoch)):
            # beta = self.beta_scheduler.get_beta(current_epoch=epoch)
            self.epoch=epoch
            # Train for one epoch and get the training loss
            train_loss,train_itm = self.train_step(epoch)
            # Compute the validation error
            val_loss, val_itm = self.test(epoch)
            if epoch % 10==0:
                print(f"Epoch {epoch}: train loss = {train_loss:.4f}, val error = {val_loss:.4f}")
            append_loss_histories(self.history['train'],train_itm)
            append_loss_histories(self.history['val'],val_itm)

            if self.early_stopping:
                # Check if the validation error has decreased by at least tol
                if best_val_error - val_loss >= self.tol:
                    best_val_error = val_loss
                    num_patience_epochs = 0
                else:
                    num_patience_epochs += 1
                    # Check if we have exceeded the patience threshold
                if num_patience_epochs >= self.patience:
                    print(f"No improvement in validation error for {self.patience} epochs. Stopping early.")
                    break
        self.history['train'] = self.history['train'].convert_list()
        self.history['val'] = self.history['val'].convert_list()
        print(f"Best validation error = {best_val_error:.4f}")
        self.model.scaler.fit(self.model.encoder.emb(self.X))
        
        torch.cuda.empty_cache()

        
    