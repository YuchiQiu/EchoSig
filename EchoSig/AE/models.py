import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import BaseAEConfig
import os
from typing import Optional, List
from EchoSig.utility import create_activation
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self,     
                 layers_list: List[int] ,
                 dropout: float = 0.
                 ,norm: bool =False,
                 activation: str = 'leakyrelu',
                 last_act: bool = False):
        """This class create a MLP network
        Args:
            layers_list (List[int]): A list of sizes of hidden layers 
                For example, [5,2,5] provides a 3 layers network including both input and output layers. 

            dropout (float, optional): Dropout rate at each layer. Defaults to 0..
            
            norm (bool, optional): If bacth normalization is included in each layer. Defaults to False.
            
            activation (str, optional): activation function. 
                Refer options in `utility.create_activation()`
                Defaults to 'relu'.
            
            last_act (bool, optional): If the last (output layer) includes activation function. 
                Defaults to False.
        """
        super(MLP, self).__init__()

        layers=nn.ModuleList()
        if activation is not None:
            activation=create_activation(activation)
        assert len(layers_list)>=2, 'no enough layers'
        
        for i in range(len(layers_list)-2):
            layers.append(nn.Linear(layers_list[i],layers_list[i+1]))
            if norm:
                layers.append(nn.BatchNorm1d(layers_list[i+1]))
            if activation is not None:
                layers.append(activation)
            if dropout>0.:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layers_list[-2],layers_list[-1]))
        if norm:
            layers.append(nn.BatchNorm1d(layers_list[-1]))
        if isinstance(last_act, bool):
            if last_act:
                if activation is not None:
                    layers.append(activation)
        elif isinstance(last_act, str):
            layers.append(create_activation(last_act))
            
        if dropout>0.:
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        for layer in self.network:
            x=layer(x)
        return x
    
#####
# Encoder
#####

class Encoder(nn.Module):
    def __init__(self,     
                 layers_list: List[int] ,
                 dropout: float = 0.,
                 norm: bool =False,
                 activation: str = 'leakyrelu'):
        """This is the Encoder without variational inference

        Args:
            layers_list (List[int]): _description_
            dropout (float, optional): _description_. Defaults to 0..
            norm (bool, optional): _description_. Defaults to False.
            activation (str, optional): _description_. Defaults to 'relu'.
        """
        super(Encoder,self).__init__()
        self.encoder=MLP(layers_list,
                         dropout,
                         norm,
                         activation,
                         last_act=False)
        self.encoder_layers = layers_list
        self.dropout = dropout
        self.norm = norm
        self.activation = activation
    def forward(self,x):
        z=self.encoder(x)
        return z
    def emb(self,x,eval=True):
        """ Get latent space representation

        Args:
            x: tensor or numpy.array
            eval (bool): if this is in evaluation mode. 
        """

        if isinstance(x, np.ndarray):
            x=torch.tensor(x,dtype=torch.float32)
            return_numpy = True
        else:
            return_numpy = False
        if eval:
            self.eval()
            x = x.to(next(self.parameters()).device)
        z=self.encoder(x)
        if eval:
            if return_numpy:
                z=z.cpu().detach().numpy()
        return z

class EncoderVAE(nn.Module):
    def __init__(self,     
                 layers_list: List[int] ,
                 dropout: float = 0.,
                 norm: bool =False,
                 activation: str = 'leakyrelu'):
        """This is the Encoder for VAE

        Args:
            layers_list (List[int]): _description_
            dropout (float, optional): _description_. Defaults to 0..
            norm (bool, optional): _description_. Defaults to False.
            activation (str, optional): _description_. Defaults to 'relu'.
        """
        super(EncoderVAE,self).__init__()
        self.encoder=MLP(layers_list[0:-1],
                         dropout,
                         norm,
                         activation,
                         last_act=True)
        self.mu = MLP(layers_list[-2:],
                      dropout=0.,
                      norm=False,
                      activation=None,
                      last_act=False)
        self.log_var = MLP(layers_list[-2:],
                      dropout=0.,
                      norm=False,
                      activation=None,
                      last_act=False)
        self.encoder_layers = layers_list
        self.dropout = dropout
        self.norm = norm
        self.activation = activation
    def forward(self,x):
        out=self.encoder(x)
        mu = self.mu(out)
        log_var = self.log_var(out)
        return {'mu':mu,'log_var':log_var}
    def emb(self,x,eval=True):
        """ Get latent space representation

        Args:
            x: tensor or numpy.array
            eval (bool): if this is in evaluation mode. 
        """
        
        if isinstance(x, np.ndarray):
            x=torch.tensor(x,dtype=torch.float32)
            return_numpy = True
        else:
            return_numpy = False
        if eval:
            self.eval()
            x = x.to(next(self.parameters()).device)
        out=self.encoder(x)
        z=self.mu(out)
        if eval:
            if return_numpy:
                z=z.cpu().detach().numpy()
        return z
    

#####
# Decoder
#####

class Decoder(nn.Module):
    def __init__(self,     
                layers_list: List[int] ,
                dropout: float = 0.,
                norm: bool =False,
                activation: str = 'leakyrelu'):
        """This is the Decoder for Gaussian distribution

        Args:
            layers_list (List[int]): _description_
            dropout (float, optional): _description_. Defaults to 0..
            norm (bool, optional): _description_. Defaults to False.
            activation (str, optional): _description_. Defaults to 'relu'.
        """
        super(Decoder,self).__init__()
        self.decoder=MLP(layers_list,
                    dropout,
                    norm,
                    activation,
                    last_act='softplus')
        
        self.decoder_layers = layers_list
    def forward(self,z):
        x_hat=self.decoder(z)
        # x_hat = F.softplus(x_hat)
        return x_hat
    def generate(self,z,eval=True):
        """Generate output from latent space

        Args:
            z: embedding vector
            eval (bool): if this is in evaluation mode. 
        Returns:
            x_hat
        """
        if isinstance(z, np.ndarray):
            z=torch.tensor(z,dtype=torch.float32)
            return_numpy=True
        else:
            return_numpy=False
        if eval:
            self.eval()
            z = z.to(next(self.parameters()).device)
        x_hat=self.decoder(z)
        if eval:
            if return_numpy:
                x_hat=x_hat.cpu().detach().numpy()
        return x_hat

######
# Model
######
def layer_list(n_genes,n_layers,dim_hiddens,dim_latent):
    encoder_layers = [n_genes]+[dim_hiddens]*n_layers +[dim_latent]
    decoder_layers = [dim_latent]+[dim_hiddens]*n_layers+[n_genes]
    return encoder_layers,decoder_layers


class AutoEncoder(nn.Module):
    # def __init__(self, encoder_layers: List[int], decoder_layers: List[int], dropout: float = 0., norm: bool = False, activation: str = 'relu'):
    def __init__(self, n_genes, n_layers,dim_hiddens,dim_latent,
                 dropout: float = 0., norm: bool = False, activation: str = 'relu'):
        super(AutoEncoder,self).__init__()
        encoder_layers, decoder_layers = layer_list(n_genes,n_layers,dim_hiddens,dim_latent)
        self.model_name = 'AE'
        self.encoder = Encoder(encoder_layers, dropout, norm, activation)
        self.decoder = Decoder(decoder_layers, dropout, norm, activation)
        self.scaler = StandardScaler()
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    def recontruct(self, x,eval=True):
        z=self.encoder.emb(x,eval)
        x_hat=self.decoder.generate(z,eval)
        return x_hat
    def emb(self,x,eval=True):
        if isinstance(x, np.ndarray):
            return self.scaler.transform(self.encoder.emb(x,eval))
        else:
            mean=torch.tensor(self.scaler.mean_,dtype=torch.float32).to(x.device).unsqueeze(0)
            std = torch.tensor(self.scaler.scale_,dtype=torch.float32).to(x.device).unsqueeze(0)
            z = self.encoder.emb(x,eval)
            return (z-mean)/std
        
    def generate(self,z,eval=True):
        if isinstance(z, np.ndarray):
            return self.decoder.generate(self.scaler.inverse_transform(z),eval)
        else:
            mean=torch.tensor(self.scaler.mean_,dtype=torch.float32).to(z.device).unsqueeze(0)
            std = torch.tensor(self.scaler.scale_,dtype=torch.float32).to(z.device).unsqueeze(0)
            z_trans = z*std+mean
            return self.decoder.generate(z_trans,eval)       

class VAE(nn.Module):
    # def __init__(self, encoder_layers: List[int], decoder_layers: List[int], dropout: float = 0., norm: bool = False, activation: str = 'leakyrelu'):
    def __init__(self, n_genes, n_layers,dim_hiddens,dim_latent,
                 dropout: float = 0., norm: bool = False, activation: str = 'relu'):
        super(VAE,self).__init__()
        self.model_name = 'VAE'
        encoder_layers, decoder_layers = layer_list(n_genes,n_layers,dim_hiddens,dim_latent)
        self.encoder = EncoderVAE(encoder_layers, dropout, norm, activation)
        self.decoder = Decoder(decoder_layers, dropout, norm, activation)
        self.scaler = StandardScaler()
    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        encoder_output=self.encoder(x)
        mu, log_var = encoder_output['mu'], encoder_output['log_var']
        std = torch.exp(0.5 * log_var)
        z = self._sample_gauss(mu, std)
        x_hat = self.decoder(z)
        output = {'x_hat':x_hat,
                  'mu':mu,
                  'log_var':log_var}
        return output
    def emb(self,x,eval=True):
        if isinstance(x, np.ndarray):
            return self.scaler.transform(self.encoder.emb(x,eval))
        else:
            mean=torch.tensor(self.scaler.mean_,dtype=torch.float32).to(x.device).unsqueeze(0)
            std = torch.tensor(self.scaler.scale_,dtype=torch.float32).to(x.device).unsqueeze(0)
            z = self.encoder.emb(x,eval)
            return (z-mean)/std

    def generate(self,z,eval=True):
        if isinstance(z, np.ndarray):
            return self.decoder.generate(self.scaler.inverse_transform(z),eval)
        else:
            mean=torch.tensor(self.scaler.mean_,dtype=torch.float32).to(z.device).unsqueeze(0)
            std = torch.tensor(self.scaler.scale_,dtype=torch.float32).to(z.device).unsqueeze(0)
            z_trans = z*std+mean
            return self.decoder.generate(z_trans,eval)            
 
def create_AE(model_name,n_genes, n_layers,dim_hiddens,dim_latent,
                 dropout: float = 0., norm: bool = False, activation: str = 'relu'):
    if model_name=='AE':
        return AutoEncoder(n_genes, n_layers,dim_hiddens,dim_latent,dropout, norm, activation)
    elif model_name == 'VAE':
        return VAE(n_genes, n_layers,dim_hiddens,dim_latent,dropout, norm, activation)
    
class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
        self.recon_loss_fn = nn.MSELoss(reduction='sum')
        self.loss_name='AE'

    def forward(self, x,output):
        # recon_x=output['recon_x']
        return self.recon_loss_fn(output, x)

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.loss_name = 'VAE'
        self.recon_loss_fn = nn.MSELoss(reduction='sum')
        # self.beta=beta

    def forward(self, x,output):
        x_hat=output['x_hat']
        mu=output['mu']
        log_var=output['log_var']
        recon_loss = self.recon_loss_fn(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # self.recon_loss=recon_loss
        # self.kld_loss=kld_loss
        return recon_loss, kld_loss
    def recon_loss_cross(self,x1,x2, output1, output2):
        x1_hat_cross = output1['x_hat_cross']
        x2_hat_cross = output2['x_hat_cross']
        recon_loss1_cross = self.recon_loss_fn(x1_hat_cross,x1)
        recon_loss2_cross = self.recon_loss_fn(x2_hat_cross,x2)
        return recon_loss1_cross, recon_loss2_cross
    

    