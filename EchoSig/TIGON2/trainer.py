import torch
import torch.optim as optim
import numpy as np
from EchoSig.TIGON2.loss import OT_loss
from EchoSig.TIGON2.utility import MultimodalGaussian_density, Sampling, trans_loss
import warnings
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint
from functools import partial, reduce
from tqdm import tqdm
import time,os
import math

def save_checkpoint(epoch, func, optimizer,Loss, Loss_rec,WFR, Wass1,Wass2, Mass1, Mass2,
                    path, **kwargs):
    state = {
        'epoch': epoch,
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict':scheduler.state_dict(),
        'loss':
        {
            'Loss':Loss,
            'Loss_rec':Loss_rec,
            'WFR':WFR,
            'Wass1':Wass1,
            'Wass2':Wass2,
            'Mass1':Mass1,
            'Mass2':Mass2,
            }
    }
    
    # Add additional values to the state if provided
    state.update(kwargs)
    
    torch.save(state, path)

class Trainer(object):
    def __init__(self,
                 func,
                 seed:int=42,
                 device=None,
                #  test_size:float=0.1,
                 batch_size:int=256,
                 lr:float=1e-3,
                 weight_decay:float=0.,
                 l1_lambda:float=0.,
                 max_epoch = 5000,
                 early_stopping=True,
                 tol=0.,  
                 patience=100,
                 sigma=0.001,
                 gamma=0.1,
                 ):
                if device is None:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self.device = device
                self.func = func
                self.func.to(self.device)
                self.lr=lr
                self.weight_decay = weight_decay
                self.l1_lambda = l1_lambda
                self.max_epoch = max_epoch
                self.batch_size=batch_size
                self.seed=seed
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
                self.early_stopping = early_stopping
                self.patience = patience
                self.tol = tol
                # default_lr_scheduler_params = {
                #     'mode': 'min',
                #     'factor': 0.5,
                #     'patience': 100,
                #     'verbose': False
                # }
                self.sigma=sigma
                self.gamma=gamma
                # if lr_scheduler_params is not None:
                #     default_lr_scheduler_params.update(lr_scheduler_params)



                self.optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay= weight_decay)
                # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                #                                                       **default_lr_scheduler_params)
                self.loss=OT_loss()

    
    def train_step(self,data_train,train_time,sigma,gamma,l1_lambda,options,time_dic):
        #using pot
        warnings.filterwarnings("ignore")

        loss_rec = 0
        wass1 = torch.zeros(len(data_train)-1).type(torch.float32).to(self.device)
        wass2 = torch.zeros(len(data_train)-2).type(torch.float32).to(self.device)
        mass1 = torch.zeros(len(data_train)-1).type(torch.float32).to(self.device)
        mass2 = torch.zeros(len(data_train)-2).type(torch.float32).to(self.device)
        #trans_cost = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
        # odeint_setp = train_time[-1]/5#gcd_list([num * 100 for num in train_time])/100
        # options.update({'t0': train_time[0]})
        # options.update({'t1': train_time[-1]})
        # options.update({'t_eval':train_time}) 
        t1=time.time()
        x = Sampling(self.batch_size,0,data_train,sigma,self.device)
        t2 = time.time()
        time_dic['sample']+=t2-t1
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(self.device)
        g_t1 = logp_diff_t1
        t1 = time.time()
        # z_t0, _, logp_diff_t0 = odesolve(self.func,y0=(x, g_t1, logp_diff_t1),options=options)
        z_t0, _, logp_diff_t0 = odeint(self.func,y0=(x, g_t1, logp_diff_t1),
                            t=torch.tensor(train_time).type(torch.float32).to(self.device),
                            method = options['method'],
                            rtol = options['rtol'],
                            atol = options['atol'],
                            options={'step_size':options['step_size']})
        t2=time.time()
        time_dic['ode'] +=t2-t1
        t1=time.time()
        aa = MultimodalGaussian_density(x, 0, data_train,sigma,self.device)   
        t2=time.time()
        time_dic['density']+=t2-t1
        for i in range(1,len(train_time)): 
            logp_x = torch.log(aa)+logp_diff_t0[i].view(-1)
            aaa = torch.exp(logp_x.view(-1))
            x0 = data_train[i]#Sampling(self.batch_size,i+1,data_train,sigma,device)
            num_tsamples = x0.shape[0]
            rho_true = torch.full((num_tsamples,1), float(1/num_tsamples), requires_grad=True).to(self.device)
            #MultimodalGaussian_density(x0, i+1, data_train,sigma,device)* torch.tensor(data_train[i+1].shape[0]/data_train[0].shape[0])
            t1=time.time()
            wass1[i-1] = self.loss(z_t0[i], x0, aaa.unsqueeze(1)/aaa.sum(),rho_true) #wsd(aaa.unsqueeze(1)/aaa.sum(),z_t0[i],rho_true,x0)
            #WassDist_fn(z_t0,x0,aaa.unsqueeze(1),aaaa.unsqueeze(1))
            mass1[i-1] = torch.abs(aaa.sum()/aa.sum()-torch.tensor(data_train[i].shape[0]/data_train[0].shape[0]))
            t2=time.time()
            time_dic['recon']+=t2-t1
            loss_rec = loss_rec  + wass1[i-1] + mass1[i-1]

            # check the density loss at i+1
            #loss_density = density_loss(z_t0, data_train[i+1])
            #loss = loss + loss_density*1e3
            
            # loss between each two time points
            if i < len(train_time)-1:
                # options.update({'t0': train_time[i]})
                # options.update({'t1': train_time[i+1]})
                # options.update({'t_eval':None}) 
                t1=time.time()
                x2 = Sampling(self.batch_size,i,data_train,sigma,self.device)
                t2=time.time()
                time_dic['sample']+=t2-t1
                x2.requires_grad=True
                t1=time.time()
                # z_t2, _, logp_diff_t2= odesolve(self.func,y0=(x2, g_t1, logp_diff_t1),options=options)
                z_t2, _, logp_diff_t2 = odeint(self.func,y0=(x2, g_t1, logp_diff_t1),
                                               t=torch.tensor([train_time[i],train_time[i+1]]).type(torch.float32).to(self.device),
                                               method = options['method'],
                                               rtol = options['rtol'],
                                               atol = options['atol'],
                                               options={'step_size':options['step_size']})
                z_t2=z_t2[-1,:,:]
                logp_diff_t2=logp_diff_t2[1:,:,:]
                t2=time.time()
                time_dic['ode']+=t2-t1
                t1=time.time()
                aa2 = MultimodalGaussian_density(x2, i, data_train,sigma,self.device)* torch.tensor(data_train[i].shape[0]/data_train[0].shape[0])
                t2=time.time()
                time_dic['density']+=t2-t1
                logp_x = torch.log(aa2)+logp_diff_t2.view(-1)
                aaa = torch.exp(logp_x.view(-1))
                x02 = data_train[i+1]
                num_tsamples2 = x02.shape[0]
                rho_true2 = torch.full((num_tsamples2,1), float(1/num_tsamples2), requires_grad=True).to(self.device)            
                t1=time.time()
                wass2[i-1] =  self.loss(z_t2, x02,aaa.unsqueeze(1)/aaa.sum(),rho_true2) #wsd(aaa.unsqueeze(1)/aaa.sum(),z_t2,rho_true2,x02)
                    #WassDist_fn(z_t2,x0,aaa.unsqueeze(1),aaaa.unsqueeze(1))
                mass2[i-1] = torch.abs(aaa.sum()/aa2.sum()-torch.tensor(data_train[i+1].shape[0]/data_train[i].shape[0]))    
                t2=time.time()
                time_dic['recon']+=t2-t1

                loss_rec = loss_rec  + wass2[i-1] + mass2[i-1]

                #loss_density = density_loss(z_t2, data_train[i+1])
            #loss = loss + loss_density*1e3

            torch.cuda.empty_cache()
        if gamma>0.:
            # compute transport cost efficiency
            for i in range(0,len(train_time)-1):
                t1=time.time()
                x0 = Sampling(self.batch_size,i,data_train,sigma,self.device) 
                t2=time.time()
                time_dic['sample']+=t2-t1
                transport_cost = partial(trans_loss,func=self.func,device=self.device,odeint_setp=options['wfr']['step_size'], t_start=train_time[i])
                logp_diff_t00 = torch.zeros(self.batch_size, 1).type(torch.float32).to(self.device)
                g_t00 = logp_diff_t00
                t1=time.time()
                _,_,wfr = odeint(transport_cost,y0=(x0, g_t00, logp_diff_t00),
                                t = torch.tensor([train_time[i], train_time[i+1]]).type(torch.float32).to(self.device),
                                atol=options['wfr']['atol'],rtol=options['wfr']['rtol'],
                                method=options['wfr']['method'],
                                options = {'step_size': options['wfr']['step_size']})#{'step_size': options['wfr']['step_size']})
                wfr=( train_time[i+1]-train_time[i]) * wfr[-1].mean(0)
                t2=time.time()
                time_dic['wfr']+=t2-t1
        else:
            wfr=torch.zeros_like(loss_rec)
        loss = loss_rec +  gamma * wfr    #/1e10
        if l1_lambda>0:
            l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.func.parameters())
            loss = loss + l1_lambda * l1_penalty
        return loss,loss_rec, wfr, wass1, wass2,mass1,mass2,time_dic
    
    def train(self,data_train,train_time,save_dir):
        if not self.func.odesolver['step_size']:
            self.func.odesolver['step_size']=np.array(train_time).max()/self.func.odesolver['num_steps']
        if not self.func.odesolver['wfr']['step_size']:
            ## have a step_size with using greatest common divisor (gcd) for all training time. 
            scaled = [int(t * 100) for t in train_time if t != 0]
            gcd = reduce(math.gcd, scaled)/100
            self.func.odesolver['wfr']['step_size']=np.float64(gcd/4)
            
            #self.func.odesolver['wfr']['step_size']=np.array(train_time).max()/self.func.odesolver['wfr']['num_steps']
        options=self.func.odesolver

        # options = diffeq_args()
        Loss = []
        Loss_rec=[]
        Wass1 = []
        Wass2 = []
        WFR = []
        Sigma = []
        Mass1 = []
        Mass2 = []
        sigma=self.sigma
        l1_lambda=self.l1_lambda
        # gamma=self.gamma
        gamma=0.
        trigger_times = 0
        if save_dir is not None:
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            ckpt_path = os.path.join(save_dir, 'ckpt.pth')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                self.func.load_state_dict(checkpoint['func_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print('Loaded ckpt from {}'.format(ckpt_path))
                start_epoch = checkpoint['epoch']+1

                if checkpoint['finish']:
                    time_dic={}
                    torch.cuda.empty_cache()
                    return time_dic
                trigger_times = checkpoint['trigger_times']
            else:
                start_epoch=0
        try:
            print('start from epoch: '+str(start_epoch))
            # trigger_times = 0
            best_loss = float('inf')
            # wsd = OT_loss() #OTLoss(device,0.01, 0.1)
            time_dic = {'sample':0.,
                        'recon':0.,
                        'density':0.,
                        'wfr':0.,
                        'ode':0.}
            start_time = time.time()
            time_fwd = 0 
            time_bck = 0
            # for itr in tqdm(range(self.max_epoch)):
            for itr in range(start_epoch,self.max_epoch):
                self.optimizer.zero_grad()
                t1=time.time()
                loss, loss_rec, wfr, wass1, wass2,mass1,mass2,time_dic = self.train_step(data_train,train_time,sigma,gamma,l1_lambda,options,time_dic)
                t2=time.time()
                val_loss = loss.item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= self.patience:
                        if gamma==0. and self.gamma>0: 
                            # turn on WFR metrics; reset trigger_times and best_loss to default
                            gamma=self.gamma
                            print("WFR penelty is ON!!!!!!!!!!!!!!!!!!!")
                            trigger_times=0
                            best_loss = float('inf')
                        else:
                            print(f"Early stopping at iteration {itr}")
                            Loss.append(loss.item())
                            Loss_rec.append(loss_rec.item())
                            WFR.append(wfr.item())
                            Sigma.append(sigma)
                            Wass1.append(wass1.tolist())
                            Wass2.append(wass2.tolist())
                            Mass1.append(mass1.tolist())
                            Mass2.append(mass2.tolist())
                            ckpt_path = os.path.join(save_dir, 'ckpt.pth')
                            save_checkpoint(itr, 
                                            func=self.func, 
                                            optimizer = self.optimizer,
                                            # scheduler = self.scheduler, 
                                            Loss=Loss,Loss_rec=Loss_rec, 
                                            WFR=WFR, Sigma=Sigma, 
                                            Wass1=Wass1,Wass2=Wass2,
                                            Mass1=Mass1, Mass2=Mass2,
                                            path=ckpt_path,
                                            finish=True,
                                            trigger_times=trigger_times,
                                            )
                            break
                t3=time.time()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(loss.item())
                t4=time.time()
                time_fwd += t2-t1
                time_bck += t4-t3
                Loss.append(loss.item())
                Loss_rec.append(loss_rec.item())
                WFR.append(wfr.item())
                Sigma.append(sigma)
                Wass1.append(wass1.tolist())
                Wass2.append(wass2.tolist())
                Mass1.append(mass1.tolist())
                Mass2.append(mass2.tolist())
                
                print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
                
                if itr % 500 == 0:
                    ckpt_path = os.path.join(save_dir, 'ckpt_itr{}.pth'.format(itr))
                    save_checkpoint(itr, 
                                    func=self.func, 
                                    optimizer = self.optimizer,
                                    # scheduler = self.scheduler, 
                                    Loss=Loss,Loss_rec=Loss_rec,
                                    WFR=WFR, Sigma=Sigma, 
                                    Wass1=Wass1,Wass2=Wass2,
                                    Mass1=Mass1, Mass2=Mass2,
                                    path=ckpt_path,
                                    finish=False,
                                    trigger_times=trigger_times,)
                    print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                    
                
            run_time = time.time() - start_time
            print(f'Total run time: {np.round(run_time, 5)}s | Total epochs: {self.max_epoch}') 
            time_dic['totoal']=run_time
            time_dic['fwd_total']=time_fwd
            time_dic['bck_total']=time_bck            
            # return LOSS, WFR, Wass1,Wass2, Mass1, Mass2     
            torch.cuda.empty_cache()
            return time_dic
        except KeyboardInterrupt:
            if save_dir is not None:
                ckpt_path = os.path.join(save_dir, 'ckpt.pth')
                save_checkpoint(itr, 
                                func=self.func, 
                                optimizer = self.optimizer,
                                # scheduler = self.scheduler, 
                                Loss=Loss,Loss_rec=Loss_rec, 
                                WFR=WFR, Sigma=Sigma, 
                                Wass1=Wass1,Wass2=Wass2,
                                Mass1=Mass1, Mass2=Mass2,
                                path=ckpt_path,
                                finish=False,
                                trigger_times=trigger_times,)
                print('Stored ckpt at {}'.format(ckpt_path))
        print('Training complete after {} iters.'.format(itr))
        torch.cuda.empty_cache()
        
        ckpt_path = os.path.join(save_dir, 'ckpt.pth')
        save_checkpoint(itr, 
                        func=self.func, 
                        optimizer = self.optimizer,
                        # scheduler = self.scheduler, 
                        Loss=Loss,Loss_rec=Loss_rec, 
                        WFR=WFR, Sigma=Sigma, 
                        Wass1=Wass1,Wass2=Wass2,
                        Mass1=Mass1, Mass2=Mass2,
                        path=ckpt_path,
                        finish=True,
                        trigger_times=trigger_times,)
        print('Stored ckpt at {}'.format(ckpt_path))
        torch.cuda.empty_cache()



    
    
    