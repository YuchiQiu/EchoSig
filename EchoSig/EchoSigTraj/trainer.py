import torch
import torch.optim as optim
import numpy as np
from EchoSig.EchoSigTraj.loss import OT_loss
from EchoSig.EchoSigTraj.utility import Sampling, wfr_dynamics
import warnings
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
import time,os
import random

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
                 max_epoch = 5000,
                 tol=0.,  
                 patience=100,
                 sigma=None,
                 gamma=0.1,
                 alpha_wfr=1.,
                 l_mass=1.,
                 ):
                if device is None:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self.device = device
                self.func = func
                self.func.to(self.device)
                self.lr=lr
                self.weight_decay = weight_decay
                self.max_epoch = max_epoch
                self.batch_size=batch_size
                self.seed=seed
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
                random.seed(self.seed)
                self.patience = patience
                self.tol = tol
                # default_lr_scheduler_params = {
                #     'mode': 'min',
                #     'factor': 0.5,
                #     'patience': 100,
                #     'verbose': False
                # }
                # Deprecated compatibility argument for older configs.
                # Training uses unperturbed samples; trajectory-generation
                # utilities still have their own meaningful sigma parameter.
                self.gamma=gamma
                self.alpha_wfr=alpha_wfr
                self.l_mass=l_mass
                # if lr_scheduler_params is not None:
                #     default_lr_scheduler_params.update(lr_scheduler_params)



                self.optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay= weight_decay)
                # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                #                                                       **default_lr_scheduler_params)
                self.loss=OT_loss()


    def reconstruction_loss(
        self,
        x_pred,
        logm_pred,
        x_target,
        target_mass_ratio,
        l_mass=None,
    ):
        """Compute endpoint Wasserstein reconstruction and mass penalty.

        This function does not perform an ODE solve. ``logm_pred`` is the
        relative log-mass accumulated from the starting time of the rollout,
        and ``target_mass_ratio`` is the corresponding observed end/start mass
        ratio.

        Returns:
            A tuple ``(total, wasserstein, mass)``.
        """
        if l_mass is None:
            l_mass = self.l_mass

        logm_pred = logm_pred.reshape(-1)
        particle_mass = torch.exp(logm_pred)
        rho_pred = particle_mass / particle_mass.sum()

        num_target = x_target.shape[0]
        rho_target = torch.full(
            (num_target,),
            1.0 / num_target,
            dtype=x_pred.dtype,
            device=x_pred.device,
        )

        wasserstein = self.loss(
            x_pred,
            x_target,
            rho_pred,
            rho_target,
        )

        predicted_mass_ratio = particle_mass.mean()
        target_mass_ratio = torch.as_tensor(
            target_mass_ratio,
            dtype=x_pred.dtype,
            device=x_pred.device,
        )
        mass = torch.abs(predicted_mass_ratio - target_mass_ratio)
        total = wasserstein + l_mass * mass

        return total, wasserstein, mass


    def train_step(self,data_train,train_time,gamma,l_mass,alpha_wfr,options,time_dic):
        #using pot
        warnings.filterwarnings("ignore")

        loss_rec = 0
        wass1 = torch.zeros(len(data_train)-1).type(torch.float32).to(self.device)
        wass2 = torch.zeros(len(data_train)-2).type(torch.float32).to(self.device)
        mass1 = torch.zeros(len(data_train)-1).type(torch.float32).to(self.device)
        mass2 = torch.zeros(len(data_train)-2).type(torch.float32).to(self.device)

        ##############################
        ## LOSS between TWO consective time points ##
        ##############################
        for i in range(1,len(train_time)-1):
            # options.update({'t0': train_time[i]})
            # options.update({'t1': train_time[i+1]})
            # options.update({'t_eval':None})
            t1=time.perf_counter()
            x0_local = Sampling(self.batch_size,i,data_train,sigma=0,device = self.device)
            t2=time.perf_counter()
            time_dic['sampling'] += t2 - t1
            # x0_local.requires_grad_(True)  # only needed for divergence/Jacobian
            logm0_local = torch.zeros(
                x0_local.shape[0],
                1,
                dtype=x0_local.dtype,
                device=x0_local.device,
            )
            t1=time.perf_counter()
            x_local, logm_local = odeint(self.func,y0=(x0_local, logm0_local),
                                            t=torch.tensor([train_time[i],train_time[i+1]]).type(torch.float32).to(self.device),
                                            method = options['method'],
                                            rtol = options['rtol'],
                                            atol = options['atol'])
            x1_local = x_local[-1,:,:]
            logm1_local = logm_local[-1,:,:]
            t2=time.perf_counter()
            time_dic['dynamics'] += t2 - t1
            x1_target_local = data_train[i+1]
            t1=time.perf_counter()
            loss_rec_i, wass2_i, mass2_i = self.reconstruction_loss(
                x_pred=x1_local,
                logm_pred=logm1_local,
                x_target=x1_target_local,
                target_mass_ratio=data_train[i+1].shape[0]/data_train[i].shape[0],
                l_mass=l_mass,
            )
            wass2[i-1] = wass2_i
            mass2[i-1] = mass2_i

            t2=time.perf_counter()
            time_dic['reconstruction'] += t2 - t1

            loss_rec = loss_rec + loss_rec_i

            #loss_density = density_loss(x1_local, data_train[i+1])
        #loss = loss + loss_density*1e3

        ##############################
        ## GLOBAL LOSS ##
        ##############################
        
        #trans_cost = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
        # odeint_setp = train_time[-1]/5#gcd_list([num * 100 for num in train_time])/100
        # options.update({'t0': train_time[0]})
        # options.update({'t1': train_time[-1]})
        # options.update({'t_eval':train_time}) 
        t1=time.perf_counter()
        x0_global = Sampling(self.batch_size,0,data_train,sigma=0,device=self.device)
        t2 = time.perf_counter()
        time_dic['sampling'] += t2 - t1
        # x0_global.requires_grad_(True)  # only needed for divergence/Jacobian
        logm0_global = torch.zeros(
            x0_global.shape[0],
            1,
            dtype=x0_global.dtype,
            device=x0_global.device,
        )
        # g_t1 = logp_diff_t1
        t1 = time.perf_counter()
        global_time = torch.tensor(train_time).type(torch.float32).to(self.device)
        if gamma > 0.:
            wfr0_global = torch.zeros_like(logm0_global)
            global_dynamics = partial(
                wfr_dynamics,
                func=self.func,
                alpha_wfr=alpha_wfr,
            )
            x_traj_global, logm_traj_global, wfr_traj_global = odeint(
                global_dynamics,
                y0=(x0_global, logm0_global, wfr0_global),
                t=global_time,
                method=options['method'],
                rtol=options['rtol'],
                atol=options['atol'],
            )
        else:
            x_traj_global, logm_traj_global = odeint(
                self.func,
                y0=(x0_global, logm0_global),
                t=global_time,
                method=options['method'],
                rtol=options['rtol'],
                atol=options['atol'],
            )
        t2=time.perf_counter()
        # This includes the augmented WFR state when gamma > 0, so the more
        # general name ``dynamics`` is more accurate than ``ode`` or ``wfr``.
        time_dic['dynamics'] += t2 - t1
        for i in range(1,len(train_time)):
            x_target_global = data_train[i]
            t1=time.perf_counter()
            loss_rec_i, wass1_i, mass1_i = self.reconstruction_loss(
                x_pred=x_traj_global[i],
                logm_pred=logm_traj_global[i],
                x_target=x_target_global,
                target_mass_ratio=data_train[i].shape[0]/data_train[0].shape[0],
                l_mass=l_mass,
            )
            wass1[i-1] = wass1_i
            mass1[i-1] = mass1_i

            t2=time.perf_counter()
            time_dic['reconstruction'] += t2 - t1
            loss_rec = loss_rec + loss_rec_i

            # check the density loss at i+1
            #loss_density = density_loss(x_traj_global[i], data_train[i+1])
            #loss = loss + loss_density*1e3

            torch.cuda.empty_cache()
        if gamma>0:
            wfr = (train_time[-1] - train_time[0]) *wfr_traj_global[-1].mean()
            
        else:
            wfr = torch.zeros_like(loss_rec)
        loss = loss_rec +  gamma * wfr 

        return loss,loss_rec, wfr, wass1, wass2,mass1,mass2,time_dic
    
    def train(self,data_train,train_time,save_dir):
        options=self.func.odesolver

        # options = diffeq_args()
        Loss = []
        Loss_rec=[]
        Wass1 = []
        Wass2 = []
        WFR = []
        Mass1 = []
        Mass2 = []
        # sigma=self.sigma
        l_mass=self.l_mass
        alpha_wfr=self.alpha_wfr
        # gamma=self.gamma
        gamma=0.
        trigger_times = 0
        best_loss = float('inf')
        ckpt_path = None
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
                gamma = checkpoint.get('gamma', gamma)
                best_loss = checkpoint.get('best_loss', best_loss)
            else:
                start_epoch=0
        else:
            start_epoch=0
        try:
            print('start from epoch: '+str(start_epoch))
            # trigger_times = 0
            # wsd = OT_loss() #OTLoss(device,0.01, 0.1)
            time_dic = {
                'sampling': 0.,
                'dynamics': 0.,
                'reconstruction': 0.,
            }
            start_time = time.perf_counter()
            time_fwd = 0 
            time_bck = 0
            # for itr in tqdm(range(self.max_epoch)):
            for itr in range(start_epoch,self.max_epoch):
                self.optimizer.zero_grad()
                t1=time.perf_counter()
                loss, loss_rec, wfr, wass1, wass2,mass1,mass2,time_dic = self.train_step(data_train,train_time,gamma,l_mass,alpha_wfr,options,time_dic)
                t2=time.perf_counter()
                val_loss = loss.item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    trigger_times = 0
                    # If gamma > 0 is configured, gamma == 0 is only the
                    # pretraining phase.  For gamma == 0 runs, the first phase
                    # is already the final phase and is eligible immediately.
                    final_phase = self.gamma <= 0 or gamma > 0
                    if ckpt_path is not None and final_phase:
                        save_checkpoint(
                            itr,
                            func=self.func,
                            optimizer=self.optimizer,
                            Loss=Loss,
                            Loss_rec=Loss_rec,
                            WFR=WFR,
                            Wass1=Wass1,
                            Wass2=Wass2,
                            Mass1=Mass1,
                            Mass2=Mass2,
                            path=ckpt_path,
                            finish=False,
                            trigger_times=trigger_times,
                            best_loss=best_loss,
                            gamma=gamma,
                        )
                        print(
                            'Iter {}, loss improved to {:.6f}; stored best '
                            'ckpt at {}'.format(itr, best_loss, ckpt_path)
                        )
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
                            break
                t3=time.perf_counter()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(loss.item())
                t4=time.perf_counter()
                time_fwd += t2-t1
                time_bck += t4-t3
                Loss.append(loss.item())
                Loss_rec.append(loss_rec.item())
                WFR.append(wfr.item())
                Wass1.append(wass1.tolist())
                Wass2.append(wass2.tolist())
                Mass1.append(mass1.tolist())
                Mass2.append(mass2.tolist())
                
                print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
                
                if itr % 500 == 0 and save_dir is not None:
                    periodic_ckpt_path = os.path.join(save_dir, 'ckpt_itr{}.pth'.format(itr))
                    save_checkpoint(itr, 
                                    func=self.func, 
                                    optimizer = self.optimizer,
                                    # scheduler = self.scheduler, 
                                    Loss=Loss,Loss_rec=Loss_rec,
                                    WFR=WFR, 
                                    Wass1=Wass1,Wass2=Wass2,
                                    Mass1=Mass1, Mass2=Mass2,
                                    path=periodic_ckpt_path,
                                    finish=False,
                                    trigger_times=trigger_times,
                                    best_loss=best_loss,
                                    gamma=gamma,)
                    print('Iter {}, Stored ckpt at {}'.format(itr, periodic_ckpt_path))
                    
            # Keep ckpt.pth pointing at the minimum-loss model from the final
            # phase, and restore that model into the trainer before returning.
            if ckpt_path is not None and os.path.exists(ckpt_path):
                best_checkpoint = torch.load(ckpt_path, map_location=self.device)
                if 'best_loss' in best_checkpoint:
                    self.func.load_state_dict(best_checkpoint['func_state_dict'])
                    self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
                    best_checkpoint['finish'] = True
                    torch.save(best_checkpoint, ckpt_path)
                    print(
                        'Stored best model from iteration {} (loss {:.6f}) at {}'.format(
                            best_checkpoint['epoch'],
                            best_checkpoint['best_loss'],
                            ckpt_path,
                        )
                    )
                elif self.gamma > 0:
                    print('No best model was saved because the WFR phase did not start.')
            elif self.gamma > 0:
                print('No best model was saved because the WFR phase did not start.')

            run_time = time.perf_counter() - start_time
            print(f'Total run time: {np.round(run_time, 5)}s | Total epochs: {self.max_epoch}') 
            time_dic['training_total'] = run_time
            time_dic['forward_total'] = time_fwd
            time_dic['backward_total'] = time_bck
            # return LOSS, WFR, Wass1,Wass2, Mass1, Mass2     
            torch.cuda.empty_cache()
            return time_dic
        except KeyboardInterrupt:
            # A worse/interrupted model must not overwrite the best checkpoint.
            print('Training interrupted after {} iters.'.format(itr))
            if ckpt_path is not None and os.path.exists(ckpt_path):
                best_checkpoint = torch.load(ckpt_path, map_location=self.device)
                if 'best_loss' in best_checkpoint:
                    self.func.load_state_dict(best_checkpoint['func_state_dict'])
                    self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
                    print('Kept best ckpt at {}'.format(ckpt_path))
            time_dic['training_total'] = time.perf_counter() - start_time
            time_dic['forward_total'] = time_fwd
            time_dic['backward_total'] = time_bck
            torch.cuda.empty_cache()
            return time_dic



    
    
    
