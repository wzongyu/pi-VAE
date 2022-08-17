import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from VAE_utils import *


class VAEEncoder(nn.Module):
    def __init__(self,x_dim,z_dim,gen_nodes):
        super(VAEEncoder,self).__init__()
        u_dim = 1   
        self.z_prior_disc  = Z_Prior_Disc(z_dim,u_dim) 
        self.z_mean_encoder = Encoder(x_dim,gen_nodes,z_dim)
        self.z_log_var_encoder = Encoder(x_dim,gen_nodes,z_dim)
        
    def forward(self,x_input,u_input):
        lam_mean , lam_log_var = self.z_prior_disc(u_input)
        
        z_mean = self.z_mean_encoder(x_input)
        z_log_var = self.z_log_var_encoder(x_input)
        
        # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
        post_mean, post_log_var = compute_posterior([z_mean, z_log_var, lam_mean, lam_log_var]);
        
        z_sample = sampling([post_mean, post_log_var]);
        
        return lam_mean,lam_log_var,z_mean,z_log_var,post_mean, post_log_var,z_sample

class VAEDecoder(nn.Module):
    def __init__(self,x_dim,z_dim,gen_nodes,n_blk=None,mdl='poisson',):
        super(VAEDecoder,self).__init__()
        self.mdl = mdl
        if n_blk is not None:
            self.decode = Decoder_Nflow(x_dim,z_dim,mdl,n_blk)
        else:
            self.decode = Decoder(z_dim,gen_nodes,x_dim,mdl)
    
    def forward(self,z_input):
        fire_rate = self.decode(z_input)
        
        if self.mdl == 'poisson':
            fire_rate = torch.clip(fire_rate,min=1e-7,max=1e7)
        return fire_rate
        
class VAE(nn.Module):
    def __init__(self,dim_x,dim_z,gen_nodes,n_blk,
                mdl = 'poisson'):
        super(VAE,self).__init__()
        self.mdl = mdl
        self.encoder = VAEEncoder(dim_x,dim_z,gen_nodes)
        self.decoder = VAEDecoder(dim_x,dim_z,gen_nodes,n_blk,mdl)
        if mdl == 'gaussian':
            self.one_layer = nn.Linear(1,dim_x,bias=False)
            
    def forward(self,x_input,u_input):
        
        post_mean, post_log_var,z_sample,z_mean,z_log_var,lam_mean , lam_log_var = self.encoder(x_input,u_input)
        
        fire_rate = self.decoder(z_sample)
        if self.mdl == 'gaussian':
            obs_log_var = self.one_layer(torch.ones([1,1]))
            
            obs_loglik = torch.sum(
                            torch.square(fire_rate-x_input) / (2.*torch.exp(obs_log_var)) + (obs_log_var/2.) ,
                            dim = -1
                        )
            return post_mean, post_log_var,z_sample,z_mean,z_log_var, \
            lam_mean , lam_log_var ,fire_rate,obs_loglik
        else:
            obs_loglik = torch.sum(fire_rate - x_input*torch.log(fire_rate),dim = -1)
            return post_mean, post_log_var,z_sample,z_mean,z_log_var, \
            lam_mean , lam_log_var ,fire_rate,obs_loglik
    
        
def VAE_Loss(post_log_var,
                 lam_log_var,
                 post_mean,
                 lam_mean,
                 obs_loglik,
                ):
    kl_loss = 1 + post_log_var - lam_log_var - ((torch.square(post_mean-lam_mean) + torch.exp(post_log_var))/torch.exp(lam_log_var));
    kl_loss = torch.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = torch.mean(obs_loglik + kl_loss)
    return vae_loss

def VAE_Optimizer(params,lr):
    return torch.optim.Adam(params,lr)

def VAE_Dataloader(xs,us,batch_size=32):
    class MyDataset(Dataset):
        def __init__(self,xs,us):
            super(MyDataset,self).__init__()
            self.xs = np.concatenate(xs)
            self.us = np.concatenate(us)
        def __len__(self):
            return len(self.xs)
        
        def __getitem__(self,idx):
            return self.xs[idx],self.us[idx]
    return DataLoader(MyDataset(xs,us),shuffle=False,batch_size=batch_size)
    
def trainDiscLabel(dataloader, model, loss_fn, optimizer,device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, u) in enumerate(dataloader):
        x,u = x.float(),u.int()
        # Compute prediction and loss
        x,u = x.to(device),u.to(device)
        post_mean, post_log_var,z_sample,z_mean,z_log_var, \
            lam_mean , lam_log_var ,fire_rate,obs_loglik = model(x,u)
        
        loss = loss_fn(post_log_var,lam_log_var,post_mean,lam_mean,obs_loglik)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            batch += 1 # full batch size
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
