import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from piVAE_utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAEEncoder(nn.Module):
    def __init__(self,dim_x,dim_z,dim_u,gen_nodes,disc = True):
        super(VAEEncoder,self).__init__()
        if disc:
            self.z_prior = Z_Prior_Disc(dim_z,dim_u)
        else:
            self.z_prior = Z_Prior_Nn(dim_u,dim_z)
        
        self.z_mean_encode = Encoder(dim_x,gen_nodes,dim_z)
        self.z_log_var_encode = Encoder(dim_x,gen_nodes,dim_z)
    
    def forward(self,x_input,u_input):
        lam_mean, lam_log_var = self.z_prior(u_input)
        z_mean  = self.z_mean_encode(x_input)
        z_log_var = self.z_log_var_encode(x_input)
        post_mean, post_log_var  = compute_posterior([z_mean, z_log_var, lam_mean, lam_log_var])
        z_sample = sampling([post_mean,post_log_var])
    
        return post_mean, post_log_var,z_sample,z_mean,z_log_var,lam_mean , lam_log_var
            
        
class VAEDecoder(nn.Module):
    def __init__(self,dim_x,dim_z,dim_u,n_blk=None,
                min_gen_nodes_decoder_nflow = 30,
                mdl = 'poisson'):
        super(VAEDecoder,self).__init__()
        
        if n_blk is not None:
            self.decode_nflow = Decoder_Nflow(dim_x,dim_z,mdl,min_gen_nodes_decoder_nflow,n_blk)
        self.mdl = mdl
        
    def forward(self,z_input):
        fire_rate = self.decode_nflow(z_input)
        
        if self.mdl == 'poisson':
            fire_rate = torch.clip(fire_rate,min=1e-7 , max = 1e7 )
        
        return fire_rate
        
class piVAE(nn.Module):
    def __init__(self,dim_x,dim_z,dim_u,gen_nodes,n_blk,min_gen_nodes_decoder_nflow = 30,mdl = 'poisson',disc = True):
        super(piVAE,self).__init__()
        
        self.mdl = mdl
        self.encoder = VAEEncoder(dim_x,dim_z,dim_u,gen_nodes,disc)
        self.decoder = VAEDecoder(dim_x,dim_z,dim_u,n_blk,min_gen_nodes_decoder_nflow,mdl)
        if mdl == 'gaussian':
            self.one_layer = nn.Linear(1,dim_x,bias=False)
        
            
    def forward(self,x_input,u_input):
        post_mean, post_log_var,z_sample,z_mean,z_log_var,lam_mean , lam_log_var = self.encoder(x_input,u_input)
        
        fire_rate = self.decoder(z_sample)
        if self.mdl == 'gaussian':
            obs_log_var = self.one_layer(torch.ones([1,1]).to(device))
            
            obs_loglik = torch.sum(
                            torch.square(fire_rate-x_input) / (2.*torch.exp(obs_log_var)) + (obs_log_var/2.) ,
                            dim = -1
                        )
            
            return post_mean, post_log_var,z_sample,z_mean,z_log_var, \
            lam_mean , lam_log_var ,fire_rate,obs_loglik
        
        else:
            obs_loglik = torch.sum(
                        fire_rate - x_input*torch.log(fire_rate)
                        ,dim = -1)
            return post_mean, post_log_var,z_sample,z_mean,z_log_var, \
            lam_mean , lam_log_var ,fire_rate,obs_loglik
    
def piVAE_Dataloader(xs,us,batch_size=32):
    class MyDataset(Dataset):
        """
            Wrapper
        """
        def __init__(self,xs,us):
            super(MyDataset,self).__init__()
            self.xs = np.concatenate(xs)
            self.us = np.concatenate(us)
        def __len__(self):
            return len(self.xs)
        
        def __getitem__(self,idx):
            return self.xs[idx],self.us[idx]
        
    return DataLoader(MyDataset(xs,us),shuffle=False,batch_size=batch_size)

def piVAE_Optimizer(params,lr):
    return torch.optim.Adam(params,lr)

def piVAE_Loss(post_log_var,lam_log_var,post_mean,lam_mean,obs_loglik,):
    kl_loss = 1 + post_log_var - lam_log_var - ((torch.square(post_mean-lam_mean) +torch.exp(post_log_var))/torch.exp(lam_log_var));
    kl_loss = torch.sum(kl_loss, axis=-1)
    vae_loss = torch.mean(obs_loglik -0.5*kl_loss)
    return vae_loss



def trainContLabel(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, u) in enumerate(dataloader):
        # if u is continuos, then type to float.
        x,u = x.float(),u.float()
        x,u = x.to(device),u.to(device)
        # Compute prediction and loss
        post_mean, post_log_var,z_sample,z_mean,z_log_var, lam_mean , lam_log_var ,fire_rate,obs_loglik = model(x,u)
        loss = loss_fn(post_log_var,lam_log_var,post_mean,lam_mean,obs_loglik)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            batch += 1 # for full batch size which is equal to dataset size
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def trainDiscLabel(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, u) in enumerate(dataloader):
        # if u is discrete, then type to int.
        x,u = x.float(),u.int()
        x,u = x.to(device),u.to(device)
        # Compute prediction and loss  
        post_mean, post_log_var,z_sample,z_mean,z_log_var, lam_mean , lam_log_var ,fire_rate,obs_loglik = model(x,u)
        loss = loss_fn(post_log_var,lam_log_var,post_mean,lam_mean,obs_loglik)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            batch += 1 # for full batch size which is equal to dataset size
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        


        
    