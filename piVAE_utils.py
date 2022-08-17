import torch
import numpy as np
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    """
        Define mean or log of variance of q(z|x).
        
        Arguments:
            x_dim: dim of observations x
            num_gen_nodes: number of nodes in the hidden layer of encoder network
            z_dim: dimension of latents z
    """
    def __init__(self,x_dim,num_gen_nodes,z_dim):
        super(Encoder,self).__init__()
        n_nodes = [x_dim,num_gen_nodes,num_gen_nodes,z_dim];
        n_layers = len(n_nodes)-1
        act_func = ['tanh','tanh','linear']
        self.module = nn.Sequential()
        for ii in range(n_layers):
            self.module.add_module(name =  f"linear_{ii}",module = nn.Linear(n_nodes[ii],n_nodes[ii+1]))
            if act_func[ii] == 'tanh':
                self.module.add_module(name = f"act_{ii}",module = nn.Tanh())
    def forward(self,x_input):
        output = x_input
        for module in self.module:
            output = module(output)
        return output


class Z_Prior_Disc(nn.Module):
    """
        Compute the prior mean and log of variance of prior p(z|u) for discrete u.
        We assume p(z|u) as gaussian distribution with mean and log of variance treated as different real numbers for different u.
        
        Arguments:
            z_dim : dimension of latents z
            num_u : number of different labels u
    """
    def __init__(self,z_dim,num_u):
        super(Z_Prior_Disc,self).__init__()
        self.lam_mean = nn.Embedding(num_embeddings=num_u,embedding_dim = z_dim)
        self.lam_log_var = nn.Embedding(num_embeddings=num_u,embedding_dim = z_dim)
    def forward(self,u_input):
        lam_mean = self.lam_mean(u_input)
        lam_log_var = self.lam_log_var(u_input)
        return torch.squeeze(input=lam_mean,dim=1),torch.squeeze(input=lam_log_var,dim=1)

class Z_Prior_Nn(nn.Module):
    """
        Compute the prior mean and log of variance of prior p(z|u) for continuous u. 
        We assume p(z|u) as gaussian distribution with mean and log of variance parameterized by feed-forward neural network as a function of u.
    """
    def __init__(self,u_dim,z_dim):
        super(Z_Prior_Nn,self).__init__()
        self.z_dim = z_dim
        n_hidden_nodes_in_prior = 20
        n_nodes = [u_dim,n_hidden_nodes_in_prior, n_hidden_nodes_in_prior, 2*z_dim]
        act_func = ['tanh', 'tanh', 'linear']
        n_layers = len(n_nodes) - 1
        self.module = nn.Sequential()
        for ii in range(n_layers):
            self.module.add_module(
                        name = f"linear_{ii}",
                        module= nn.Linear(n_nodes[ii],n_nodes[ii+1])
                        )
            if act_func[ii] == 'tanh':
                self.module.add_module(name = f"act_{ii}",module= nn.Tanh())
    def forward(self,u_input):
        output = u_input    
        for module in self.module:
            output = module(output)
        lam_mean = output[:,:self.z_dim]
        lam_log_var = output[:,self.z_dim:]
        
        return lam_mean,lam_log_var

def compute_posterior(args):
    """
        Compute the full posterior of q(z|x, u). We assume that q(z|x, u) \prop q(z|x)*p(z|u). Both q(z|x) and p(z|u) are gaussian distributed.
        
        Arguments:
            args (tensor): mean and log of variance of q(z|x) and p(z|u)  
            
        Returns:
            mean and log of variance of q(z|x, u) (tensor)
    """
    z_mean, z_log_var, lam_mean, lam_log_var = args
    post_mean = (z_mean/(1+torch.exp(z_log_var-lam_log_var))) + (lam_mean/(1+torch.exp(lam_log_var-z_log_var)))
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))
    
    return [post_mean, post_log_var]


def sampling(args):
    """
        Reparameterization trick by sampling from an isotropic unit Gaussian.

        Arguments:
            args (tensor): mean and log of variance of q(z|x)

        Returns:
            z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch =torch.tensor(z_mean.shape[0]).to(device)
    dim = torch.tensor(z_mean.shape[1]).to(device)
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = torch.randn([batch,dim]).to(device)
    
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon



class First_Nflow_Layer(nn.Module):
    def __init__(self,dim_x,dim_z,min_gen_nodes=30):
        super(First_Nflow_Layer,self).__init__()
        gen_nodes = max(min_gen_nodes,dim_x//4)
        
        n_nodes = [dim_z,gen_nodes,gen_nodes,dim_x-dim_z]
        act_func = ['relu','relu','linear']
        n_layers = len(n_nodes) -1
        self.module = nn.Sequential()
        for ii in range(n_layers):
            self.module.add_module(
                    name = f"linear_{ii}",
                    module= nn.Linear(n_nodes[ii],n_nodes[ii+1])
                )
            if act_func[ii] == 'relu':
                self.module.add_module(
                        name = f"act_{ii}",
                        module= nn.ReLU()
                        )
    def forward(self,z_input):
        output = z_input
        for module in self.module:
            output = module(output)
        return torch.concat([z_input,output],-1)


class Affine_Coupling_Layer(nn.Module):
    """
        Define each affine_coupling_layer, which maps input x to [x_{1:dd}, x_{dd+1:n} * exp(s(x_{1:dd})) + t(x_{1:dd})].
    """
    def __init__(self,layer_input_dim,min_gen_nodes=30,dd=None):
        super(Affine_Coupling_Layer,self).__init__()
        DD = layer_input_dim
        if dd is None:
            dd = (DD // 2)
        self.dd = dd
        self.DD = DD
        
        n_nodes = [ dd,
                    max(min_gen_nodes,DD//4),
                   max(min_gen_nodes,DD//4),
                  2*(DD-dd)-1]
        act_func = ['relu', 'relu', 'linear'];
        self.module = nn.Sequential()
        
        for ii in range(3):
            self.module.add_module(
                    name = f"linear_{ii}",
                    module= nn.Linear(n_nodes[ii],n_nodes[ii+1])
                )
            if act_func[ii] == 'relu':
                self.module.add_module(
                    name = f"act_{ii}",
                    module= nn.ReLU()
                )
    def forward(self,layer_input):
        x_input1 = layer_input[:,:self.dd];
        x_input2 = layer_input[:,self.dd:self.dd+self.DD-self.dd];
        st_output = x_input1
        for module in self.module:
            st_output = module(st_output)
        s_output = st_output[:,:self.DD-self.dd-1]
        t_output = st_output[:,self.DD-self.dd-1:self.DD-self.dd]
        s_output =  torch.tanh(s_output) * 0.1 
        s_output = torch.concat([s_output,torch.sum(-1.* s_output,dim=-1,keepdim=True)],-1)
        trans_x =  x_input2 * torch.exp(s_output) + t_output
        output = torch.concat([trans_x,x_input1],-1)
        return output
        
        
class Affine_Coupling_Block(nn.Module):
    def __init__(self,x_dim,min_gen_nodes=30,dd=None):
        super(Affine_Coupling_Block,self).__init__() 
        self.module = nn.Sequential(Affine_Coupling_Layer(x_dim,min_gen_nodes,dd))
    def forward(self,x):
        for module in self.module:
            x = module(x)
        return x
        

class Decoder_Nflow(nn.Module):
    """
        Define mean(p(x|z)) using GIN volume preserving flow.
    """
    def __init__(self,x_dim,z_dim,mdl,min_gen_nodes,n_blk,dd=None):
        super(Decoder_Nflow,self).__init__()
        
        self.permute_ind = []
        self.mdl = mdl
        
        for ii in range(n_blk):
            np.random.seed(ii)
            self.permute_ind.append(
                torch.tensor(np.random.permutation(x_dim))
            )
        self.first_nflow_layer = First_Nflow_Layer(x_dim,z_dim,min_gen_nodes)
        self.affine_coupling_blocks = nn.Sequential()
        for ii in range(n_blk):
            self.affine_coupling_blocks.add_module(
                           name = f"affine_coupling_block_{ii}",
                    module = Affine_Coupling_Block(x_dim,min_gen_nodes,dd)
                        )
        
    def forward(self,z_input):
        output = self.first_nflow_layer(z_input).to(device)
        for ii,module in enumerate(self.affine_coupling_blocks):
            output = torch.gather(output,-1,self.permute_ind[ii].repeat(output.shape[0],1).to(device))
            output = module(output)
        if self.mdl == 'poisson':
            output = torch.nn.functional.softplus(output)
        return output

    