import torch 
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numba import jit


#MLP with pytorch
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1=nn.Linear(input_size, hidden_size)
        self.layer2=nn.Linear(hidden_size, 1)

    def forward(self, x):
        x=torch.tanh(self.layer1(x))
        x=self.layer2(x)

        return x
    
def inviscid_burgers(u, du_dx, du_dt):
    residual=du_dt+u*du_dx
    return (residual**2).mean()


def main():
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if device=='xpu':
        device='cpu'
    print(f"Using {device} device")
    


    #initializing data for the MLP
    rng=np.random.RandomState(123)
    #initial conditions 50 xi for t=0 where u(x,0)=sin(pi*x)
    Ni=50
    x_t_in=np.zeros((Ni,2))
    x_t_in[:,0]=rng.uniform(-1,1,size=Ni)
    x_t_in[:,1]=0
    u_in=np.sin(np.pi*x_t_in[:,0])
    

    #boundary conditions 25 ti for x=-1 and 25 ti for x=1
    Nb=25
    x_t_bound_dx=np.zeros((Nb,2))
    x_t_bound_sx=np.zeros((Nb,2))
    x_t_bound_dx[:,1]=rng.uniform(0,1 ,size=Nb).astype(np.float32)
    x_t_bound_dx[:,0]=1
    x_t_bound_sx[:,1]=rng.uniform(0,1 ,size=Nb).astype(np.float32)
    x_t_bound_sx[:,0]=-1
    x_t_bound=np.concatenate([x_t_bound_sx ,x_t_bound_dx], axis=0)
    u_bound=np.zeros(2*Nb)
    

    #initialization of N random tuples (x,t) normally distributed
    N=10000
    x_t_tuple=rng.normal(loc=0.0, scale=0.1, size=(N, 2)).astype(np.float32)
    u_tuple=np.empty(N)
    u_tuple[:]=np.nan
    u_tuple=u_tuple.astype(np.float32)
    
    x_t_in_bound=np.concatenate([x_t_in, x_t_bound], axis=0)
    u_in_bound=np.concatenate([u_in, u_bound])
    #print(x_t.shape)

    x_t=np.concatenate([x_t_in_bound, x_t_tuple], axis=0)
    
    print(u_in_bound.shape)
    print(u_tuple.shape)
    u=np.concatenate([u_in_bound, u_tuple])

    #pytorching
    x_t=torch.from_numpy(x_t.astype(np.float32)).requires_grad_()
    u=torch.from_numpy(u.astype(np.float32)).requires_grad_()

    train_ds=TensorDataset(x_t, u)
    torch.manual_seed(1)
    batch_size=100
    train_dl=DataLoader(train_ds, batch_size,shuffle=True)
    input_size=2
    hidden_size=100
    eta=0.001
    epochs=50

    model=Model(input_size, hidden_size)
    model.to(device)

    loss_fn=nn.MSELoss()

    optimizer=torch.optim.Adam(model.parameters(), lr=eta)

    #-----MLP-----
    losses=np.zeros(epochs, dtype=np.float32)
    lambda_data=100.0
    lambda_phy=1.0
    p_losses=np.zeros(epochs, dtype=np.float32)
    for epoch in range(epochs):
        for x_batch, u_batch in train_dl:
            optimizer.zero_grad()
            u_batch=u_batch.reshape(-1,1)
            x_batch, u_batch = x_batch.to(device).requires_grad_(), u_batch.to(device)
            pred=model(x_batch)

            #nan mask
            mask=~torch.isnan(u_batch.squeeze())
            
            loss=loss_fn(pred[mask], u_batch[mask])
            
            #---- automatic differentiation ----
            derivatives = torch.autograd.grad(pred, x_batch, grad_outputs=torch.ones_like(pred),  create_graph=True)[0]
            du_dx=derivatives[:,0]
            du_dt=derivatives[:,1]
            pred=pred.squeeze()
            phy_loss=inviscid_burgers(pred[~mask], du_dx, du_dt)
            loss=lambda_data*loss+lambda_phy*phy_loss
            loss.backward()
            optimizer.step()
            

            losses[epoch] += loss.item()*u_batch.size(0)
            p_losses[epoch] += phy_loss.item()*u_batch.size(0)
        losses[epoch] /= len(train_dl.dataset)
        p_losses[epoch] /= len(train_dl.dataset)
        print(f"Epoch: {epoch+1}/{epochs}\ntot loss : {losses[epoch]:.5f}\n phys loss : {p_losses[epoch]:.5f}\n----------------------------\n")




if __name__=='__main__':
    main()