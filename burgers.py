import torch 
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as FuncAnimation


#MLP with pytorch
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1=nn.Linear(input_size, hidden_size)
        self.layer2=nn.Linear(hidden_size, hidden_size)
        self.layer3=nn.Linear(hidden_size, hidden_size)
        self.layer4=nn.Linear(hidden_size,1)

    def forward(self, x):
        x=torch.tanh(self.layer1(x))
        x=self.layer2(x)
        x=torch.tanh(self.layer2(x))
        x=self.layer3(x)
        x=torch.tanh(self.layer3(x))
        x=self.layer4(x)

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
    epochs=1000

    model=Model(input_size, hidden_size)
    model.to(device)

    loss_fn=nn.MSELoss()

    optimizer=torch.optim.Adam(model.parameters(), lr=eta)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    #-----MLP-----
    losses=np.zeros(epochs, dtype=np.float32)
    lambda_data=1000.0
    lambda_phy=1.0
    p_losses=np.zeros(epochs, dtype=np.float32)
    burgers=[]
    plot_x=[]
    for epoch in range(epochs):
        for x_batch, u_batch in train_dl:
            optimizer.zero_grad()
            u_batch=u_batch.reshape(-1,1)
            x_batch, u_batch = x_batch.to(device).requires_grad_(), u_batch.to(device)
            pred=model(x_batch)
            
            #nan mask -> using it returns true where there isnt a nan in u_batch
            mask=~torch.isnan(u_batch.squeeze())
            if epoch==(epochs-1):
                burgers.append(pred[~mask].detach().cpu())
                plot_x.append(x_batch[~mask].detach().cpu())
            if mask.any():
                loss=loss_fn(pred[mask], u_batch[mask])
            else:
                loss=torch.tensor(0.0, device=device)
            
            #---- automatic differentiation ----
            derivatives = torch.autograd.grad(pred, x_batch, grad_outputs=torch.ones_like(pred),  create_graph=True)[0]
            du_dx=derivatives[:,0]
            du_dx=du_dx[~mask]
            du_dt=derivatives[:,1]
            du_dt=du_dt[~mask]
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
        scheduler.step()

    burgers=torch.cat(burgers).numpy()
    plot_x=torch.cat(plot_x).numpy()
    burgers=burgers.flatten()

    #ordering for time
    x_data=plot_x[:,0]
    t_data=plot_x[:,1]

    frames=50
    time_steps=np.linspace(0,1,frames)
    #tolerance for t value
    tol=1/frames

    fig, ax=plt.subplots()
    line=ax.plot([],[], lw=2)
    ax.set_xlim(-1,1)
    ax.set_ylim(burgers.min()-0.1, burgers.max()+0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('velocity')

    def init():
        line.set_data([],[])
        return line,

    def update(t_val):
        time_mask=np.abs(t_data-t_val)<tol

        if np.any(mask):
            x_frame=x_data[mask]
            u_frame=burgers[mask]

            #sorting
            idx=np.argsort(x_frame)
            line.set_data(x_frame[idx], u_frame[idx])
        return line
    ani=FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True)

    ani.save('plot.mp4', fps=15)

    plt.close(fig)
    
    
    data=np.column_stack((plot_x[:,0], burgers))
    with open('output.txt', 'a') as f:
        np.savetxt(f,data)

if __name__=='__main__':
    main()