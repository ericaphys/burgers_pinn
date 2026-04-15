PINN model using Pytorch to solve the inviscid Burgers' equation for x in [-1,1] and t in [0,1]. Within this time frame a 
schock wave is expected and therefore numerical instability has to be managed.
A network with 4 layers of 20 neurons each and a loss function modified to satisfy the PDE has been implemented. The learning rate decays with a
scheduler, the weights of the loss function components vary during training.
Initial conditions are u(x,0)=-sin(pi*x) and boundary conditions are u(1,t)=u(-1,t)=0. Collocation points are generated uniformly.
Two optimizers are used in combination: Adam for the first 10k epochs and L-BFGS for the remaining. Norm of gradients is then clipped
to avoid explosion.
