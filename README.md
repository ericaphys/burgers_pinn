# Physics informed neural network (PINN) for Burgers' equation
<p>This project contains a PINN model implemented from scratch using Pytorch, oriented to solving the Burgers' equation for x in [-1,1] and t in [0,1].</p>
<p> Two separate functions, one for the inviscid case and one with the viscosity term were added for clarity during developing; they obviously reduce to each other for eta=0.</p>
<p>For the inviscid case, within the considered time frame (at approximately tt=1/pi) a schock wave is expected and therefore numerical instability has to be managed.</p>
<p>A network with 4 layers of 20 neurons each and a loss function modified to satisfy the PDE has been implemented. The learning rate decays with a
scheduler, the weights of the loss function components vary during training. Two optimizers are used in combination: Adam for the first 10k epochs and L-BFGS for the remaining. Norm of gradients is then clipped to avoid explosion.</p>
<p>Initial conditions are u(x,0)=-sin(pi*x) and boundary conditions are u(1,t)=u(-1,t)=0. Collocation points are generated uniformly.</p>

