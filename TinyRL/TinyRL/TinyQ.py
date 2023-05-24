# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:06:31 2023

@author: MiloPC
"""

import torch
import TinyGame

buffer_len = 100;
num_envs = 10
siz = 11
gamma = 1.0
epsi = 0.0
num_epochs = 1000
num_q_holds = 1
torch.set_default_device('cuda')

#%%
# Define model
class Q_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 9),

           
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
#%%

env = TinyGame.simplEnv(num_envs,siz,buffer_len)
net1 = Q_nn()
optimizer = torch.optim.Adam(net1.parameters(), lr=1e-3)

# with torch.no_grad():
net2 = Q_nn()
# net2.training = False
net2.eval()


#%%


for epoch in range(num_epochs):

    net2.load_state_dict(net1.state_dict())
    net2.eval()
    # net2.training = False
    
    for i in range(num_q_holds):

        optimizer.zero_grad()
        obs_1 = env.buffer.s1*2./(env.siz-1.)-1.
        obs_2 = env.buffer.s2*2./(env.siz-1.)-1.

    
        Q_vals_1 = net1(obs_1)
        Q_vals_2 = net2(obs_2)
    
        Y_0 = Q_vals_1.gather(1,torch.reshape(env.buffer.a,[-1,1]))
    
    
        maxQ = torch.max(Q_vals_2,dim=1)
    
        reward = env.buffer.r*(1.0 - 1.0*env.buffer.d)
        Y_1 = torch.reshape(gamma*maxQ[0] + reward,[-1,1])
        loss = torch.mean((Y_0 - Y_1)**2)
    
    
        loss.backward()
        optimizer.step()
            
        epsilon_idx = torch.rand([env.num_envs,]) > epsi
        rand_actions = torch.randint(0,env.num_actions,[torch.sum(epsilon_idx),1])
        actions = maxQ[1][:env.num_envs]
        actions[epsilon_idx.nonzero()] = rand_actions
    
        env.update(actions)
    
    a = torch.linspace(0,env.siz-1,env.siz)
    grid = torch.meshgrid(a,a)
    grid_coords = torch.concat([grid[0].reshape([-1,1]),grid[1].reshape([-1,1])],1)
    goal_coord =torch.tensor([[(env.siz-1)/2 ,(env.siz-1)/2]])
    sample_states = torch.concat([grid_coords,goal_coord*torch.ones([env.siz**2,1])],1)
    sample_obs = sample_states*2./(env.siz-1.)-1.
    Qvals = net(sample_states)
    MaxQ = torch.max(Qvals,dim=1)
    action_world = MaxQ[1].reshape(env.siz,env.siz).cpu().numpy() +1
    
    print(action_world)
    
    print(epoch,loss.detach().cpu().numpy(),torch.mean(maxQ[0]).detach().cpu().numpy())
        

#%%
view_len = 100
history = torch.zeros(view_len,env.siz,env.siz)

env = TinyGame.simplEnv(num_envs,siz,buffer_len)

for i in range(view_len):
    
    history[i,env.states[0,0].int(),env.states[0,1].int()] = 1.0;
    history[i,env.states[0,2].int(),env.states[0,3].int()] = -1.0;
    
    obs = env.states*2./(env.siz-1.)-1.
    
    Q_vals = net(env.states)
    
    # print(env.states[0])
    
    maxQ = torch.max(Q_vals,dim=1)
    actions = maxQ[1]
    env.update(actions)
    # print(actions[0])
history_cpu = history.cpu()
history_np = history_cpu.numpy()