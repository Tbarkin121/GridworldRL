# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:06:31 2023

@author: MiloPC
"""

import torch
import TinyGame

buffer_len = 100000; #10k environments, but store 100k state transitions in buffer. 
#Can re-operate on old data since its saved, is like the last ten time steps of data in this case
#Experience replay buffer
#predicted reward across the buffer means: 
num_envs = 10000
siz = 11
gamma = 0.7
epsi = 0.7
num_epochs = 100
num_q_holds = 1
num_inner_steps = 100
torch.set_default_device('cuda')

#%%
# Define model
class Q_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 9),

           
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
#%%

net1 = Q_nn()
optimizer = torch.optim.Adam(net1.parameters(), lr=1e-3)

net2 = Q_nn()
net2.eval()


#%%

env = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode='pygame', window_size = 1024, font_size = 24)
# env = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 1024, font_size = 12)

for epoch in range(num_epochs):

    net2.load_state_dict(net1.state_dict())
    # net2.eval()
    
    for i in range(num_q_holds):
        obs_1, action_1, reward_1, obs_2 = env.get_SARS() 
        
        for _ in range(num_inner_steps):
            optimizer.zero_grad()
                
            Q_vals_1 = net1(obs_1)
            Q_vals_2 = net2(obs_2)
        
            Y_0 = Q_vals_1.gather(1,torch.reshape(action_1,[-1,1]))    
            maxQ = torch.max(Q_vals_2,dim=1)
            Y_1 = torch.reshape(gamma*maxQ[0] + reward_1,[-1,1])
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
    # goal_coord =torch.tensor([[(env.siz-1)/2 ,(env.siz-1)/2]])
    goal_coord = torch.unsqueeze(env.states[0,2:4],0)
    sample_states = torch.concat([grid_coords,goal_coord*torch.ones([env.siz**2,1])],1)
    sample_obs = sample_states*2./(env.siz-1.)-1.
    Qvals = net1(sample_obs)
    MaxQ = torch.max(Qvals,dim=1)
    action_world = MaxQ[1].reshape(env.siz,env.siz).cpu().numpy()
    env.render(0, action_world)
    
    # print(action_world)
    
    print(epoch,loss.detach().cpu().numpy(),torch.mean(maxQ[0]).detach().cpu().numpy())
        

#%%
import time
None
view_len = 1000

env_render = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 1024, font_size = 24)
env_view_id = 0
for i in range(view_len):
    a = torch.linspace(0,env_render.siz-1,env_render.siz)
    grid = torch.meshgrid(a,a)
    grid_coords = torch.concat([grid[0].reshape([-1,1]),grid[1].reshape([-1,1])],1)
    goal_coord = torch.unsqueeze(env_render.states[env_view_id,2:4],0)
    sample_states = torch.concat([grid_coords,goal_coord*torch.ones([env_render.siz**2,1])],1)
    sample_obs = sample_states*2./(env_render.siz-1.)-1.
    Qvals = net1(sample_obs)
    MaxQ = torch.max(Qvals,dim=1)
    
    action_world = MaxQ[1].reshape(env_render.siz,env_render.siz).cpu().numpy()
    env_render.render(env_view_id, action_world)
    

    obs = env_render.states*2./(env_render.siz-1.)-1.
    Q_vals = net1(obs)
    
    # print(env_render.states[0])
    
    maxQ = torch.max(Q_vals,dim=1)
    actions = maxQ[1]

    # print('-----')
    # print(env_render.states[env_view_id,...])
    # print('numpad action : {}'.format(actions[env_view_id].detach().cpu().numpy()+1))
    env_render.update(actions)
    # print(env_render.states[env_view_id,...])
    
    # print('.......')
    # print(obs[0,...])
    # print(actions)
    # print()
    

    # time.sleep(1)
""
    # print(actions[0])

# env_render.close()
    
