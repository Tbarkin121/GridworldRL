# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:06:31 2023

@author: MiloPC
"""

import torch
import torch.nn as nn
from torch import optim
import TinyGame

buffer_len = 100000;
num_envs = 10000
siz = 9
gamma = 0.99
ent_coef = 0.0
vf_coef = 0.5
num_epochs = 1000
num_q_holds = 1
num_inner_steps = 1
torch.set_default_device('cpu')

#%%
# Define model
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = 4
        n_actions = 9
        
        critic_layers = [
                        nn.Linear(n_features, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),  # estimate V(s)
                    ]
                    
        actor_layers = [
                        nn.Linear(n_features, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(
                            32, n_actions
                        ), 
                        nn.Softmax(),
                    ]
        
        
        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)
        
        
        # self.linear_relu_stack = torch.nn.Sequential(
        #     torch.nn.Linear(4, 128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.LeakyReLU()
        # )
        # self.linear_head1 = torch.nn.Sequential(
        #     torch.nn.Linear(128, 1)
        # )
        # self.linear_head2 = torch.nn.Sequential(
        #     torch.nn.Linear(128, 9)
        # )


    def forward(self, x):
        # out0 = self.linear_relu_stack(x)
        # out1 = self.linear_head1(out0)
        # out2 = self.linear_head2(out0)
        # out2 = torch.nn.functional.softmax(out2, 1)
        
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

        # out1 = torch.nn.functional.relu(out1)

        return out1, out2
    
#%%
a = torch.rand([3,4])
PolicyNet1 = Policy()

critic_optim = optim.Adam(PolicyNet1.critic.parameters(), lr=1e-3)
actor_optim = optim.Adam(PolicyNet1.actor.parameters(), lr=1e-3)
        
# optimizer_Policy = torch.optim.Adam(PolicyNet1.parameters(), lr=1e-3)

PolicyNet2 = Policy()
PolicyNet2.eval()

mse_loss = torch.nn.MSELoss()

#%%

env = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 1024, font_size = 24)
# env = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 1024, font_size = 12)

a = torch.linspace(0,env.siz-1,env.siz)
grid = torch.meshgrid(a,a)
grid_coords = torch.concat([grid[0].reshape([-1,1]),grid[1].reshape([-1,1])],1)

for epoch in range(num_epochs):

    PolicyNet2.load_state_dict(PolicyNet1.state_dict())
    
    for i in range(num_q_holds):
        obs, actions, reward, obs_prime = env.get_SARS() 
        
        for _ in range(num_inner_steps):
            mask = 1-torch.reshape(reward,[-1,1])
            [vals,probs] = PolicyNet1(obs)
            returns = mask*gamma*PolicyNet2(obs_prime)[0] + torch.reshape(reward,[-1,1])
            td_error = returns - vals
            
            advantage = returns - vals
            
            # calculate the loss of the minibatch for actor and critic
            critic_loss = advantage.pow(2).mean()
        
            prob_act = probs.gather(1, actions.reshape([-1,1]))
            log_probs = torch.log(prob_act)
            
            actor_loss = -(advantage.detach() * log_probs).mean()
            
            
            
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            
            
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
   
        # next_actions = torch.distributions.Categorical(probs=probs[:env.num_envs]).sample()
        action_pd = torch.distributions.Categorical(logits=probs[:env.num_envs])
        next_actions = action_pd.sample()
        # action_log_probs = action_pd.log_prob(actions)
        # entropy = action_pd.entropy()
        
        env.update(next_actions)
        
        
    goal_coord = torch.unsqueeze(env.states[0,2:4],0)
    sample_states = torch.concat([grid_coords,goal_coord*torch.ones([env.siz**2,1])],1)
    sample_obs = sample_states*2./(env.siz-1.)-1.
    sampleVals,sampleProbs = PolicyNet1(sample_obs)
    max_sampleProbs = torch.max(sampleProbs,dim=1)
    
    
    
    action_map = max_sampleProbs[1].reshape(env.siz,env.siz).cpu().numpy()
    env.render(0, action_map)
    print(epoch)
    # print(Qvals.reshape([siz,siz]))
    
        

#%%
import time

view_len = 1000

env_render = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 512, font_size = 24)
env_view_id = 0
for i in range(view_len):

    print(i)
    goal_coord = torch.unsqueeze(env_render.states[env_view_id,2:4],0)
    map_states = torch.concat([grid_coords,goal_coord*torch.ones([env_render.siz**2,1])],1)
    map_obs = map_states*2./(env_render.siz-1.)-1.
    map_probs = PolicyNet1(map_obs)[1]
    map_actions = torch.max(map_probs,dim=1)[1]
    
    actions_map = map_actions.reshape(env_render.siz,env_render.siz).cpu().numpy()
    env_render.render(env_view_id, actions_map)
    
    print(map_probs)

    obs_inf = env_render.states*2./(env_render.siz-1.)-1.
    probs_inf = PolicyNet1(obs_inf)[1]
    actions_inf = torch.max(probs_inf,dim=1)[1]
    env_render.update(actions_inf)



# env_render.close()
    
