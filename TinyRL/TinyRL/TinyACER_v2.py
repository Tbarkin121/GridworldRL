# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:06:31 2023

@author: MiloPC
"""

import torch
import TinyGame

buffer_len = 1000;
num_envs = 1000
siz = 9
gamma = 0.9
ent_coef = 0.0
vf_coef = 0.5
num_epochs = 1000
num_q_holds = 1
num_inner_steps = 1
torch.set_default_device('cuda')

#%%
# Define model
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_linear = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU()
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(128, 1)
        )
        self.actor_head = torch.nn.Sequential(
            torch.nn.Linear(128, 5)
        )


    def forward(self, x):
        x = self.shared_linear(x)
        state_values = self.value_head(x)
        state_values = self.actor_head(x)
        action_prob = torch.nn.functional.softmax(action_prob, 1)

        # out1 = torch.nn.functional.relu(out1)

        return state_values, state_values
    
#%%
model = Policy()
rand_obs = 2*torch.rand((10,4))-1

model_out = model(rand_obs)
print(model_out)
#%%
PolicyNet1 = Policy()
PolicyNet2 = Policy()
PolicyNet2.eval()

optimizer_Policy = torch.optim.Adam(PolicyNet1.parameters(), lr=1e-3)
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
            optimizer_Policy.zero_grad()
            [vals,probs] = PolicyNet1(obs)
            

            returns = gamma*PolicyNet2(obs_prime)[0] + torch.reshape(reward,[-1,1])
            
            advantage = returns - vals
            prob_act = probs.gather(1, actions.reshape([-1,1]))
            log_probs = torch.log(prob_act)
            
            value_loss = torch.nn.functional.mse_loss(returns.squeeze(),vals.squeeze())
            
            # entropy_loss = torch.mean(-log_probs)
            entropy_loss = -torch.mean(prob_act*log_probs)
            
            # policy_loss = -torch.mean(log_probs*advantage)
            policy_loss = torch.mean(-log_probs*advantage)
            
            loss = policy_loss + ent_coef*entropy_loss + vf_coef*value_loss
            
           
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(PolicyNet1.parameters(), 0.5)
            optimizer_Policy.step()
            
   
        next_actions = torch.distributions.categorical.Categorical(probs=probs[:env.num_envs]).sample()
        # next_actions = torch.max(probs,dim=1)[1][:env.num_envs]
        # next_actions[epsilon_idx] = rand_actions
        
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

env_render = TinyGame.simplEnv(num_envs,siz,buffer_len, render_mode="pygame", window_size = 1024, font_size = 24)
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
    
