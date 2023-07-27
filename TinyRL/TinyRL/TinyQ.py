import torch
import TinyGame
import yaml

# Define model
class Q_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(10, 128), #changing 4 to ten
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 4), #idk why

           
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x) #this is the nn defined above
        return logits
#CLASS ENDS HERE    
class Training():
    def __init__(self):
        with open("gridworld_params.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        # print(self.cfg)

        self.device = self.cfg["env"]["device"] #sets it to cuda
        torch.set_default_device(self.device)

        self.board_size = self.cfg["env"]["board_size"] #gets board size from yaml (51) square grid??
        self.num_envs = self.cfg["train_param"]["num_envs"] #gets num_env from yaml

        self.buffer_len = self.cfg["train_param"]["buffer_len"] #buffer length seems to be set wrong ehre
        self.gamma = self.cfg["train_param"]["gamma"] 
        self.epsi = self.cfg["train_param"]["epsi"]
        self.num_epochs = self.cfg["train_param"]["num_epochs"]
        self.num_q_holds = self.cfg["train_param"]["num_q_holds"]
        self.num_inner_steps = self.cfg["train_param"]["num_inner_steps"]
        self.env_view_id = self.cfg["debug"]["env_view_id"]

        self.net1 = Q_nn() #initializes the class above
        self.optimizer = torch.optim.Adam(self.net1.parameters(), lr=self.cfg["train_param"]["learning_rate"])
        self.net2 = Q_nn()
        self.net2.eval() #why evaluate the second net first? I guess it will be the states first done?? Or maybe a first pick so the game can start up with the code
        self.env_train = TinyGame.simplEnv(self.num_envs,self.board_size,self.buffer_len, render_mode="pygame", window_size = 1024, font_size = 12)
        self.env_eval = TinyGame.simplEnv(self.num_envs,self.board_size,self.buffer_len, render_mode="pygame", window_size = 1024, font_size = 12)

    def train(self):
        a = torch.linspace(0,self.env_train.siz-1,self.env_train.siz)
        grid = torch.meshgrid(a,a)
        grid_coords = torch.concat([grid[0].reshape([-1,1]),grid[1].reshape([-1,1])],1)
        
        for epoch in range(self.num_epochs):

            self.net2.load_state_dict(self.net1.state_dict())
            # net2.eval()
            
            for i in range(self.num_q_holds):
                obs_1, action_1, reward_1, obs_2 = self.env_train.get_SARS() 
                
                for _ in range(self.num_inner_steps):
                    self.optimizer.zero_grad()
                        
                    Q_vals_1 = self.net1(obs_1)
                    Q_vals_2 = self.net2(obs_2)
                
                    Y_0 = Q_vals_1.gather(1,torch.reshape(action_1,[-1,1]))    
                    maxQ = torch.max(Q_vals_2,dim=1)
                    Y_1 = torch.reshape(self.gamma*maxQ[0] + reward_1,[-1,1])
                    loss = torch.mean((Y_0 - Y_1)**2)
                
                    
                    loss.backward()
                    self.optimizer.step()
                        
                epsilon_idx = torch.rand([self.env_train.num_envs,]) > self.epsi
                rand_actions = torch.randint(0,self.env_train.num_actions,[torch.sum(epsilon_idx),])
                actions = maxQ[1][:self.env_train.num_envs]
                actions[epsilon_idx] = rand_actions
                
                self.env_train.update(actions)
            
            
            goal_coord = torch.unsqueeze(self.env_eval.states[self.env_view_id,2:4],0)
            # sample_states = torch.concat([grid_coords,goal_coord*torch.ones([self.env_eval.siz**2,1])],1)
            # sample_obs = sample_states*2./(self.env_train.siz-1.)-1.
            # Qvals = self.net1(sample_obs)
            # MaxQ = torch.max(Qvals,dim=1)
            # action_map = MaxQ[1].reshape(self.env_train.siz,self.env_train.siz).cpu().numpy()
            self.env_train.render(0)
            
            # print(action_world)
            
            print(epoch,loss.detach().cpu().numpy(),torch.mean(maxQ[0]).detach().cpu().numpy())

    # def test(self, test_len):
    #     import time
        
    #     a = torch.linspace(0,self.env_eval.siz-1,self.env_eval.siz)
    #     grid = torch.meshgrid(a,a)
    #     grid_coords = torch.concat([grid[0].reshape([-1,1]),grid[1].reshape([-1,1])],1)
    #     for i in range(test_len):
            
    #         goal_coord = torch.unsqueeze(self.env_eval.states[self.env_view_id,2:4],0)
    #         sample_states = torch.concat([grid_coords,goal_coord*torch.ones([self.env_eval.siz**2,1])],1)
    #         sample_obs = sample_states*2./(self.env_eval.siz-1.)-1.
    #         Qvals = self.net1(sample_obs)
    #         MaxQ = torch.max(Qvals,dim=1)
            
    #         action_map = MaxQ[1].reshape(self.env_eval.siz,self.env_eval.siz).cpu().numpy()
    #         self.env_eval.render(self.env_view_id, action_map)
            

    #         obs = self.env_eval.states*2./(self.env_eval.siz-1.)-1.
    #         Q_vals = self.net1(obs)
            
    #         # print(self.env_eval.states[0])
            
    #         maxQ = torch.max(Q_vals,dim=1)
    #         actions = maxQ[1]

    #         # print('-----')
    #         # print(self.env_eval.states[env_view_id,...])
    #         # print('numpad action : {}'.format(actions[env_view_id].detach().cpu().numpy()+1))
    #         self.env_eval.update(actions)
    #         # print(self.env_eval.states[env_view_id,...])
            
    #         # print('.......')
    #         # print(obs[0,...])
    #         # print(actions)
    #         # print()
            

    #         # time.sleep(1)
    #     ""
    #         # print(actions[0])

    #     # self.env_eval.close()
    
grid_world = Training()
grid_world.train()
# grid_world.test(100)

