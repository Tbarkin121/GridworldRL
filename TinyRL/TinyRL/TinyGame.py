import torch
import pygame
import pygame.freetype

class Buffer():
  def __init__(self,buffer_len,num_actions):
    self.s1= torch.zeros([buffer_len,4]) #state 1
    self.a= torch.randint(0,num_actions,[buffer_len,]) 
    self.r= torch.zeros([buffer_len,]) #reward maybe?
    self.s2 = torch.zeros([buffer_len,4]) #state 2
    self.d = torch.zeros([buffer_len,])==1 #not sure


class simplEnv():
    metadata = {"render_modes": ["pygame"], "render_fps": 4}
    def __init__(self, num_envs, siz, buffer_len, render_mode = None, window_size = 512, font_size=36):
        self.num_envs = num_envs
        self.siz = siz
        self.buffer_len = buffer_len
        # self.action_table =  torch.tensor([[-1, 1], #remove
        #                                     [0, 1],
        #                                     [1, 1], #remove
        #                                     [-1, 0],
        #                                     [0, 0],
        #                                     [1, 0],
        #                                     [-1, -1],#remove
        #                                     [0, -1],
        #                                     [1, -1]])*1.0 #remove
        
        self.action_table =  torch.tensor([[0, 1],
                                            [-1, 0],
                                            [1, 0],
                                            [0, -1]])*1.0 #remove
        
                
        self.num_actions = self.action_table.shape[0] #I guess this is the first entry in returned shape. Must be 9x1 
        self.states = torch.zeros([self.num_envs,4]) #4 states present, forget what they are here to be honest
        self.max_episode_length = 100
        self.episode_time = torch.zeros([self.num_envs,]) #I dont get this line. I guess 1 time interation for each environment
        self.get_reset_idx() #declairng this and defining it later?
        self.reset_env() #same as above
        self.buffer = Buffer(self.buffer_len,self.num_actions) #creates a buffer with desired atributes?
        self.fill() #define later check it out
        
        # Pygames Stuff
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None     #no window might be a problem here 
        self.clock = None     
        if(self.render_mode == "pygame"):
            pygame.init()
            pygame.display.init()
            self.font = pygame.font.SysFont("Arial", font_size)
            self.board_size = self.siz
            self.window_size = window_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))        
            self.clock = pygame.time.Clock()
            self.action_ascii = ['↓','←','→','↑']



        
    def fill(self): # I think this might be actual game loop?
        num = torch.ceil(torch.tensor([self.buffer_len/self.num_envs])).int()
        for i in range(num):
            actions = torch.randint(0,self.num_actions,[self.num_envs,])
            self.update(actions)
            

    def update(self,actions):
        with torch.no_grad():
            self.buffer.s1 = self.buffer.s1.roll(self.num_envs,0)
            self.buffer.s1[0:self.num_envs,] = self.states
            self.buffer.a = self.buffer.a.roll(self.num_envs,0)
            self.buffer.a[0:self.num_envs,] = actions
            
            acts = torch.matmul(torch.nn.functional.one_hot(actions, num_classes = self.num_actions)*1.0, self.action_table)
            
            self.states =  self.states + torch.concat([torch.reshape(acts,[self.num_envs,2]),torch.zeros([self.num_envs,2])],1)
            self.states = torch.clip(self.states,0,self.siz-1)
            
            self.get_reset_idx()
            self.get_rewards()
            
            self.buffer.d = self.buffer.d.roll(self.num_envs,0)
            self.buffer.d[0:self.num_envs,] = self.reset_idx
            self.buffer.r = self.buffer.r.roll(self.num_envs,0)
            self.buffer.r[0:self.num_envs,] = self.rewards
            self.buffer.s2 = self.buffer.s2.roll(self.num_envs,0)
            self.buffer.s2[0:self.num_envs,] = self.states
            
            
            self.reset_env()
            self.episode_time[:] = self.episode_time[:] + 1
            
    def get_rewards(self):    
        # self.rewards = -1*torch.ones(self.num_envs,)    
        self.rewards = (torch.sum((self.states[:,0:2] - self.states[:,2:])**2,dim=1) == 0)*1.0
                
    def get_reset_idx(self):
        goal_reset = torch.sum((self.states[:,0:2] - self.states[:,2:])**2,dim=1) == 0
        timer_reset = self.episode_time > self.max_episode_length
        self.reset_idx = goal_reset | timer_reset

    def reset_env(self):
        num_resets = torch.sum(self.reset_idx)
        self.states[self.reset_idx,:] = torch.randint(0,self.siz,[num_resets,4])*1.0  
        self.episode_time[self.reset_idx] = torch.zeros([num_resets,])
    
    def render(self, env_id, action_map=None):
        if(self.render_mode == "pygame"):
            self.render_world(env_id)
            if(action_map.any() != None):
                self.render_actions(action_map)
                
            pygame.event.pump()
            pygame.display.update()
    
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
    
    def get_SARS(self):
        with torch.no_grad():
            S1 = self.buffer.s1*2./(self.siz-1.)-1.
            A1 = self.buffer.a
            R1 = self.buffer.r
            S2 = self.buffer.s2*2./(self.siz-1.)-1.
        
        return S1, A1, R1, S2


    def render_world(self, env_id):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.board_size
        )  # The size of a single grid square in pixels


        # # Draw Apple
        # _al_xt, _al_yt = torch.where(self.apple_layer[self.view_env, ...] == 1)
        # for i in range(len(_al_xt)):
        #     _al_x = _al_xt[i].cpu().numpy()
        #     _al_y = _al_yt[i].cpu().numpy()
        #     _apple_location = np.array([_al_x, _al_y])
        #     pygame.draw.circle(
        #         canvas,
        #         (0, 255, 0),
        #         (_apple_location + 0.5) * pix_square_size,
        #         pix_square_size / 3,
        #     )
        # # Draw Fire
        # _fl_xt, _fl_yt = torch.where(self.fire_layer[self.view_env, ...] == 1)
        # for i in range(len(_fl_xt)):
        #     _fl_x = _fl_xt[i].cpu().numpy()
        #     _fl_y = _fl_yt[i].cpu().numpy()
        #     _fire_location = np.array([_fl_x, _fl_y])
        #     pygame.draw.rect(
        #         canvas,
        #         (255, 0, 0),
        #         pygame.Rect(
        #             pix_square_size * _fire_location,
        #             (pix_square_size, pix_square_size),
        #         ),
        #     )
        # Draw Goal

        _goal_location = self.states[env_id, 2:4].detach().cpu().numpy()
        
        print(pix_square_size,_goal_location,pix_square_size,pix_square_size)
        
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * _goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw Player
        _player_location = self.states[env_id, 0:2].detach().cpu().numpy()
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (_player_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )


        # Add some gridlines
        for x in range(self.board_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())

        
    def render_actions(self, action_map):
        pix_square_size = (
            self.window_size / self.board_size
        )  # The size of a single grid square in pixels
        offset = pix_square_size/6


        # The following line copies our drawings from `canvas` to the visible window

        for x in range(self.board_size):
            for y in range(self.board_size):
                action_img = self.action_ascii[action_map[x,y]]
                txtsurf = self.font.render(action_img, True, (0,0,0))
                # self.window.blit(txtsurf,(self.window_size - txtsurf.get_width() // 2, self.window_size - txtsurf.get_height() // 2))
                self.window.blit(txtsurf,(pix_square_size*x+offset, pix_square_size*y+offset))



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


        
        
   

#%%
# import time
# num_envs = 1000
# siz = 11

# env = simplEnv(num_envs,siz, 1000)

# for i in range(100):
#     actions = torch.randint(0,env.num_actions,[env.num_envs,])
#     action_map = torch.randint(0,9, (siz, siz)).detach().cpu().numpy()
#     env.render(0, action_map)
#     env.update(actions)
#     env.get_rewards()
#     env.get_reset_idx()
#     time.sleep(1)

# env.close()