import torch
# basic info

game_scene_size = (580, 670)
'''
(w, h)
'''

if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

# hyper param
epsilon_decay = 0.99 # exploration: 0.99 ** epoch

min_exploration = 0.01 

lr = 0.0001

batch_size = 16

gamma = 0.85

smooth_l1_beta = 1

img_stack_num = 4

# reward setting
alive_reward = 0.5

dead_penalty = -10

