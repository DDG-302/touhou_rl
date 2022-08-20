import torch
# basic info

game_scene_size = (580, 670)
'''
(w, h)
'''
game_scene_resize_to = (290, 335)
'''
(w, h)
'''

if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

use_policy_v2 = True

# hyper param
epsilon_decay = 0.99 # exploration: 0.99 ** epoch

min_exploration = 0.01 

lr = 0.0001

batch_num = 1

batch_size = 8

gamma = 0.75

smooth_l1_beta = 1.0

img_stack_num = 4

update_frequency = 2

# reward setting
alive_reward = 0.2

dead_penalty = -8

