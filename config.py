import time
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

save_replay_per_epoch = 50

random_seed = int(time.time())

# hyper param
replay_buffer_limit = 10000

epsilon_decay = 0.995 # exploration: epsilon_decay ** epoch

epsilon_decay_linear_epochs = 400 # epoch0 -> exploration = 1; epoch400 -> exploration = 0.01
# exploration: 1 - (1 - min_exploration) / epsilon_decay_linear_epochs * epoch

min_exploration = 0.01 

lr = 0.00001

batch_num = 1

batch_size = 8

gamma = 0.85

smooth_l1_beta = 1.0

img_stack_num = 4

update_frequency = 4

std_init = 0.5 # default value in the paper

# reward setting
alive_reward = 0.2

dead_penalty = -8

