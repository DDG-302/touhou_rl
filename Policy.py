import copy
from re import S
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import cv2
import numpy as np
import random
import torch.optim
import os
from torch.utils.data import Dataset, DataLoader
import time
import pickle as pkl
import math

random.seed(config.random_seed)

class DQNNet(nn.Module):
    def __init__(self, use_noisy_net):
        super(DQNNet, self).__init__()
        # input shape: (BCHW)

        self.use_nosiy_net = use_noisy_net
        self.cnn = nn.Sequential(
            nn.Conv2d(int(config.img_stack_num), 32, 8, 4), # (290, 335) -> (71, 82)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2), # (71, 82) -> (34, 40)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1), # (34, 40) -> (32, 38)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1), # (32, 38) -> (30, 36)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 30 * 36, 512),
            nn.ReLU(),
        )

        if(use_noisy_net):
            self.NoisyLayer_e1 = NoisyLayer(512, 256)
            self.NoisyLayer_e2 = NoisyLayer(256, 1)
            self.estimate = nn.Sequential(
                self.NoisyLayer_e1,
                nn.ReLU(),
                self.NoisyLayer_e2
            )

            self.NoisyLayer_a1 = NoisyLayer(512, 256)
            self.NoisyLayer_a2 = NoisyLayer(256, 5)
            self.advantages = nn.Sequential(
                self.NoisyLayer_a1,
                nn.ReLU(),
                self.NoisyLayer_a2 # 上 下 左 右 不动
            )
        else:
            self.estimate = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

            self.advantages = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 5),# 上 下 左 右 不动
            )

    def forward(self, x):
        '''
            x_shape: (batchsize, stack_num, H, W)
        '''
        x = x.to(config.device)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        e = self.estimate(x)
        a = self.advantages(x)
        x = a + e - a.mean(dim=-1, keepdim=True)
        return x

    def reset_noisy(self):
        self.NoisyLayer_a1.reset_noise()
        self.NoisyLayer_a2.reset_noise()
        self.NoisyLayer_e1.reset_noise()
        self.NoisyLayer_e2.reset_noise()

class NoisyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(self.out_features))
        
        # register_buffer中的参数不会作为模型参数
        self.register_buffer("weight_epsilon", torch.Tensor(self.out_features, self.in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # ref: https://pytorch.org/docs/1.7.0/generated/torch.nn.Linear.html?highlight=nn%20linear#torch.nn.Linear
        # 与nn.linear的权重初始化过程相同
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            config.std_init / math.sqrt(self.out_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            config.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(x, 
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )



class GamePolicy_train():
    def __init__(self, use_noisy_net:bool, dqnnet=None, init_epoch = 0, epsilon_offset = None) -> None:
        self.model_save_path = "0_500_model.model"
        self.model_load_path = "64_500_model.model"

        self.use_noisy_net = use_noisy_net

        '''
        idx: 指向img_stack下一个写地址
        '''
        self.img_stack = []
        self.debug_last_img_stack = []
        if(dqnnet is not None):
            self.dqnnet = dqnnet
        else:
            self.__load_model()

        self.target_dqnnet = copy.deepcopy(self.dqnnet)

        self.opt = torch.optim.Adam(self.dqnnet.parameters(), lr=config.lr, weight_decay=0.01)
        self.init_epoch = init_epoch
        
        if(epsilon_offset is not None):
            self.epsilon_epoch = epsilon_offset
        else:
            self.epsilon_epoch = init_epoch
        self.epoch = 0
        self.explore_rate = max(
                max(config.epsilon_decay ** self.epsilon_epoch, 1 - (1 - config.min_exploration) / config.epsilon_decay_linear_epochs * self.epsilon_epoch),
                config.min_exploration)
        self.epsilon_epoch += 1
        
        self.replay_buffer_file = "replay_buffer.pkl"

        if(os.path.exists(self.replay_buffer_file)):
            print("load replay buffer...")
            with open(self.replay_buffer_file, "rb") as f:
                self.replay_buffer, self.replay_head = pkl.load(f)
        else:
            self.replay_head = 0
            '''
            replay_head: 指向下一个写地址
            '''
            self.replay_buffer = [] # (statei, ri, ai, statei+1, is_dead)
            

        self.img_r = []
        self.reward_r = []
        self.action_r = []
        self.is_dead_r = []

        

    def reset(self):
        # self.dqnnet.eval()
        if(self.use_noisy_net):
            self.dqnnet.reset_noisy()
            self.target_dqnnet.reset_noisy()
        else:
            self.explore_rate = max(
                max(config.epsilon_decay ** self.epsilon_epoch, 1 - (1 - config.min_exploration) / config.epsilon_decay_linear_epochs * self.epsilon_epoch),
                config.min_exploration)
        self.img_stack = []

        self.img_r = []
        self.reward_r = []
        self.action_r = []
        self.is_dead_r = []

    def sample_action(self, img):
        '''
        img: coverted img, should be CHW (1, H, W)
        return: None or (action, state)
        - action: 0,1,2,3,4 -> up, down, left, right, idle; if None, means can't make decision
        - state: tensor -> [img0, img1, img2, img3]
        '''
        if(len(self.img_stack) < config.img_stack_num):
            # 直接插入到img_stack
            self.img_stack.append(img)
            return None

        rand = random.random()
        # 查看img_stack中的图片内容
        # for i in range(len(self.img_stack)):
        #     self.img_stack[i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
        #     cv2.imshow("1", self.img_stack[i])
        #     cv2.waitKey()
        state = torch.tensor(self.img_stack, dtype=torch.float).transpose(0, 1).to(config.device)
        rtn_state = copy.copy(self.img_stack)

        if(self.use_noisy_net):
            # noisy network不需要epsilon greedy
            with torch.no_grad():
                self.dqnnet.eval()
                net_result:torch.Tensor = self.dqnnet(state.to(torch.float))
                action = net_result.flatten().argmax().item()
                print(net_result)
                print(action)
        else:
            # print("no noisy")
            # if(len(self.replay_buffer) < config.replay_buffer_limit and rand < 0.5):
            #     action = random.randint(0, 4)
            if(rand < self.explore_rate):
            # if(False):
                # 随机探索， 不需要经过nn运算
                action = random.randint(0, 4)

            else:
                with torch.no_grad():
                    self.dqnnet.eval()
                    net_result:torch.Tensor = self.dqnnet(state.to(torch.float))
                    action = net_result.flatten().argmax().item()
                    print(net_result)
                    print(action)
     
        
        self.debug_last_img_stack = self.img_stack
        self.img_stack = []  # 等待下一次img_stack

        return action, rtn_state

    def train(self):
        '''
        call "make_record" first!
        '''
        self.dqnnet.train()
        self.target_dqnnet.eval()
        if(len(self.img_r) < 4):
            return None
        # if(len(self.replay_buffer) < 1024):
        #     time.sleep(1)
        #     return None
        # 1. 从records中取数据构建batch
        if(len(self.replay_buffer) > config.batch_num * config.batch_size):
            mini_records = random.sample(self.replay_buffer, k=config.batch_num * config.batch_size)
        else:
            mini_records = self.replay_buffer
        dataset = ReplayDataset(mini_records)
        dataloader = DataLoader(
            dataset, config.batch_size,
            shuffle=True
        )

        # 2. 分别计算Q(S_i, a_i)和max_a(Q(S_i+1, a))
        avg_loss = torch.tensor(0.)
        
        update_num = 0
        for state0, action, state1, is_dead, reward in dataloader:
            # state0 = data[0]
            # action = data[1]
            # state1 = data[2]
            # is_dead = data[3]
            with torch.no_grad():
                v1 = self.target_dqnnet(state1.to(torch.float32)).detach()
            self.dqnnet.train()
            v0 = self.dqnnet(state0.to(torch.float32))
            
            loss = None
            Q1_argmax = v1.argmax(1)
            # 3. 计算损失

            for i in range(len(action)):
                if(is_dead[i]):
                    # smooth l1
                    if(loss is None):
                        if(abs(v0[i][action[i]] - reward[i]) < config.smooth_l1_beta):
                            loss = 0.5 * (v0[i][action[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss = abs(v0[i][action[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                    else:
                        if(abs(v0[i][action[i]] - reward[i]) < config.smooth_l1_beta):
                            loss += 0.5 * (v0[i][action[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss += abs(v0[i][action[i]] - reward[i]) - 0.5 * config.smooth_l1_beta

                    # l2
                    # if(loss is None):
                    #     loss = 0.5 * (v0[i][action[i]] - reward[i]) ** 2
                    # else:
                    #     loss += 0.5 * (v0[i][action[i]] - reward[i]) ** 2 
                        
                else:
                    # smooth l1
                    if(loss is None):
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss = 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss = abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                    else:
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss += 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss += abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                    
                    # l2
                    # if(loss is None):
                    #     loss = 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2
                    # else:
                    #     loss += 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2
                       
            loss /= len(action)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            avg_loss = avg_loss.to(config.device)
            avg_loss += loss.detach()
            update_num += 1
        if(os.path.exists("train_data")): 
            with open("train_data/avg_loss.txt", "a") as f:
                f.write(str((avg_loss / update_num).item()) + "\n")
        # if(os.path.exists("train_data/img.txt")):
        #     with open("train_data/img.txt", "a") as f:
        #         f.write(str(self.epsilon_epoch+1) + ":" + str(len(self.img_r)) + "\n")
        if( self.epoch == 0 or
            (self.epoch != self.init_epoch and (self.epoch - 1) % config.save_replay_per_epoch == 0)
            ):
            with open(self.replay_buffer_file, "wb") as f:
                pkl.dump((self.replay_buffer, self.replay_head), f)
        if(self.epoch % config.update_frequency == 0):
            self.update_target_dqn()
        self.epsilon_epoch += 1
        self.epoch += 1
        return avg_loss.item() / update_num
    
    def update_target_dqn(self):
        self.target_dqnnet = copy.deepcopy(self.dqnnet)
    

    
    def save_replay_simple(self, img_stack=None, reward=None, action=None, is_dead=None):
        if(img_stack is not None):
            self.img_r.append(copy.copy(img_stack))
        if(reward is not None):
            self.reward_r.append(reward)
        if(action is not None):
            self.action_r.append(action)
        if(is_dead is not None):
            self.is_dead_r.append(is_dead)
        
        

    def make_record(self):
        if(len(self.img_r) < 1):
            return
        print("img:", len(self.img_r))
        nr_num = 0
        # print("act:", len(self.action_r))
        # print("red:", len(self.reward_r))
        # print("end:", len(self.is_dead_r))
        if(len(self.img_r) != len(self.action_r) or 
            len(self.img_r) != len(self.reward_r) or
            len(self.img_r) != len(self.is_dead_r)):
            print("img:", len(self.img_r))
            print("act:", len(self.action_r))
            print("red:", len(self.reward_r))
            print("end:", len(self.is_dead_r))
            raise Exception("数据数量不对，检查代码")
        state0 = None
        state1 = None
        for i in range(len(self.action_r)):
            a = []
            if(self.reward_r[i] > 0 and self.is_dead_r[i]):
                print(self.reward_r[i])
                print(self.is_dead_r[i])
                raise Exception("miss但奖励是正数")
            if(self.reward_r[i] < 0 and not self.is_dead_r[i]):
                raise Exception("存活但奖励是负数")
            if(self.reward_r[i] < 0):
                nr_num += 1
            for j in range(4):
                a.append(self.img_r[i][j])
            state0 = a
            # state0 = torch.tensor(a, dtype=torch.float).squeeze(1)
            # state0 = (state0 - state0.mean(0)) / (state0.std(0) + 1e-10)
            if(self.is_dead_r[i]):
                state1 = np.zeros((4, config.game_scene_resize_to[1], config.game_scene_resize_to[0]))
                # state1 = torch.zeros((4, config.game_scene_resize_to[1], config.game_scene_resize_to[0])).to(config.device)
            else:
                a = []
                
                for j in range(4):
                    a.append(self.img_r[i + 1][j])
                state1 = a
                # state1 = torch.tensor(a, dtype=torch.float).squeeze(1).to(config.device)
                # state1 = (state1 - state1.mean(0)) / (state1.std(0) + 1e-10)
            
            if(len(self.replay_buffer) < config.replay_buffer_limit):
                self.replay_buffer.append((state0, self.action_r[i], self.reward_r[i], state1, self.is_dead_r[i]))
                self.replay_head = len(self.replay_buffer) % config.replay_buffer_limit
            else:
                self.replay_buffer[self.replay_head] = (state0, self.action_r[i], self.reward_r[i], state1, self.is_dead_r[i])
                self.replay_head = (self.replay_head + 1) % config.replay_buffer_limit
        print("replay_buffer:", len(self.replay_buffer))
        
        # with open(self.replay_buffer_file, "wb") as f:
        #     pkl.dump((self.replay_buffer, self.replay_head), f)
        print("neg_reward_num:", nr_num)
        if(len(self.replay_buffer) == config.replay_buffer_limit):
            if(os.path.exists("train_data/img.txt")):
                with open("train_data/img.txt", "a") as f:
                    f.write(str(self.epsilon_epoch+1) + ":" + str(len(self.img_r)) + "\n")

    
    def save_model(self):
        torch.save(self.dqnnet, self.model_save_path)

    def save_checkpoint(self, model_name, replay_name):
        with open(replay_name, "wb") as f:
            pkl.dump(self.replay_buffer, f)
        torch.save(self.dqnnet, model_name)

    def __load_model(self):
        if(os.path.exists(self.model_load_path)):
            print("load exist")
            with open(self.model_load_path, "rb") as f:
                self.dqnnet = torch.load(f, map_location=torch.device(config.device))
        else:
            print("create new")
            self.dqnnet = DQNNet(use_noisy_net=self.use_noisy_net).to(config.device)

class GamePolicy_eval():
    def __init__(self, model_save_path, use_noisy_net) -> None:
        self.model_save_path = model_save_path
        self.use_noisy_net = use_noisy_net
        if(os.path.exists(self.model_save_path)):
            with open(self.model_save_path, "rb") as f:
                self.model = torch.load(f, map_location=config.device)
        else:
            raise Exception("model doesn't exist")
        self.img_stack = []

    def reset(self):
        self.img_stack = []
        if(self.use_noisy_net):
            self.model.reset_noisy()
        
    def sample_action(self, img):
        if(len(self.img_stack) < config.img_stack_num):
            self.img_stack.append(img)
            return None
        else:
            with torch.no_grad():
                state = torch.tensor(self.img_stack, dtype=torch.float).transpose(0, 1).to(config.device)
                action = self.model(state).flatten().argmax().item()
            self.img_stack = []
            return action
            
        

        

class ReplayDataset(Dataset):
    def __init__(self, record:list) -> None:
        '''
        record: [(state0, action, reward, state1, is_dead), (...)]
        '''
        super().__init__()
        self.record = record

    def __getitem__(self, index):
        data = (torch.tensor(self.record[index][0], dtype=torch.float).squeeze(1).to(config.device),
        self.record[index][1], 
        torch.tensor(self.record[index][3], dtype=torch.float).squeeze(1).to(config.device), 
        self.record[index][4],
        self.record[index][2])

        return data

    def __len__(self):
        return len(self.record)
    




