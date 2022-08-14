import torch
import torch.nn as nn
import config
import cv2
import numpy as np
import random
import torch.optim
import os
from torch.utils.data import Dataset, DataLoader
import time

class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (BCHW)
        self.cnn = nn.Sequential(
            nn.Conv2d(int(config.img_stack_num), 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * int(config.game_scene_size[1]/4) * int(config.game_scene_size[0]/4), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5) # 上 下 左 右 不动
        )

    def forward(self, x):
        '''
            outpu_shape: (batchsize, 32, H, W)
            in my case, batchisize should be one
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class GamePolicy():
    def __init__(self, dqnnet=None, init_epoch = 0) -> None:
        self.model_save_path = "dqnmodel_0_10.model"
        self.model_load_path = "dqnmodel_0.model"
        self.idx = 0
        '''
        idx: 指向img_stack下一个写地址
        '''
        self.img_stack = []
        if(dqnnet is not None):
            self.dqnnet = dqnnet
        else:
            self.__load_model()


        self.opt = torch.optim.Adam(self.dqnnet.parameters(), lr=config.lr, weight_decay=0.01)
        
        self.epoch = init_epoch
        self.explore_rate = max(config.epsilon_decay ** self.epoch, config.min_exploration)
        
        self.record_limit = 512
        self.record_head = 0
        '''
        record_head: 指向下一个写地址
        '''
        self.records = [] # (statei, ri, ai, statei+1, is_done)

        self.img_r = []
        self.reward_r = []
        self.action_r = []
        self.is_done_r = []

        

    def reset(self):
        self.dqnnet.eval()
        self.record_head = 0
        self.records = []
        self.explore_rate = config.epsilon_decay ** self.epoch
        self.img_stack = []

        self.img_r = []
        self.reward_r = []
        self.action_r = []
        self.is_done_r = []

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
            self.idx = (self.idx + 1) % config.img_stack_num
            return None
        else:
            # 更新img_stack
            self.img_stack[self.idx] = img
            self.idx = (self.idx + 1) % config.img_stack_num

        rand = random.random()
        j = self.idx
        a = []
        a.append(self.img_stack[j])
        j = (j + 1) % len(self.img_stack)
        while(j != self.idx):
            a.append(self.img_stack[j])
            j = (j + 1) % len(self.img_stack)
        state = torch.tensor(a, dtype=torch.float).transpose(0, 1).to(config.device)
        state = (state - state.mean(1)) / (state.std(1) + 1e-10)
        # print(state)

        if(rand < self.explore_rate):
        # if(False):
            # 随机探索， 不需要经过nn运算
            print("random choice")
            print()
            return random.randint(0, 4), state

        else:
            with torch.no_grad():
                self.dqnnet.eval()
                net_result:torch.Tensor = self.dqnnet(state.to(torch.float))
                action = net_result.flatten().argmax().item()
                print(net_result)
                print(action)
            
            
            
        return action, state

    def train(self):
        self.make_record()
        
        if(len(self.records) == 0):
            return None
        # 1. 从records中取数据构建batch
        
        dataset = RecordDataset(self.records)
        dataloader = DataLoader(
            dataset, config.batch_size,
            shuffle=True
        )

        # 2. 分别计算Q(S_i, a_i)和max_a(Q(S_i+1, a))
        avg_loss = torch.tensor(0.)
        
        update_num = 0
        for data, reward in dataloader:
            state0 = data[0]
            action = data[1]
            state1 = data[2]
            is_done = data[3]
            with torch.no_grad():
                v1 = self.dqnnet(state1.to(torch.float32)).detach()
            self.dqnnet.train()
            v0 = self.dqnnet(state0.to(torch.float32))
            
            loss = None
            Q1_argmax = v1.argmax(1)
            # 3. 计算损失

            for i in range(len(action)):
                if(is_done[i]):
                    if(loss is None):
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss = 0.5 * (v0[i][action[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss = abs(v0[i][action[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                    else:
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss += (v0[i][action[i]] - reward[i]) ** 2
                        else:
                            loss += abs(v0[i][action[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                        
                else:
                    if(loss is None):
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss = 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss = abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
                    else:
                        if(abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) < config.smooth_l1_beta):
                            loss += 0.5 * (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) ** 2 / config.smooth_l1_beta
                        else:
                            loss += (v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
            loss /= len(action)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            avg_loss = avg_loss.to(device)
            avg_loss += loss.detach()
            update_num += 1


        self.epoch += 1
        return avg_loss / update_num

    def save_record(self,state0, action, reward, next_img, is_done):
        '''
            action: sampled action
            reward: get reward from environment after excuting avtion
            next_img: get next_img from environment after excuting action
        '''
        if(len(self.img_stack) < config.img_stack_num):
            return
        if(is_done == False):
            j = self.idx
            a = []
            a.append(next_img/255)
            j = (j + 1) % len(self.img_stack)
            while(j != self.idx):
                a.append(self.img_stack[j])
                j = (j + 1) % len(self.img_stack)
            state1 = torch.tensor(a, dtype=torch.float).squeeze(1)
        else:
            state1 = torch.zeros((4, config.game_scene_size[1], config.game_scene_size[0]), dtype=torch.float)
        if(len(self.records) < self.record_limit):
            self.records.append((state0, action, reward, state1, is_done))
        else:
            self.records[self.record_head] = (state0, action, reward, state1, is_done)
        self.record_head += 1
        self.record_head %= self.record_limit
    
    def save_record_simple(self, img, reward=None, action=None, is_done=None):
        self.img_r.append(img)
        if(reward is not None):
            self.reward_r.append(reward)
        if(action is not None):
            self.action_r.append(action)
        if(is_done is not None):
            self.is_done_r.append(is_done)

    def make_record(self):
        if(len(self.img_r) < 4):
            return
        print("img:", len(self.img_r))
        print("act:", len(self.action_r))
        print("red:", len(self.reward_r))
        print("end:", len(self.is_done_r))
        state0 = None
        state1 = None
        for i in range(len(self.action_r)):
            a = []
            for j in range(i, i + 4):
                a.append(self.img_r[j])
            state0 = torch.tensor(a, dtype=torch.float).squeeze(1)
            state0 = (state0 - state0.mean(0)) / (state0.std(0) + 1e-10)
            if(self.is_done_r[i]):
                state1 = torch.zeros((4, config.game_scene_size[1], config.game_scene_size[0])).to(config.device)
            else:
                a = []
                for j in range(i+1, i + 5):
                    a.append(self.img_r[j])
                state1 = torch.tensor(a, dtype=torch.float).squeeze(1).to(config.device)
                state1 = (state1 - state1.mean(0)) / (state1.std(0) + 1e-10)
            self.records.append((state0, self.action_r[i], self.reward_r[i], state1, self.is_done_r[i]))
            if(self.is_done_r[i]):
                i += 3
                
       


    def save_model(self):
        torch.save(self.dqnnet, self.model_save_path)

    def __load_model(self):
        if(os.path.exists(self.model_load_path)):
            with open(self.model_load_path, "rb") as f:
                self.dqnnet = torch.load(f, map_location=torch.device(config.device))
        else:
            self.dqnnet = DQNNet().to(config.device)

class RecordDataset(Dataset):
    def __init__(self, record:list) -> None:
        '''
        record: [(state0, action, reward, state1, is_done), (...)]
        '''
        super().__init__()
        self.record = record

    def __getitem__(self, index):
        data = (self.record[index][0], self.record[index][1], self.record[index][3], self.record[index][4])
        reward = self.record[index][2]
        return data, reward

    def __len__(self):
        return len(self.record)
    




