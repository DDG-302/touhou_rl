import torch
import torch.nn as nn
import config
import cv2
import numpy as np
import random
import torch.optim
import os

class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (BCHW)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * int(config.game_scene_size[1]/2) * int(config.game_scene_size[0]/2), 32),
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
        x = self.cnn(x)
        x = x.flatten()
        x = self.fc(x)
        return x

class GamePolicy():
    def __init__(self, dqnnet=None, init_epoch = 0) -> None:
        self.idx = 0
        '''
        idx: 指向img_stack下一个写地址
        '''
        self.img_stack = []
        if(dqnnet is not None):
            self.dqnnet = dqnnet
        else:
            self.dqnnet = self.__load_model()

        self.opt = torch.optim.Adam(self.dqnnet.parameters(), lr=config.lr)
        
        self.epoch = init_epoch
        self.explore_rate = config.epsilon_decay ** self.epoch
        
        self.record_limit = 512
        self.record_head = 0
        '''
        record_head: 指向下一个写地址
        '''
        self.records = [] # (statei, ri, ai, statei+1)

        self.model_save_path = "dqnmodel.model"

    def reset(self):
        self.record_head = 0
        self.records = []
        self.explore_rate = config.epsilon_decay ** self.epoch

    def sample_action(self, img):
        '''
        img: coverted img, should be CHW (1, H, W)
        return: None or (action, state)
        - action: 0,1,2,3,4 -> up, down, left, right, idle; if None, means can't make decision
        - state: tensor -> [img0, img1, img2, img3]
        '''
        if(len(self.img_stack) < 4):
            # 直接插入到img_stack
            self.img_stack.append(img)
            self.idx = (self.idx + 1) % 4
            return None
        else:
            # 更新img_stack
            self.img_stack[self.idx] = img
            self.idx = (self.idx + 1) % 4

        rand = random.random()
        if(rand < self.explore_rate):
            # 随机探索， 不需要经过nn运算
            return random.randint(0, 4)

        else:
            j = self.idx
            a = []
            a.append(self.img_stack[j])
            j = (j + 1) % len(self.img_stack)
            while(j != self.idx):
                a.append(self.img_stack[j])
                j = (j + 1) % len(self.img_stack)
            state = torch.tensor(a)
            net_result:torch.Tensor = self.dqnnet(state)
            action = net_result.flatten().argmax().item()
            
            
        return action, state

    def train(self):
        # 1. 从records中取数据构建batch

        # 2. 分别计算Q(S_i, a_i)和max_a(Q(S_i+1, a))

        # 3. 计算损失
        loss = 0
        # 4. 更新
        pass

        self.epoch += 1
        return loss

    def save_record(self,state0, action, reward, next_img):
        '''
            action: sampled action
            reward: get reward from environment after excuting avtion
            next_img: get next_img from environment after excuting action
        '''
        j = self.idx
        a = []
        a.append(self.img_stack[j])
        j = (j + 1) % len(self.img_stack)
        while(j != self.idx):
            a.append(self.img_stack[j])
            j = (j + 1) % len(self.img_stack)
        state1 = torch.tensor(a)

        self.records[self.record_head] = (state0, action, reward, state1)
        self.record_head += 1
        if(self.record_head >= self.record_limit):
            self.record_head = 0

    def save_model(self):
        torch.save(self.dqnnet, self.model_save_path)

    def __load_model(self):
        if(os.path.exists(self.model_save_path)):
            with open(self.model_save_path, "rb") as f:
                self.dqnnet = torch.load(f)
        else:
            self.dqnnet = DQNNet()
            




