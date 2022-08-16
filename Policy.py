import copy
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
import pickle as pkl

class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (BCHW)
        self.cnn = nn.Sequential(
            nn.Conv2d(int(config.img_stack_num), 32, 8, 4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


        )

        self.fc = nn.Sequential(
            # nn.Linear(256 * int(config.game_scene_resize_to[1]/8) * int(config.game_scene_resize_to[0]/8), 128),
            nn.Linear(15232, 512),
            nn.ReLU(),
            nn.Linear(512, 5),# 上 下 左 右 不动
        )

    def forward(self, x):
        '''
            outpu_shape: (batchsize, 32, H, W)
            in my case, batchisize should be one
        '''
        x = x.to(config.device)

        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class GamePolicy():
    def __init__(self, dqnnet=None, init_epoch = 0) -> None:
        self.model_save_path = "dqnmodel_" + str(init_epoch) + "_" + str(init_epoch + 1) + ".model"
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

        self.target_dqnnet = copy.deepcopy(self.dqnnet)

        self.opt = torch.optim.Adam(self.dqnnet.parameters(), lr=config.lr, weight_decay=0.01)
        self.init_epoch = init_epoch
        self.epoch = init_epoch
        self.explore_rate = max(config.epsilon_decay ** self.epoch, config.min_exploration)
        
        self.record_limit = 512
        self.records_file = "records.pkl"
        if(os.path.exists(self.records_file)):
            with open(self.records_file, "rb") as f:
                self.records, self.record_head = pkl.load(f)
        else:
            self.record_head = 0
            '''
            record_head: 指向下一个写地址
            '''
            self.records = [] # (statei, ri, ai, statei+1, is_dead)

        self.img_r = []
        self.reward_r = []
        self.action_r = []
        self.is_dead_r = []

        

    def reset(self):
        self.dqnnet.eval()
        # self.record_head = 0
        # self.records = []
        self.model_save_path = "dqnmodel_" + str(self.init_epoch) + "_" + str(self.epoch + 1) + ".model"
        self.explore_rate = config.epsilon_decay ** self.epoch
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
            self.idx = (self.idx + 1) % config.img_stack_num
            return None
        elif(not config.use_policy_v2):
            # 更新img_stack
            self.img_stack[self.idx] = img
            self.idx = (self.idx + 1) % config.img_stack_num
        rand = random.random()
        if(config.use_policy_v2):
            # for i in range(len(self.img_stack)):
            #     self.img_stack[i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
            #     cv2.imshow("1", self.img_stack[i])
            #     cv2.waitKey()
            state = torch.tensor(self.img_stack, dtype=torch.float).transpose(0, 1).to(config.device)
        else:
            j = self.idx
            a = []
            a.append(self.img_stack[j])
            j = (j + 1) % len(self.img_stack)
            while(j != self.idx):
                a.append(self.img_stack[j])
                j = (j + 1) % len(self.img_stack)
            state = torch.tensor(a, dtype=torch.float).transpose(0, 1).to(config.device)
        # state = (state - state.mean(1)) / (state.std(1) + 1e-10)
        # print(state)

        if(rand < self.explore_rate):
        # if(False):
            # 随机探索， 不需要经过nn运算
            # print("random choice")
            action = random.randint(0, 4)
            # print("action:", action)
        else:
            with torch.no_grad():
                self.dqnnet.eval()
                net_result:torch.Tensor = self.dqnnet(state.to(torch.float))
                action = net_result.flatten().argmax().item()
                print(net_result)
                print(action)

        if(config.use_policy_v2):        
            record_img = []
            for i in self.img_stack:
                record_img.append(i)
            self.img_r.append(record_img)
            self.img_stack = []  # 等待下一次img_stack
            self.idx = 0
        
            
        return action, state

    def train(self):
        if(config.use_policy_v2):
            self.make_record_v2()
        else:
            self.make_record()
        
        if(len(self.records) == 0):
            time.sleep(1)
            return None
        # 1. 从records中取数据构建batch
        if(len(self.records) > config.batch_num * config.batch_size):
            mini_records = random.sample(self.records, k=config.batch_num * config.batch_size)
        else:
            mini_records = self.records

        dataset = RecordDataset(mini_records)
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
            is_dead = data[3]
            with torch.no_grad():
                v1 = self.target_dqnnet(state1.to(torch.float32)).detach()
            self.dqnnet.train()
            v0 = self.dqnnet(state0.to(torch.float32))
            
            loss = None
            Q1_argmax = v1.argmax(1)
            # 3. 计算损失

            for i in range(len(action)):
                if(is_dead[i]):
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
                            loss += abs(v0[i][action[i]] - config.gamma * v1[i][Q1_argmax[i]] - reward[i]) - 0.5 * config.smooth_l1_beta
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
        self.epoch += 1
        if(self.epoch % config.update_frequency == 0):
            self.update_target_dqn()
        return avg_loss.item() / update_num
    
    def update_target_dqn(self):
        self.target_dqnnet = copy.deepcopy(self.dqnnet)

    def save_record(self,state0, action, reward, next_img, is_dead):
        '''
            action: sampled action
            reward: get reward from environment after excuting avtion
            next_img: get next_img from environment after excuting action
        '''
        if(len(self.img_stack) < config.img_stack_num):
            return
        if(is_dead == False):
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
            self.records.append((state0, action, reward, state1, is_dead))
        else:
            self.records[self.record_head] = (state0, action, reward, state1, is_dead)
        self.record_head += 1
        self.record_head %= self.record_limit
    
    def save_record_simple(self, img=None, reward=None, action=None, is_dead=None):
        if(img is not None):
            self.img_r.append(img)
        if(reward is not None):
            self.reward_r.append(reward)
        if(action is not None):
            self.action_r.append(action)
        if(is_dead is not None):
            self.is_dead_r.append(is_dead)

    def make_record(self):
        if(len(self.img_r) < 4):
            return
        # print("img:", len(self.img_r))
        # print("act:", len(self.action_r))
        # print("red:", len(self.reward_r))
        # print("end:", len(self.is_dead_r))
        state0 = None
        state1 = None
        for i in range(len(self.action_r)):
            a = []
            for j in range(i, i + 4):
                a.append(self.img_r[j])
            state0 = torch.tensor(a, dtype=torch.float).squeeze(1)
            state0 = (state0 - state0.mean(0)) / (state0.std(0) + 1e-10)
            if(self.is_dead_r[i]):
                state1 = torch.zeros((4, config.game_scene_size[1], config.game_scene_size[0])).to(config.device)
            else:
                a = []
                for j in range(i+1, i + 5):
                    a.append(self.img_r[j])
                state1 = torch.tensor(a, dtype=torch.float).squeeze(1).to(config.device)
                state1 = (state1 - state1.mean(0)) / (state1.std(0) + 1e-10)
            self.records.append((state0, self.action_r[i], self.reward_r[i], state1, self.is_dead_r[i]))
            if(self.is_dead_r[i]):
                i += 3
        
        

    def make_record_v2(self):
        if(len(self.img_r) < 1):
            return
        print("img:", len(self.img_r))
        print("last_reward:", self.reward_r[len(self.reward_r) - 1])
        # print("act:", len(self.action_r))
        # print("red:", len(self.reward_r))
        # print("end:", len(self.is_dead_r))
        state0 = None
        state1 = None
        for i in range(len(self.action_r)):
            a = []
            for j in range(4):
                a.append(self.img_r[i][j])
            state0 = torch.tensor(a, dtype=torch.float).squeeze(1)
            # state0 = (state0 - state0.mean(0)) / (state0.std(0) + 1e-10)
            if(self.is_dead_r[i]):
                state1 = torch.zeros((4, config.game_scene_resize_to[1], config.game_scene_resize_to[0])).to(config.device)
            else:
                a = []
                # print(i+1)
                # print("done:", self.is_dead_r[i])
                # print("stack len:",len(self.img_r[i+1]))
                
                for j in range(4):
                    a.append(self.img_r[i + 1][j])
                state1 = torch.tensor(a, dtype=torch.float).squeeze(1).to(config.device)
                # state1 = (state1 - state1.mean(0)) / (state1.std(0) + 1e-10)
            
            if(len(self.records) < self.record_limit):
                self.record_head = (self.record_head + 1) % self.record_limit
                self.records.append((state0, self.action_r[i], self.reward_r[i], state1, self.is_dead_r[i]))
            else:
                self.records[self.record_head] = (state0, self.action_r[i], self.reward_r[i], state1, self.is_dead_r[i])
                self.record_head = (self.record_head + 1) % self.record_limit
        print("records_buffer:", len(self.records))
        # with open("train_data/img.txt", "a") as f:
        #     f.write(str(self.epoch+1) + ":" + str(len(self.img_r)) + "\n")
        # with open(self.records_file, "wb") as f:
        #     pkl.dump((self.records, self.record_head), f)

    def save_model(self):
        torch.save(self.dqnnet, self.model_save_path)

    def __load_model(self):
        if(os.path.exists(self.model_load_path)):
            print("load exist")
            with open(self.model_load_path, "rb") as f:
                self.dqnnet = torch.load(f, map_location=torch.device(config.device))
        else:
            print("create new")
            self.dqnnet = DQNNet().to(config.device)

class RecordDataset(Dataset):
    def __init__(self, record:list) -> None:
        '''
        record: [(state0, action, reward, state1, is_dead), (...)]
        '''
        super().__init__()
        self.record = record

    def __getitem__(self, index):
        data = (self.record[index][0], self.record[index][1], self.record[index][3], self.record[index][4])
        reward = self.record[index][2]
        return data, reward

    def __len__(self):
        return len(self.record)
    




