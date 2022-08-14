from turtle import done
from Environment import TouhouEnvironment
from Policy import GamePolicy
from tqdm import tqdm
import time
import torch

train_epoch = 10
start_epoch = 0

env = TouhouEnvironment()
policy = GamePolicy(init_epoch=start_epoch)


pbar = tqdm(range(start_epoch + train_epoch))
time.sleep(2)
for _ in pbar:
    reward, img = env.reset()
    policy.reset()
    i = 0
    
    while(True):
        # start_time = time.perf_counter()
        rtn = policy.sample_action(img)
        if(rtn == None):
            # None则不动
            reward, img = env.step(4)
            policy.save_record_simple(img)
            # print("get none?")
        else:
            # 非None则根据返回值移动，并保存游戏记录
            action, state = rtn
            reward, img = env.step(action)

            policy.save_record_simple(img, reward, action, env.done)
            # policy.save_record(state, action, reward, img, env.done)
        # print(i)
        # i += 1
        
        # end_time = time.perf_counter()
        # print("dleta time=", end_time-start_time)
        if(env.done):
            print("game over...")
            break
    loss = policy.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss=loss.to(device)

    if(loss is not None):
        pbar.set_description("avgloss= %f"%loss)
    else:
        pbar.set_description("avgloss= None")
    policy.save_model()

