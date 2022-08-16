from Environment import TouhouEnvironment
from Policy import GamePolicy
from tqdm import tqdm
import time
import config
import torch

train_epoch = 0
start_epoch = 0
env = TouhouEnvironment()
policy = GamePolicy(init_epoch=start_epoch, epsilon_temp=250)


pbar = tqdm(range(train_epoch))
time.sleep(2)

for _ in pbar:
    reward, img, is_dead = env.reset()
    policy.reset()
    i = 0
    
    while(True):
        if(len(policy.img_r) >= 500):
            break
        # start_time = time.perf_counter()
        
        rtn = policy.sample_action(img)
        if(rtn == None):
            # None则不动
            reward, img, is_dead = env.step(4)
            if not config.use_policy_v2:
                policy.save_record_simple(img)
                
            # time.sleep(0.01)
            # print("get none?")
        else:
            if(env.is_done()):
                if(len(policy.is_dead_r) != 0):
                    policy.is_dead_r[len(policy.is_dead_r)-1] = True
                    policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
                print("game over...")            
                break  
            # 非None则根据返回值移动，并保存游戏记录
            action, state = rtn
            reward, img, is_dead = env.step(action)
            if config.use_policy_v2:
                # if(len(policy.is_dead_r) > 0):
                #     policy.is_dead_r[len(policy.is_dead_r)-1] = is_dead
                #     policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
                policy.save_record_simple(None, reward, action, False)
            else:
                policy.save_record_simple(img, reward, action, is_dead)
            # policy.save_record(state, action, reward, img, env.done)
        # print(i)
        # i += 1
        
        # end_time = time.perf_counter()
        # print("dleta time=", end_time-start_time)
        if(env.done):
            if(len(policy.is_dead_r) != 0):
                policy.is_dead_r[len(policy.is_dead_r)-1] = True
                policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
            print("game over...")            
            break    

    loss = policy.train()

    if(loss is not None):
        pbar.set_description("avgloss= %f"%loss)
    else:
        pbar.set_description("avgloss= None")
    policy.save_model()
    if(len(policy.img_r) >= 500):
        print("maybe better_enough")
        break
        

