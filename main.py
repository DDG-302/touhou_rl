from Environment import TouhouEnvironment
from Policy import GamePolicy
from tqdm import tqdm
import time
import config

train_epoch = 4
start_epoch = 21

env = TouhouEnvironment()
policy = GamePolicy(init_epoch=start_epoch)


pbar = tqdm(range(train_epoch))
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
            if not config.use_policy_v2:
                policy.save_record_simple(img)
            time.sleep(0.01)
            # print("get none?")
        else:
            # 非None则根据返回值移动，并保存游戏记录
            action, state = rtn
            reward, img = env.step(action)
            if config.use_policy_v2:
                pass
                policy.save_record_simple(None, reward, action, env.done)
            else:
                policy.save_record_simple(img, reward, action, env.done)
            # policy.save_record(state, action, reward, img, env.done)
        # print(i)
        # i += 1
        
        # end_time = time.perf_counter()
        # print("dleta time=", end_time-start_time)
        if(env.done):
            if(len(policy.is_done_r) != 0):
                policy.is_done_r[len(policy.is_done_r)-1] = True
            print("game over...")
            break
    loss = policy.train()
    if(loss is not None):
        pbar.set_description("avgloss= %f"%loss)
    else:
        pbar.set_description("avgloss= None")
    policy.save_model()

