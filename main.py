from Environment import TouhouEnvironment
from Policy import GamePolicy
from tqdm import tqdm



env = TouhouEnvironment()
policy = GamePolicy()

epoch = 10
pbar = tqdm(range(epoch))

for _ in pbar:
    reward, img = env.reset()
    policy.reset()

    while(True):
        rtn = policy.sample_action(img)
        if(rtn == None):
            # None则不动
            reward, img = env.step(4)
            continue
        else:
            # 非None则根据返回值移动，并保存游戏记录
            action, state = rtn
            reward, img = env.step(action)
            policy.save_record(state, action, reward, img)
            
        
        if(env.done):
            break
    loss = policy.train()
    pbar.set_description("loss= %f"%loss)
    policy.save_model()

