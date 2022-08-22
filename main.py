from Environment import TouhouEnvironment
from Policy import GamePolicy_train, GamePolicy_eval
from tqdm import tqdm
import time
import config
import torch

train_epoch = 150
start_epoch = 0
env = TouhouEnvironment()
policy = GamePolicy_train(True, init_epoch=start_epoch)


pbar = tqdm(range(train_epoch))
time.sleep(2)
trained_epochs = 0
for _ in pbar:
    reward, img, is_dead = env.reset()
    policy.reset()
    i = 0
    
    while(True):
        if(len(policy.img_r) >= 500):
            print("too much img...")
            break
        # start_time = time.perf_counter()
        if(env.is_done()):
            import Move
            Move.ReleaseKey(0x2C)
            if(len(policy.action_r) > 0):
                policy.action_r.pop()
                policy.img_r.pop()
                policy.reward_r.pop()
                policy.is_dead_r.pop()

            if(len(policy.is_dead_r) != 0):
                policy.is_dead_r[len(policy.is_dead_r)-1] = True
                policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
            print("game over1...")  
            
            policy.make_record()   

            if(len(policy.replay_buffer) < config.replay_buffer_limit):
                time.sleep(1)
                reward, img, is_dead = env.reset()
                policy.reset()
                continue       
            break  
        rtn = policy.sample_action(img)
        if(rtn == None):
            # None则不动
            reward, img, is_dead = env.step(4, True)
                
            # time.sleep(0.01)
            # print("get none?")
        else:
            # 非None则根据返回值移动，并保存游戏记录
            action, state = rtn
            reward, img, is_dead = env.step(action)
            # if(len(policy.is_dead_r) > 0 and is_dead):
            #     policy.is_dead_r[len(policy.is_dead_r)-1] = is_dead
            #     policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
            #     policy.save_replay_simple(None, config.alive_reward, action, False)
            # else:
            #     policy.save_replay_simple(None, reward, action, False)
            if(not is_dead):
                policy.save_replay_simple(state, reward, action, is_dead)
            else:
                policy.reward_r[len(policy.reward_r) - 1] = reward
                policy.is_dead_r[len(policy.is_dead_r) - 1] = is_dead
                # import cv2
                # for i in range(len(policy.img_r[len(policy.img_r)-1])):
                #     policy.img_r[len(policy.img_r)-1][i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
                #     cv2.imshow("dead", policy.img_r[len(policy.img_r)-1][i])
                #     cv2.waitKey()
                # for i in range(4):
                #     state[i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
                #     cv2.imshow("1", state[i])
                #     cv2.waitKey()

        
        # end_time = time.perf_counter()
        # print("dleta time=", end_time-start_time)
        if(env.done):

            if(len(policy.is_dead_r) != 0):
                policy.is_dead_r[len(policy.is_dead_r)-1] = True
                policy.reward_r[len(policy.reward_r)-1] = config.dead_penalty
            print("game over2...")     

            policy.make_record()

            if(len(policy.replay_buffer) < config.replay_buffer_limit):
                time.sleep(1)
                reward, img, is_dead = env.reset()
                policy.reset()
                continue           
            break    
    # import cv2
    # for i in range(len(policy.img_r[len(policy.img_r)-1])):
    #                 policy.img_r[len(policy.img_r)-1][i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
    #                 cv2.imshow("dead", policy.img_r[len(policy.img_r)-1][i])
    #                 cv2.waitKey()     
    # for i in range(4):
    #                 state[i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
    #                 cv2.imshow("state dead", state[i])
    #                 cv2.waitKey()
    # for j in range(len(policy.img_r)-1, len(policy.img_r)):
    #     for i in range(4):
    #                 policy.img_r[j][i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
    #                 cv2.imshow("end" + str(j) + "_" + str(i), policy.img_r[j][i])
    #                 cv2.waitKey()
    # for j in range(len(policy.img_r)):
    #     for i in range(4):
    #                 policy.img_r[j][i].shape = (config.game_scene_resize_to[1], config.game_scene_resize_to[0], 1)
    #                 cv2.imshow("all r" + str(j) + "_" + str(i), policy.img_r[j][i])
    #                 cv2.waitKey()
    if(len(policy.img_r) >= 500):
        print("too much img")
        break

    loss = policy.train()

    if(loss is not None):
        pbar.set_description("avgloss= %f,trained_epochs=%d"%(loss,trained_epochs))
        trained_epochs += 1
    else:
        pbar.set_description("avgloss= None,trained_epochs=%d"%trained_epochs)
    policy.save_model()

    if(trained_epochs >= train_epoch):
        break
    
        

