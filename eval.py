from time import sleep
from Policy import GamePolicy_eval
from Environment import TouhouEnvironment
import time

policy = GamePolicy_eval("easy_boss-600-700.model", False)

env = TouhouEnvironment()

time.sleep(2)
reward, img, is_dead = env.reset()
policy.reset()

while(env.done is not True):
    action = policy.sample_action(img)
    if(action is not None):
        reward, img, is_dead = env.step(action)
    else:
        reward, img, is_dead = env.step(4, True)



