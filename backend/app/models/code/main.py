# from sklearn.preprocessing import StandardScaler
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from json import JSONEncoder
# import json
import time
import random
# from operator import itemgetter
import numpy as np
import gym
import matlab.engine
# from IPython import display
import os
# import argparse
# import cv2
# from IPython.display import clear_output

# def comp(in_state, in_gamma):
#     eng = matlab.engine.start_matlab()
#     eng.cd('/home/morris/RL-demo-web/backend/app/models/code')
#     current_dir = eng.pwd()
#     print(current_dir)

def comp(in_state, in_gamma):
    os.chdir("/home/morris/RL-demo-web/backend/app/models/code")
    env = gym.make('gym_mapHR_v1:mapHR-v1')
    eng = matlab.engine.start_matlab()
    eng.cd('/home/morris/RL-demo-web/backend/app/models/code')
    eng.addpath('/home/morris/RL-demo-web/backend/app/models/code', nargout=0)
    # system size
    size = 8
    # randomly generated combinations
    # state_0 = np.random.randint(2, size=size*int(size/2)) 
    # 如果 in_state 是 list，則轉換成 numpy array
    if isinstance(in_state, list):
        state_0 = np.array(in_state)
    else:
        state_0 = in_state    # state_0 = [0,1,2]
    # set the system size and current state
    env.ENVsize(size,state_0.reshape(size*int(size/2))) 

    # set parameters
    # training times
    episodes_list = [100, 400, 1000, 3500] 
    # the number of actions is performed in an episode
    iterations_list = [8, 12, 50, 90]
    # learning rate
    alpha = 0.4         
    # the importance of future rewards
    # gamma = 0.99 
    gamma = in_gamma
    # randomness of action
    epsilon = 1 
    # minimum value of epsilon
    epsilon_min = 0.001 
    #　When does the episode reach the minimum value
    final_dec_ep_list = [20, 50, 50, 500] 
    # number of times to enlarge the system size
    increase_times = 3    

    def train():
        read_data = 1
        read_q_table = 1
        # whether to occasionally use other combinations as the initial state
        inb = 1 
        # initialization
        # starting time
        start_time = time.time()     
        initmap = np.array(env.reset().reshape(1,int(size*size/2)))

        img_buffer = None  # 初始化 img_buffer

        for ss in range(increase_times+1):
            # initialization
            episodes = episodes_list[ss]
            iterations = iterations_list[ss]
            alpha = 0.4          
            gamma = 0.99         
            epsilon = 1          
            epsilon_min = 0.001
            # decay rate of epsilon 
            epsilon_decay = epsilon_min**(1/(episodes-final_dec_ep_list[ss]))
            max_lengh = 5
            top_map = [0]
            top_map_ep = [0]
            top_map_it = [0]
            top_map_time = [0]
            length = 0
            best_num = 0
            env.ENVsize(size,initmap)
            #　current state
            state = initmap  
            # id of the current state
            state_id = 0    
            training_rewards = [] 
            
            if read_data == 1:
                state_id_table = pd.read_csv(str(size)+'_state_table.txt',delim_whitespace=True, header=None).values
                toughness_table = pd.read_csv(str(size)+'_t_table.txt',delim_whitespace=True, header=None).values
                toughness_table = list(toughness_table.reshape(toughness_table.shape[0]))
            else:
                # build state table
                state_id_table = initmap 
                # build toughness table
                toughness_table = [float(env.toughness(state_id_table))]    
            if read_q_table == 1:
                q_table = pd.read_csv(str(size)+'_q_table.txt',delim_whitespace=True, header=None).values
            else:
                # build q_table
                q_table = 0.00001*np.round(np.random.randn(1, size*int(size/2)), decimals = 2)    
            
            #　the property of the current state
            toughness_now = float(env.toughness(state_id_table[0]))    
            top_tou = [toughness_now]   
            # the property of the initial state
            toughness_0 = toughness_now     
            # env.plotmap_save(state_id_table[0],(np.round(time.time() - start_time, decimals = 2)),0,0) 
            initmap_pd = pd.DataFrame(data=state_id_table)    
            initmap_pd.to_csv(str(size)+'initmap.txt', header=None, index=None, sep=' ', mode='w')
            
            for ep in range(episodes):

                rewards = 0

                for it in range(iterations):
                    
                    Higher_tou = 0
                    #　generate random numbers to determine the randomness of actions
                    exp_tradeoff = random.uniform(0, 1) 
                    
                    if exp_tradeoff > epsilon:
                        # action determined by q table
                        action = np.argmax(q_table[state_id,:])       #由q_table決定動作
                        
                    else:
                        # random action
                        action = random.sample(list(np.arange(0,int(size*size/2))) ,1)      
                
                    # new state
                    last_state = env.step(action)
                    last_state_re = np.array([last_state[0:int(size*size/2)]])
                    
                    # check for the same state
                    for i in range(state_id_table.shape[0]):
                        
                        same_map = np.array_equal(state_id_table[i], np.array(last_state_re[0])) 
                        if same_map == True:     
                            last_state_id = i
                            toughness_2 = toughness_table[i]   
                            break
                        elif i == state_id_table.shape[0]-1:  
                            state_id_table = np.concatenate((state_id_table, last_state_re), axis=0)
                            toughness_2 = env.toughness(last_state_re)
                            toughness_table.append(toughness_2) 
                            last_state_id= state_id_table.shape[0]-1
                            q_table = np.concatenate((q_table, 0.00001*np.round(np.random.randn(1, size*int(size/2)), 
                                                                                decimals = 2)), axis=0)
                    # get reward                        
                    reward, Higher_tou= env.get_rewaed(toughness_now, toughness_2,
                                                            toughness_table[int(top_map[len(top_map)-1])]) 
                
                    # save better state
                    if Higher_tou == 1: 
                        if length >=  max_lengh:
                            top_map.pop(0)  
                            top_map_ep.pop(0) 
                            top_map_it.pop(0) 
                            top_map_time.pop(0)
                            top_tou.pop(0)
                            length -= 1
                            env.plotmap_save(last_state_re,(np.round(time.time() - start_time, decimals = 2)),ep,it)
                        top_map.append(int(last_state_id))
                        top_map_ep.append(ep+1)
                        top_map_it.append(it+1)
                        top_map_time.append((np.round(time.time() - start_time, decimals = 2)))
                        top_tou.append(toughness_2)
                        length += 1    
                        best_num = last_state_id
                    
                    # update q table
                    q_table[state_id, action] += alpha * (reward + gamma * 
                                                        (np.max(q_table[last_state_id, :])) - q_table[state_id, action]) 
                    # reward accumulation
                    rewards += reward     
                    # update state
                    state = last_state
                    # update state id
                    state_id = last_state_id     
                    # update the property of the current state
                    toughness_now = toughness_2

            
                    # show current status
                    # if int((it)%1) == 0:
                        # env.showmap(ep,it,(np.round(time.time() - start_time, decimals = 2))) 
                        #clear_output()
                        #print("Toughness: {} episode: {} iteration: {} time: {}" .format(np.round(toughness_2, decimals = 2),ep, it, (np.round(time.time() - start_time, decimals = 2))))
                
                # epsilon greedy
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay  
            
                training_rewards.append(rewards)
                
                if ep >= episodes-final_dec_ep_list[ss]:
                    inb = 0
                    
                # use a better state as the initial state every 5 episodes    
                if inb == 1:
                    if int((ep%5))== 0: 
                        env.ENVsize(size,state_id_table[int(top_map[len(top_map)-1])])
                        state = env.reset()
                        state_id = int(top_map[len(top_map)-1])
                        toughness_now = top_tou[len(top_map)-1]
                    else:
                        env.ENVsize(size,np.array(initmap,int))
                        state = env.reset()
                        state_id = 0
                        toughness_now = toughness_0
                else:
                    env.ENVsize(size,np.array(initmap,int))
                    state = env.reset()
                    state_id = 0
                    toughness_now = toughness_0
                    
                # storage table and reward accumulation every 100 episodes        
                if int((ep%100))== 0:   
                    q_pd = pd.DataFrame(data=q_table)
                    q_pd.to_csv(str(size)+'_q_table.txt', header=None, index=None, sep=' ', mode='w')
                    state_id_table_df= pd.DataFrame(data=state_id_table)
                    state_id_table_df.to_csv(str(size)+'_state_table.txt', header=None, index=None, sep=' ', mode='w')
                    tt_pd = pd.DataFrame(data=toughness_table)
                    tt_pd.to_csv(str(size)+'_t_table.txt', header=None, index=None, sep=' ', mode='w')
                    ep_hl = np.arange(ep+1)+1       
                    # 不再调用 plt.show()，而是关闭图像窗口
                    # plt.figure()
                    # plt.plot(ep_hl, training_rewards)
                    # plt.xlabel('episode')
                    # plt.ylabel('reward')
                    # plt.title('episode_rewards' + str(ss + 1))
                    # plt.legend()
                    # plt.close()  # 关闭图像窗口，图像不会显示也不会保存到本地文件
            
            # enlarge the best state
            next_initmap = state_id_table[best_num]
            img_buffer = env.plotmap_save(next_initmap, (np.round(time.time() - start_time, decimals=2)), ep, it)
        return img_buffer

    img_buffer = None  # 初始化 img_buffer

    for _ in range(3):
        try:
            img_buffer = train()  # 這裡調用你的train函數
            break  # 如果train成功，跳出迴圈
        except Exception as e:
            print(f"Training failed with error: {str(e)}. Retrying...")
    else:
        # 這裡的代碼將在循環完成但沒有被break語句中斷時執行
        print(f"Training failed after {3} retries.")

    return img_buffer  # 在comp的最後回傳img_buffer



def comp_in(grid, gamma):
    img_buffer = comp(grid, gamma)
    return img_buffer