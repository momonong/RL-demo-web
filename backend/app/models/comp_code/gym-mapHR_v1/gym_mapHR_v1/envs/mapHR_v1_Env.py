import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from json import JSONEncoder
import gym
import math
from gym import error, spaces
from gym.utils import closer
import matlab.engine
from IPython import display

class mapHR_v1_Env(gym.Env):
    action_space = None
    observation_space = None
    
    
    def __init__(self):
        bm = 0
        self.bm = 0
        self.size = 8
        self.size_half = int(self.size/2)
        #self.env = np.ones((self.size,self.size_half,1)).astype(int)
        #last_observation =[1,1,1,1,1,1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, #1,0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  
        #last_observation = np.array(last_observation)
        #last_observation = last_observation.reshape(1, 8, 8)
        #self.env = last_observation[0 ,:, 0:4].reshape(self.size,self.size_half)
        #self.intenv = last_observation[0 ,:, 0:4].reshape(self.size,self.size_half)
        self.episodes = 0
        self.episode_steps = 0
        self.action_space = spaces.Discrete(self.size*self.size_half)
        self.observation_space = spaces.Box(low=0.0, high=float(self.size), shape=(int(self.size),int(self.size),1))
        self.state = None
        self.obs = None
        self.obs_pre = None
        self.eng = matlab.engine.start_matlab()
        
    def ENVsize(self,size,pop=[1,1,1,1,1,1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,     1,0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]):
        self.size = int(size)
        self.size_half = int(size/2)
        self.env = np.ones((self.size,self.size_half,1)).astype(int)
        if self.size == 8:
            #last_observation =[1,1,1,1,1,1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,     #1,0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  
            if len(pop) == 64:
                pop = np.array(pop)
                pop = pop.reshape(1, 8, 8)
                pop = pop[0 ,:, 0:4].reshape(self.size,self.size_half)  
        #       self.intenv = last_observation[0 ,:, 0:4].reshape(self.size,self.size_half)
            self.env = np.array(pop).reshape(self.size,self.size_half)  
            self.intenv = np.array(pop).reshape(self.size,self.size_half) 
        else:
            self.env = np.array(pop).reshape(self.size,self.size_half)  
            self.intenv = np.array(pop).reshape(self.size,self.size_half)  
            
    def toughness(self, pop):
        state = pop.reshape((self.size,self.size_half))
        cont = int(math.log(64/self.size)/math.log(2))
        #if self.size == 8:
        #    cont = 3
           
        #else:
         #   cont = int(32/self.size)
        state_0 = state  
        for ss in range(cont):
            size = int(self.size*(2**(ss+1)))
            state_inc = np.ones((size,int(size/2)))
            for i in range(0, size, 2):
                state_inc[i][0:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i][1:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i+1] = state_inc[i] 
            state_0 = state_inc
        size = 64  
        ind_L = state_0.reshape((size,int(size/2)))
        ind_R = np.fliplr(ind_L)
        ind0_C = np.block([ind_L, ind_R])
        ind0_C = np.rot90(ind0_C, 3)
        ind0_C = ind0_C.reshape(size*size,1)
        ind_m= np.matrix(ind0_C)
        ind_m=matlab.double(ind_m.tolist())        
        toughness = np.round(self.eng.test(size,ind_m), decimals = 5)
        return toughness

    #def maxstep(self, iterations):
        
        #self.max_episode_steps = iterations
        #self.f_l = 0

    def step(self, action):
        #bm_n = 0
        self.episode_steps += 1
            
        self.env = self.env.reshape(self.size*self.size_half)
        #ind_L = self.env.reshape((self.size,self.size_half))
        #ind_R = np.fliplr(ind_L)
        #ind0_C = np.block([ind_L, ind_R])
        #ind0_C = np.rot90(ind0_C, 3)
        #ind0_C = ind0_C.reshape(self.size*self.size,1)
        #ind_m= np.matrix(ind0_C)
        #ind_m=matlab.double(ind_m.tolist())
        #f1 = self.eng.test(self.size,ind_m)
        #f1 = np.round(f1, decimals = 5)    
        self.env[action] = abs(self.env[action] - 1)
        state = self.env
        return state
    def get_rewaed(self, f1, f2,bestsave):   
        bm_n = 0
        #ind_L = self.env.reshape((self.size,self.size_half))
        #ind_R = np.fliplr(ind_L)
        #ind0_C = np.block([ind_L, ind_R])
        #ind0_C = np.rot90(ind0_C, 3)
        #ind0_C = ind0_C.reshape(self.size*self.size,1)
        #ind_m= np.matrix(ind0_C)
        #ind_m=matlab.double(ind_m.tolist())
        #f2 = self.eng.test(self.size,ind_m)
        #f2 = np.round(f2, decimals = 5)
        state = self.env    
        dif = f2 - f1
        if float(f2) > float(bestsave):
            bm_n = 1
            #self.bm = f2
        else:
            bm_n = 0
                
        if dif > 0:
            reward = float(dif)
        else:
            reward = float(dif)
                
        #if self.episode_steps == self.max_episode_steps:
            #done = True
        #else:
            #done = None 
        reward = np.round(reward, decimals = 3)
        return reward, bm_n,
    
    def plotmap(self,population):
        state = population.reshape((self.size,self.size_half))
        cont = int(math.log(64/self.size)/math.log(2))
        state_0 = state  
        for ss in range(cont):
            size = int(self.size*(2**(ss+1)))
            state_inc = np.ones((size,int(size/2)))
            for i in range(0, size, 2):
                state_inc[i][0:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i][1:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i+1] = state_inc[i] 
            state_0 = state_inc
        size = 64  
        ind_L = state_0.reshape((size,int(size/2)))
        ind_R = np.fliplr(ind_L)
        ind0_C = np.block([ind_L, ind_R])
        ind_C = np.rot90(ind0_C, 3)
        ind_C = ind_C.reshape(size*size,1)
        ind_m= np.matrix(ind_C)
        ind_m=matlab.double(ind_m.tolist())
        fitness = self.eng.test(size,ind_m)
        ind0_C = ind0_C.reshape(size,size)
        plt.figure()
        plt.title("Toughness: %.2f" % fitness, size=10)
        plt.imshow(ind0_C, cmap='gray')
        plt.show()
    
    def reset(self):
        #self.env = np.ones((self.size,self.size_half,1)).astype(int)
        #last_observation =[1,1,1,1,1,1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,0, 1, 0, 0, 1, 1, 1, 1, 0, #0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  
        #last_observation = np.array(last_observation)
        #last_observation = last_observation.reshape(1, 8, 8)
        self.env = self.intenv 
        return self.env
    def plotmap_now(self):
        #population = np.array(population)
        #get_custom_objects().update({'gelu_s': gelus(gelu)})
        #CNN_model = model_from_json(open('model.json').read())
        #CNN_model.load_weights('model.h5')
        state = self.env.reshape((self.size,self.size_half))
        cont = int(math.log(64/self.size)/math.log(2))
        state_0 = state  
        for ss in range(cont):
            size = int(self.size*(2**(ss+1)))
            state_inc = np.ones((size,int(size/2)))
            for i in range(0, size, 2):
                state_inc[i][0:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i][1:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i+1] = state_inc[i] 
            state_0 = state_inc
        size = 64  
        ind_L = state_0.reshape((size,int(size/2)))
        ind_R = np.fliplr(ind_L)
        ind0_C = np.block([ind_L, ind_R])
        ind_C = np.rot90(ind0_C, 3)
        ind_C = ind_C.reshape(size*size,1)
        ind_m= np.matrix(ind_C)
        ind_m=matlab.double(ind_m.tolist())
        fitness = self.eng.test(size,ind_m)
        ind0_C = ind0_C.reshape(size,size)
        plt.figure()
        plt.title("Toughness: %.2f" % fitness, size=10)
        plt.imshow(ind0_C, cmap='gray')
        plt.show()    
    
    
    
    
    
    
    def showmap(self,ep,it,time):
        display.clear_output(wait=True)
        #self.env = np.ones((self.size,self.size_half,1)).astype(int)
        state = self.env.reshape((self.size,self.size_half))
        cont = int(math.log(64/self.size)/math.log(2))
        state_0 = state  
        for ss in range(cont):
            size = int(self.size*(2**(ss+1)))
            state_inc = np.ones((size,int(size/2)))
            for i in range(0, size, 2):
                state_inc[i][0:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i][1:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i+1] = state_inc[i] 
            state_0 = state_inc
        size = 64  
        ind_L = state_0.reshape((size,int(size/2)))
        ind_R = np.fliplr(ind_L)
        ind0_C = np.block([ind_L, ind_R])
        ind_C = np.rot90(ind0_C, 3)
        ind_C = ind_C.reshape(size*size,1)
        ind_m= np.matrix(ind_C)
        ind_m=matlab.double(ind_m.tolist())
        fitness = self.eng.test(size,ind_m)
        ind_C = ind0_C.reshape(size,size)
        plt.title("Toughness: {} episode: {} iteration: {} time: {}" .format(np.round(fitness, decimals = 2),ep, it, time), size=10)
        plt.imshow(ind_C, cmap='gray')
        plt.show()
        plt.clf()
     
    
    def plotmap_save(self,population,time,ep,it):
        #population = np.array(population)
        #get_custom_objects().update({'gelu_s': gelus(gelu)})
        #CNN_model = model_from_json(open('model.json').read())
        #CNN_model.load_weights('model.h5')
        state = population.reshape((self.size,self.size_half))
        cont = int(math.log(64/self.size)/math.log(2))   
        state_0 = state  
        for ss in range(cont):
            size = int(self.size*(2**(ss+1)))
            state_inc = np.ones((size,int(size/2)))
            for i in range(0, size, 2):
                state_inc[i][0:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i][1:state_inc.shape[1]+1:2] = state_0[int(i/2)]
                state_inc[i+1] = state_inc[i] 
            state_0 = state_inc
        size = 64  
        ind_L = state_0.reshape((size,int(size/2)))
        ind_R = np.fliplr(ind_L)
        ind0_C = np.block([ind_L, ind_R])
        ind_C = np.rot90(ind0_C, 3)
        ind_C = ind_C.reshape(size*size,1)
        ind_m= np.matrix(ind_C)
        ind_m=matlab.double(ind_m.tolist())
        fitness = self.eng.test(size,ind_m)
        ind0_C = ind0_C.reshape(size,size)
        plt.figure()
        plt.title("Toughness: {} episode: {} iteration: {} time: {}" .format(np.round(fitness, decimals = 2),ep, it, time), size=10)
        plt.imshow(ind0_C, cmap='gray')
        plt.savefig(str(self.size)+'_'+str(time)+'.jpg',bbox_inches='tight',pad_inches=0.0 )
        plt.show()
        plt.clf()
      