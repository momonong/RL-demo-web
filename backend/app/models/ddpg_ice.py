
# To store reward history of each episode
from re import L
import gc
import datetime
import csv
import time
import cv2

import numpy as np
import tensorflow as tf
import imageio

def Average(lst):
    return sum(lst) / len(lst)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# To store average reward history of last few episodes
ep_reward_list = []
gif_images = []
# state contents (current porosity, timestep, kappa,)
# image state contains one image from the data set
#mainly the second image 
# states(current porosity, timestep, kappa,)
print("total testing data amount:",X_data_for_RL.shape)
total_tests = 500
save_gif_done = False
current_frame =None
for ep in range(1):
    #one episode tests a batch of images
    #get new batch of images
    random_index = np.random.randint(120)
    # random_index = indexes[ep]
    prev_image, prev_kappa_data = X_data_for_RL[random_index], type_data_for_RL[random_index]
    prev_kappa = prev_kappa_data[0]
    prev_porosity = porosity_calculation_full(image_test)
    current_timestep = 3
    kappa_channel = prev_image*float(prev_kappa)
    org_kappa = prev_kappa
    #construct the initial inputs of the model to acquire the output of middle layers
    image_input = tf.concat([prev_image, kappa_channel],axis = 2)
    # construct the previous state 
    episodic_reward = 0
    # create target porosity 
    ######################################
    #                input               #
    ######################################
    target_porosity = np.random.uniform(0.3,0.7)
    # target_porosity = targets[ep]
    # create state
    current_kappa = prev_kappa
    prev_state = np.array([target_porosity - prev_porosity, float(current_kappa)])
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    initial_state = tf_prev_state
    # initialize parameters
    kappa_change_times = 0
    last_border = 0

    #acquire actions of this episode
    action, org_action = policy(tf_prev_state, ep)
    while True:
        
        if current_timestep == 3:
            current_kappa = action[0]
        # elif current_timestep == 10:
        #     current_kappa = action[1]
        elif current_timestep == 14:
            current_kappa = action[1]
        elif current_timestep == 29: 
            current_kappa = action[2]
        
        # turn prev_state into tf tensor and add a dim for batch        
        tf_prev_state_img = tf.expand_dims(tf.convert_to_tensor(prev_image), 0) 
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        
        # use the previous image and kappa to create image
        kappa_channel = tf_prev_state_img*current_kappa
        image_input = tf.concat([tf_prev_state_img, kappa_channel],axis = 3)
        # get the corresponding action of current state
        # print(tf_prev_state)
        kappa_change_times +=1
        
        next_img = seg_model(image_input)
        current_timestep += 1
        current_porosity = porosity_calculation_full(next_img)
        # new state
        state = np.array([target_porosity - current_porosity, current_kappa])
        porosity_error = target_porosity - current_porosity
       
        # End this episode when `done` is True
        if current_timestep == 44:
            if porosity_error < 0.1:
                GOOD_param = True
            else:
                GOOD_param = False
            current_frame = update_windows(next_img,current_porosity,target_porosity, current_kappa,Good=GOOD_param, current_difference=porosity_error)
            if ep < 20:
                current_frame = current_frame * 255
                current_frame = current_frame.astype(np.uint8)
                for i in range(10):
                    gif_images.append(current_frame)
            ep_reward_list.append(porosity_error)
            print("Episode * {} * Error is ==> {}".format(ep, porosity_error))
            print("average Episode * {} * Error is ==> {}".format(ep, Average(ep_reward_list)))
            break
        else:
            current_frame = update_windows(next_img,current_porosity,target_porosity, current_kappa)
            if ep < 20:
                current_frame = current_frame * 255
                current_frame = current_frame.astype(np.uint8)
                if current_timestep % 2 == 0:
                    gif_images.append(current_frame)
            
        prev_image = next_img[0]
        prev_state = state

imageio.mimsave('Test.gif', gif_images, fps=10)
print("SAVED GIF!!!!!!")

cv2.destroyAllWindows()
# Plotting graph
# Episodes versus Avg. Rewards