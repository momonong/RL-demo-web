
from keras.applications.imagenet_utils import obtain_input_shape

from tensorflow.keras import backend as K
from tensorflow.keras import layers
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose,Convolution2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs

from tensorflow.python.keras import regularizers

from re import L
import keras
import gc
import datetime
# import cv
import time
import cv2

import numpy as np
import tensorflow as tf
import imageio

def Average(lst):
    return sum(lst) / len(lst)

def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params

def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name

def basic_conv_block(filters, stage, block, strides=(2, 2)):

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides,
                   kernel_regularizer = regularizers.l2(1e-5), 
                   name=conv_name + '1', **conv_params)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3),
                   kernel_regularizer = regularizers.l2(1e-5)
                   , name=conv_name + '2', **conv_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer

def basic_identity_block(filters, stage, block):

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3),
                   kernel_regularizer = regularizers.l2(1e-5),
                   name=conv_name + '1', **conv_params)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3),
                   kernel_regularizer = regularizers.l2(1e-5),
                   name=conv_name + '2', **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer

def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_decode(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            kernel_regularizer = regularizers.l2(1e-5),
                            padding='same', name=up_name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same',
                   kernel_regularizer = regularizers.l2(1e-5),
                   name=conv_name+'2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'2')(x)
        x = Activation('relu', name=relu_name+'2')(x)

        return x
    return layer

def handle_block_names_decode(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name

def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_decode(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size,
                   kernel_regularizer = regularizers.l2(1e-5),
                   padding='same', name=conv_name+'1')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters, kernel_size,
                   kernel_regularizer = regularizers.l2(1e-5),
                   padding='same', name=conv_name+'2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'2')(x)
        x = Activation('relu', name=relu_name+'2')(x)

        return x
    return layer

def build_resnet(
     repetitions=(2, 2, 2, 2),
     include_top=True,
     input_tensor=None,
     input_shape=None,
     classes=1000,
     block_type='usual'):

    # Determine proper input shape
    input_shape = obtain_input_shape(input_shape,
                                      default_size=128,
                                      min_size=128,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    if block_type == 'basic':
        conv_block = basic_conv_block
        identity_block = basic_identity_block
    else:
        conv_block = usual_conv_block
        identity_block = usual_identity_block
    
    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2),
               kernel_regularizer = regularizers.l2(1e-5),
               name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
    
    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            
            filters = init_filters * (2**stage)
            
            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)
                
            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)
                
            else:
                x = identity_block(filters, stage, block)(x)
                
    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x)

    return model

def build_unet(backbone, classes, last_block_filters, skip_layers,
               n_upsample_blocks=5, upsample_rates=(2,2,2,2,2),
               block_type='upsampling', activation='sigmoid',
               **kwargs):

    input = backbone.input
    x = backbone.output
    print(x)
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_layers = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                    for l in skip_layers])
    for i in range(n_upsample_blocks):
        
        # check if there is a skip connection
        if i < len(skip_layers):
            print(backbone.layers[skip_layers[i]])
            print(backbone.layers[skip_layers[i]].output)
            skip = backbone.layers[skip_layers[i]].output
        else:
            skip = None

        up_size = (upsample_rates[i], upsample_rates[i])
        filters = last_block_filters * 2**(n_upsample_blocks-(i+1))

        x = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(x)

    if classes < 2:
        activation = 'sigmoid'

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model

def UResNet34(input_shape=(None, None,1), classes=1, decoder_filters=16, decoder_block_type='transpose',
                       encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = build_resnet(input_tensor=None,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=False,
                         block_type='basic')
    backbone._name = 'resnet34'
    
    
    skip_connections = list([129, 74, 37, 5]) # for resnet 34
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model._name = 'u-resnet34'

    return model

def get_model():
    model_url = 'app/models_data/seg_model_weights_best_three_change.hdf5'
    seg_model = UResNet34(input_shape=(128,128,2),encoder_weights=False, classes=1)
    seg_model.load_weights(model_url, by_name=False)
    return seg_model

def update_windows(img, porosity, target_por, current_kappa, border=None, Good=None, current_difference=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    test_ans_ = np.copy(img)
    test_ans_ = np.squeeze(test_ans_,axis=0)
    test_ans_ = np.squeeze(test_ans_,axis=-1)    
    test_ans_[test_ans_>=0.38] = 1
    test_ans_[test_ans_<0.38] = 0
    test_ans_ = test_ans_ * 255
    test_ans_ = cv2.cvtColor(test_ans_, cv2.COLOR_GRAY2BGR)
    extend_width = 128
    extend_box = np.zeros((128,256,3))
    extend_box[:,:128]=test_ans_
    test_ans_ = extend_box
    #print(border)
    test_ans_[np.where((test_ans_==[255,255,255]).all(axis=2))] = (225/255,105/255,65/255)
    test_ans_[np.where((test_ans_==[0,0,0]).all(axis=2))] = [255/255,255/255,255/255]
    if border != None:
        test_ans_[:,border] = (0,255,255)
    test_ans_ = cv2.resize(test_ans_, (1024,512), interpolation = cv2.INTER_AREA)
    cv2.putText(test_ans_,"Target Crystal Ratio:",(512, 118),font,1.5,(0,165/255,255/255),3,cv2.LINE_AA)
    cv2.putText(test_ans_,str(round(target_por, 3)),(512, 168),font,1.5,(0,165/255,255/255),2,cv2.LINE_AA)
    cv2.putText(test_ans_,"Current Crystal Ratio:",(512, 246),font,1.5,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(test_ans_,str(round(porosity, 3)),(512, 296),font,1.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(test_ans_,"Kappa:",(512, 374),font,1.5,(255/255,0,0),3,cv2.LINE_AA)
    cv2.putText(test_ans_,str(current_kappa),(512, 424),font,1.5,(255/255,0,0),2,cv2.LINE_AA)
    if Good == True:
        cv2.putText(test_ans_,"GOOD:"+str(round(current_difference,3)),(512, 492),font,2,(0,255/255,0),3,cv2.LINE_AA)
    elif Good == False:
        cv2.putText(test_ans_,"NO GOOD:"+str(round(current_difference,3)),(512, 492),font,2,(0,0,255/255),3,cv2.LINE_AA)
    # cv2.imshow("test",test_ans_)
    # cv2.waitKey(1)
    test_ans_1 = np.copy(test_ans_)
    test_ans_1[:,:,0] = test_ans_[:,:,2]
    test_ans_1[:,:,1] = test_ans_[:,:,1]
    test_ans_1[:,:,2] = test_ans_[:,:,0]
    return test_ans_1

def porosity_calculation_full(img):
    #total_ice = np.sum(por_img[:,:right_border])
    pic_height = 128
    pic_width = 128
    total_ice = np.sum(img * -1 + 1)
    total_space = pic_width*pic_height
    return total_ice/total_space

def get_actor(state_amount = 2, action_amount = 3):
    #other states (current porosity, timestep, kappa,)
    state_input = layers.Input(shape=(state_amount,))
    x_state = layers.Dense(1024, activation="linear", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2')(state_input)
    x_state = layers.BatchNormalization()(x_state)
    x_state = layers.LeakyReLU(0.1)(x_state)
    x_state = layers.Dense(1024, activation="linear", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2')(x_state)
    x_state = layers.BatchNormalization()(x_state)
    x_state = layers.LeakyReLU(0.1)(x_state)
    x_state = layers.Dense(1024, activation="linear", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2')(x_state)
    x_state = layers.BatchNormalization()(x_state)
    x_state = layers.LeakyReLU(0.1)(x_state)
    x_state = layers.Dense(1024, activation="linear", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2')(x_state)
    x_state = layers.BatchNormalization()(x_state)
    x_state = layers.LeakyReLU(0.1)(x_state)
    x_state = layers.Dense(1024, activation="linear", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2')(x_state)
    x_state = layers.BatchNormalization()(x_state)
    out = layers.LeakyReLU(0.1)(x_state)

    # 11 kinds of outputs, highest probability is the answer
    # outputs = layers.Dense(action_amount, activation="softmax", kernel_regularizer='l2', kernel_initializer=keras.initializers.he_uniform())(out)
    outputs = layers.Dense(action_amount, activation="tanh", kernel_regularizer='l2', kernel_initializer=keras.initializers.he_normal())(out)
    # Our upper bound is 2.2 for Kappa limit.
    outputs = 1.65 + 0.55*outputs
    model = tf.keras.Model(state_input, outputs)
    return model

actor_model = get_actor()

def policy( state_input, current_episode):
    #print(img_state)
    #print(state_input)
    sampled_actions = tf.squeeze(actor_model(state_input))
    # current_kappa_index = np.argmax(sampled_actions)
    # current_kappa = 1.1+current_kappa_index*0.1
    current_kappa = sampled_actions
    # Adding noise to action
    if current_episode <= 1000:
        normal_range = 0.2
    elif current_episode > 1000 and current_episode < 2000:
        normal_range = 0.1
    elif current_episode > 2000 and current_episode < 8000:
        normal_range = 0.07
    else:
        normal_range = 0.03
    # print(sampled_actions)
    # sampled_actions = current_kappa + np.random.normal( 0.0, normal_range )
    sampled_actions = current_kappa
    sampled_actions = np.round(sampled_actions,2)
    # We make sure action is within bounds
    upper_bound = 2.2
    lower_bound = 1.1
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action), sampled_actions

def get_size_for_RL():
    X_train_img = np.load('/home/jonny/Conan/ice_former_1024/data/X_train_img_15.npy')
    X_train_img = np.concatenate((X_train_img, 
                                np.load('/home/jonny/Conan/ice_former_1024/data/X_train_img_16.npy')
                                ))

    X_train_type = np.load('/home/jonny/Conan/ice_former_1024/data/X_train_type_15.npy')
    X_train_type = np.concatenate((X_train_type, 
                                np.load('/home/jonny/Conan/ice_former_1024/data/X_train_type_16.npy')))
    period = 44
    X_train_img_r = X_train_img.reshape(int(X_train_type.shape[0]/period),period, 128,128,1)
    X_train_type_r = X_train_type.reshape(int(X_train_type.shape[0]/period),period,1)
    X_data_for_RL = X_train_img_r[:,3,:,:,:]
    type_data_for_RL = X_train_type_r[:,3,:]
    return X_data_for_RL, type_data_for_RL

X_data_for_RL, type_data_for_RL = get_size_for_RL()

def env_reset(index):
    #reset take new image as the start of the game
    if index != -1:
        random_index = index
    else:
        random_index = np.random.randint(120)
    return X_data_for_RL[random_index], type_data_for_RL[random_index]

image_test, type_data = env_reset(-1)

def generate_gif(request_target_porosity = np.random.uniform(0.3,0.7)):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # To store average reward history of last few episodes
    ep_reward_list = []
    gif_images = []
    # state contents (current porosity, timestep, kappa,)
    # image state contains one image from the data set
    #mainly the second image 
    # states(current porosity, timestep, kappa,)
    # print("total testing data amount:",X_data_for_RL.shape)
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
        target_porosity = request_target_porosity
        ######################################
        # target_porosity = np.random.uniform(0.3,0.7)
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
            
            seg_model = get_model()
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
                # print("Episode * {} * Error is ==> {}".format(ep, porosity_error))
                # print("average Episode * {} * Error is ==> {}".format(ep, Average(ep_reward_list)))
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

def generate_gif(input):
    return input + 1