#This is a reinforcement learner on a cart pole enviroment
#it starts to do fairly well after a few 100 episodes
import numpy as np
import tensorflow as tf
import keras as ke
import keras.layers as la
import gym,random
import matplotlib.pyplot as plt
 
#where to save file
locc="rl2_model1.h5"

env = gym.make('CartPole-v0')

#tries to load an existing model from file for more training.
#otherwise, produces a model that takes in an observation and action and predicts reward
try:
    model=ke.models.load_model(locc)
except OSError:
    p1 = la.Input(shape=env.observation_space.shape,name="obs")
    p2 = la.Input(shape=(1,),name="act")
    p3=la.Dense(20,activation="relu")(p1)
    p4=la.Dense(20,activation="relu")(p2)
    p5=la.Concatenate(axis=-1)([p3,p4])
    p6=la.Dense(40,activation="relu")(p5)
    p7=la.Dense(40,activation="relu")(p6)
    p8=la.Dense(1,activation="relu")(p7)
    model = ke.Model(inputs=[p1,p2], outputs=p8, name="mnist_model")
    model.summary()
    model.compile( optimizer=ke.optimizers.RMSprop(),loss=ke.losses.MeanSquaredError())#,metrics=[ke.losses.MeanSquaredError()])


gamma=0.02
def ExpDecay(x):
    #converts x[i] into sum(x[j]*gamma**j; for j>=i)*(1-gamma)
    #this gives a reward to train the agent to predict.

    #first note that this function is linear.
    #for any i and j>i, note that j-i has a binary expansion, with (j-i).bit_length() at most n.bit_length()
    #values are stepped backwards in powers of 2, and multiplied by y**(2**s) as they are moved.
    #there is only one sequence of stepping back and staying in place that lets x[j] influence x[i]
    #and all the y terms multiply together to make y**(j-i)

    #I wrote it like this because numpy allows quick batch processing. 
    
    n=x.size
    y=1-gamma
    x=x*(1-y)
    for i in range(n.bit_length()):
        j=1<<i
        x[:-j]+=x[j:]*y
        y=y*y
    return x
acts=[]
obss=[]
rwds=[]
#these act as piles of data to train on
def s():
    model.save(locc)

for itr in range(1000):
    observation=env.reset()

    rewards_n=[]
    
    for i in range(1000):
        
        env.render()
        obss.append(observation)

        
        if  random.random()+0.1<10/(i+1):
            #random exploration. Stops after 100 iterations.
            act=env.action_space.sample()
            
        else:
            #take the action with the higher predicted reward
            prr=model.predict({"obs":np.stack([observation]*2),"act":np.array([[0],[1]])})
            if prr[0,0]>prr[1,0]:
                act=0
            else:
                act=1

        observation, reward, done, info=env.step(act) # take an action
        
        acts.append(act)
        rewards_n.append(reward)
        
        if done:
            break
    
    rwds.extend(list(ExpDecay(np.array(rewards_n))))#apply ExpDecay 
    mlen=5000
    rwds=rwds[-mlen:]
    obss=obss[-mlen:]
    acts=acts[-mlen:]
    #get rid of any data thats too old.
    
    rwdsN=np.array(rwds)[:,None]
    obssN=np.stack(obss,axis=0)
    actsN=np.array(acts)[:,None]
    #turn into np arrays for training on
    
    
    print(itr,i,model.train_on_batch({"obs":obssN,"act":actsN},rwdsN))
env.close()
