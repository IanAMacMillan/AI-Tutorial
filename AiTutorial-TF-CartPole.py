'''
Ian MacMillan 
January 2021
Based on Github: tankala
See https://blog.tanka.la/2018/10/19/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python/
'''

import os
import gym
import random
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #scilence warning about tf optimization


env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 60
intial_games = 20000

def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        
        env.reset()
    #print(accepted_scores)
    return training_data    


training_data = model_data_preparation()

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
    num_epocs=10
    print("\nTraining model with "+str(num_epocs)+" Epocs")
    model.fit(X, y, epochs=num_epocs)
    print("Model Trained")
    return model

#os.environ['KMP_DUPLICATE_LIB_OK']='True' #Mac OS problem https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial

trained_model = train_model(training_data)

#test model
scores = []
choices = []
num_tests=10
print("\nTesting Model Effectiveness With "+str(num_tests)+" Tests")
for each_game in tqdm(range(num_tests)):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        #env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break
        
    env.reset()
    scores.append(score)
    
#print(scores)
print('Average Score:',sum(scores)/len(scores))
#print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
