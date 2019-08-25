#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import signal
import sys
import gym
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from ddpg import DDPG
from actor import ActorNetwork
from critic import CriticNetwork
from exp_replay import ExpReplay
from exp_replay import Step
from ou import OUProcess
from tensorflow import keras
from ns3gym import ns3env

env = gym.make('ns3-v0')

ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.005
MEM_SIZE = 100000

START_MEM = 1000
N_H1 = 400
N_H2 = 300

STATE_SIZE = 4
ACTION_SIZE = 1
BATCH_SIZE = 10
MAX_STEPS = 200
FAIL_PENALTY = 0
ACTION_RANGE = 4
EVALUATE_EVERY = 5

NUM_EPISODES = 400
env._max_episode_steps = MAX_STEPS

time_history = []
rew_history = []
episodes_history = []

def summarize(cum_reward, i, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag="cumulative reward", simple_value=cum_reward)
  summary_writer.add_summary(summary, i)
  summary_writer.flush()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print("Plot Learning Performance")
    mpl.rcdefaults()
    mpl.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(10,4))
    plt.grid(True, linestyle='--')
    plt.title('Learning Performance')
    plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
    plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.legend(prop={'size': 12})

    #plt.savefig('learning.pdf', bbox_inches='tight')
    plt.show()

signal.signal(signal.SIGINT, signal_handler)

#env = gym.make('Pendulum-v0')

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
print('size: ' + str(s_size) + '/' + str(a_size))

actor = ActorNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=ACTOR_LEARNING_RATE, n_h1=N_H1, n_h2=N_H2, tau=TAU)
critic = CriticNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=CRITIC_LEARNING_RATE, n_h1=N_H1, n_h2=N_H2, tau=TAU)
noise = OUProcess(ACTION_SIZE)
exprep = ExpReplay(mem_size=MEM_SIZE, start_mem=START_MEM, state_size=[STATE_SIZE], kth=-1, batch_size=BATCH_SIZE)

sess = tf.Session()
with tf.device('/{}:0'.format('CPU')):
  agent = DDPG(actor=actor, critic=critic, exprep=exprep, noise=noise, action_bound=ACTION_RANGE)
sess.run(tf.initialize_all_variables())

for i in range(NUM_EPISODES):
    cur_state = env.reset()
    cum_reward = 0
    # tensorboard summary
    summary_writer = tf.summary.FileWriter('/tmp/pendulum-log-0'+'/train', graph=tf.get_default_graph())

    if (i % EVALUATE_EVERY) == 0:
      print ('====evaluation====')
    for t in range(MAX_STEPS):
      if (i % EVALUATE_EVERY) == 0:
        env.render()
        action = agent.get_action(cur_state, sess)[0]
      else:
        # decaying noise
        action = agent.get_action_noise(cur_state, sess, rate=(NUM_EPISODES-i)/NUM_EPISODES)[0]
      next_state, reward, done, info = env.step(action)
      if (i % EVALUATE_EVERY) == 0:
          print(cur_state)
          print(action)
          print('reward: ' + str(reward))
          print('------------------------------------------------------')
      if done:
        cum_reward += reward
        agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        print("Done! Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        summarize(cum_reward, i, summary_writer)
        break
      cum_reward += reward
      agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
      if t == MAX_STEPS - 1:
        print ("Done!! Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        print (action)
        summarize(cum_reward, i, summary_writer)
    agent.learn_batch(sess)
    time_history.append(t)
    rew_history.append(cum_reward)

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

#plt.savefig('learning.pdf', bbox_inches='tight')
plt.show()
