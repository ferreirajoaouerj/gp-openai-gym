import os
import sys
import time

import gym
import pybulletgym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import results_plotter

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

best_mean_reward, n_steps = -np.inf, 0
tinit = time.time()


def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Melhor recompensa média: {:.2f} - Última recompensa média (por episódio): {:.2f}"
                  .format(best_mean_reward, mean_reward))
            print("Tempo de execução (s): ", time.time() - tinit)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print("Salvando o modelo...")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


log_dir = "rl_logs/doublependulum/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
env.render()
env = Monitor(env, log_dir, allow_early_resets=True)
time_steps = 5e5

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

modo = input("New run? (y) or (n)\n")

if modo is 'y':

    sys.stdout = open(log_dir + 'console_log', 'w')
    model.learn(total_timesteps=time_steps, callback=callback)
    sys.stdout.close()

    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG Doublependulum")
    plt.show()

else:

    model = DDPG.load(log_dir + 'best_model.pkl')
    env.render()

    apt_lista = []

    for i in range(100):
        obs, done = env.reset(), False
        apt_acum = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            apt_acum += rewards
        apt_lista.append(apt_acum)

    print("Desempenho (100 ep): ", np.sum(apt_lista)/len(apt_lista))

    for i in range(5):
        obs, dones = env.reset(), False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

    obs, done, t = env.reset(), False, 0
    act, observ, rew, tsteps = np.array([]), np.ndarray(shape=(11, 0)), np.array([]), np.array([])
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        act = np.append(act, action)
        observ = np.hstack((observ, obs.reshape((11, 1))))
        rew = np.append(rew, rewards)
        tsteps = np.append(tsteps, t)
        t += 1
        time.sleep(1/120)
    figura, eixo = plt.subplots(nrows=4, ncols=1, sharex='all')
    eixo[0].plot(tsteps, act, ls='-', marker='.', markersize=3, alpha=0.3, color='b')
    eixo[0].set_ylabel('Ação')
    eixo[0].grid(True)
    eixo[1].plot(tsteps, observ[0], ls='-', marker='.', markersize=3, alpha=0.3, color='g')
    eixo[1].set_ylabel(r'$s$')
    eixo[1].grid(True)
    eixo[2].plot(tsteps, observ[1], ls='-', marker='.', markersize=3, alpha=0.3, color='r')
    eixo[2].set_ylabel(r'$\sin(\theta)$')
    eixo[2].grid(True)
    eixo[3].plot(tsteps, observ[2], ls='-', marker='.', markersize=3, alpha=0.3, color='m')
    eixo[3].set_ylabel(r'$\sin(\gamma)$')
    eixo[3].grid(True)
    eixo[3].set_xlabel('Tempo')
    eixo[3].set_xlim(0, 400)
    plt.plot()
