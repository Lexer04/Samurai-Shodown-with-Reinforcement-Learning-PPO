import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import FrameStack

GAME_ENV = 'SamuraiShodown-Genesis'
STATE_1P_1= 'Level1.HaohmaruVsWanFu'
STATE_1P_2='Level1.WanfuVsHaohmaru'
STATE_2P='Level1.HaohmaruVsWanFu.2P'
POLICY_1='MlpPolicy'
POLICY_2='MlpPolicy'
TIMESTEPS=200000
MAX_EPISODES = 10000
PRINT_INTERVAL = 1000

class RewardCallback(BaseCallback):
    def __init__(self, print_interval, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.print_interval = print_interval

    def _on_step(self):
        if self.n_calls % self.print_interval == 0:
            mean_reward = np.cumsum(self.locals['rewards'])
            #print(f'Timestep: {self.n_calls}, Mean Reward: {mean_reward}')
        return True

def apply_wrappers(env):
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)

    return env

def apply_wrappers(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env

def make_env(game, state):
    env = retro.make(game=game, state=state)
    return apply_wrappers(env)

def main():
    print("Time to Train Player 1! :)")
    env = retro.make(game=GAME_ENV, state= STATE_1P_1)
    apply_wrappers(env)

    p1_model = PPO(policy=POLICY_1, env=env, verbose=True)
    p1_model.learn(total_timesteps=TIMESTEPS, callback=RewardCallback(print_interval=PRINT_INTERVAL))

    env.close()

    print("Time to Train Player 2! :)")

    env = retro.make(game=GAME_ENV, state= STATE_1P_2)
    apply_wrappers(env)

    p2_model = PPO(policy=POLICY_2, env=env, verbose=True)
    p2_model.learn(total_timesteps=TIMESTEPS, callback=RewardCallback(print_interval=PRINT_INTERVAL))

    env.close()

    env_2p = retro.make(game=GAME_ENV, state= STATE_2P, players=2)
    apply_wrappers(env_2p)

    state = env_2p.reset()

    episode_count = 0
    print("TIME TO TEST THE PLAYERS! \(^_^)/")
    while episode_count < MAX_EPISODES:
        env_2p.render()

        if isinstance(state, tuple):
            state = state[0]

        p1_actions = p1_model.predict(state)
        p2_actions = p2_model.predict(state)

        p1_actions = p1_actions[0]
        p2_actions = p2_actions[0]

        p1_actions = np.array(p1_actions).astype(int)
        p2_actions = np.array(p2_actions).astype(int)

        actions = np.append(p1_actions, p2_actions)

        # Step in the environment
        result = env_2p.step(actions)

        # Unpack the result
        state, reward, done, info = result[:4]
        #print(f"Result: {result}")
        #print(f"Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print(f"Episode: {episode_count + 1}")
            env_2p.reset()
            episode_count += 1
    env_2p.close()

if __name__ == '__main__':
    main()
