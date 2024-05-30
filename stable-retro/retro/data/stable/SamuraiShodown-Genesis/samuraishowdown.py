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
POLICY_1='CnnPolicy'
POLICY_2='MlpPolicy'
TIMESTEPS=10000
MAX_EPISODES = 10000
PRINT_INTERVAL = 1000

class RewardCallback(BaseCallback):
    def __init__(self, print_interval, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.print_interval = print_interval

    def _on_step(self):
        if self.n_calls % self.print_interval == 0:
            mean_reward = np.mean(self.locals['rewards'])
            print(f'Timestep: {self.n_calls}, Mean Reward: {mean_reward}')
        return True

# def apply_wrappers(env):
#     env = WarpFrame(env)
#     env = ClipRewardEnv(env)
#     env = FrameStack(env, 4)
#     return env

# def make_env(game, state):
#     env = retro.make(game=game, state=state)
#     return apply_wrappers(env)

# def main():
#     try:
#         env_1p_1 = make_vec_env(lambda: make_env(GAME_ENV, STATE_1P_1), n_envs=1)
#         p1_model = PPO(POLICY_1, env_1p_1, verbose=1)
#         p1_model.learn(total_timesteps=TIMESTEPS)
#         env_1p_1.close()
#     except Exception as e:
#         print(f"Error with env_1p_1: {e}")

#     try:
#         env_1p_2 = make_vec_env(lambda: make_env(GAME_ENV, STATE_1P_2), n_envs=1)
#         p2_model = PPO(POLICY_2, env_1p_2, verbose=1)
#         p2_model.learn(total_timesteps=TIMESTEPS)
#         env_1p_2.close()
#     except Exception as e:
#         print(f"Error with env_1p_2: {e}")

#     try:
#         env_2p = retro.make(game=GAME_ENV, state=STATE_2P, players=2)
#         env_2p = apply_wrappers(env_2p)
#         state = env_2p.reset()

#         while True:
#             env_2p.render()

#             p1_actions, _ = p1_model.predict(state)
#             p2_actions, _ = p2_model.predict(state)

#             actions = np.append(p1_actions, p2_actions)

#             state, reward, done, info = env_2p.step(actions)

#             if done:
#                 state = env_2p.reset()
#     except Exception as e:
#         print(f"Error with env_2p: {e}")

#===============================================================================
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
    env = retro.make(game=GAME_ENV, state= STATE_1P_1)
    apply_wrappers(env)

    p1_model = PPO(policy=POLICY_1, env=env, verbose=True)
    p1_model.learn(total_timesteps=TIMESTEPS, callback=RewardCallback(print_interval=PRINT_INTERVAL))

    env.close()

    env = retro.make(game=GAME_ENV, state= STATE_1P_2)
    apply_wrappers(env)

    p2_model = PPO(policy=POLICY_2, env=env, verbose=True)
    p2_model.learn(total_timesteps=TIMESTEPS)

    env.close()

    env_2p = retro.make(game=GAME_ENV, state= STATE_2P, players=2)
    apply_wrappers(env_2p)

    state = env_2p.reset()

    episode_count = 0

    while episode_count < MAX_EPISODES:
        env_2p.render()

        if isinstance(state, tuple):
            state = state[0]

        p1_actions = p1_model.predict(state)
        p2_actions = p2_model.predict(state)

        # actions = np.append(p1_actions[0], p2_actions[0])

        # state, reward, done, info = env_2p.step(actions)

        # Extract actions from the tuples
        p1_actions = p1_actions[0]
        p2_actions = p2_actions[0]

        # Convert actions to integers if they are not already
        p1_actions = np.array(p1_actions).astype(int)
        p2_actions = np.array(p2_actions).astype(int)

        actions = np.append(p1_actions, p2_actions)


        # print(f"State: {state}")
        # print(f"Player 1 Actions: {p1_actions}")
        # print(f"Player 2 Actions: {p2_actions}")
        # print(f"Combined Actions: {actions}")


        # Step in the environment
        result = env_2p.step(actions)

        # Unpack the result
        state, reward, done, info = result[:4]
        #print(f"Result: {result}")
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print(f"Episode: {episode_count + 1}, Reward: {reward}")
            env_2p.reset()
            episode_count += 1
    env_2p.close()

    # while True:
    #     # env_2p.render()

    #     # # Handle state being a tuple or numpy array
    #     # if isinstance(state, tuple):
    #     #     state = state[0]
        
    #     # p1_actions, _ = p1_model.predict(state)
    #     # p2_actions, _ = p2_model.predict(state)

    #     # actions = np.append(p1_actions, p2_actions)

    #     # # Capture only the first four values returned by step
    #     # result = env_2p.step(actions)
    #     # state, reward, done, info = result[:4]

    #     # if done:
    #     #     state = env_2p.reset()
    
    #     env_2p.render()

    #     p1_actions, _ = p1_model.predict(state[0] if isinstance(state, tuple) else state)
    #     p2_actions, _ = p2_model.predict(state[0] if isinstance(state, tuple) else state)

    #     actions = np.append(p1_actions, p2_actions)

    #     state, reward, done, info = env_2p.step(actions)

    #     if done:
    #         state = env_2p.reset()

    # while episode_count < MAX_EPISODES:
    #     env_2p.render()

    #     if isinstance(state, tuple):
    #         state = state[0]
        
    #     p1_actions, _ = p1_model.predict(state)
    #     p2_actions, _ = p2_model.predict(state)

    #     actions = np.append(p1_actions, p2_actions)

    #     result = env_2p.step(actions)
    #     state, reward, done, info = result[:4]

    #     if done:
    #         state = env_2p.reset()
    #         episode_count += 1

    # env_2p.close()


#====================================================================================


if __name__ == '__main__':
    main()