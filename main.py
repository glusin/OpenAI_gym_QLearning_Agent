from collections import deque

import gym
from agent import DeepAgent

ag = DeepAgent()

gym.envs.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

env = gym.make('CartPole-v1')
score = deque(maxlen=1000)
max_score = 0
for i_episode in range(1, 1_000_001):
    render_flag = False
    if i_episode % 100 == 0:
        # printing episode number, average score for last 1000 restarts, how many moves are exploratory, longest_run
        print(i_episode, round(sum(score) / len(score), 2), ag.epsilon, max_score)
        ag.learn_from_memory()

        render_flag = True
    state = env.reset()
    for t in range(10000):
        # rgb = env.render('rgb_array')
        if render_flag:
            env.render('human')
            # print(state)
        action = ag.act(state)
        next_state, reward, done, info = env.step(action)
        # modified reward encourages stick to stay in the center
        reward = reward - (state[0] / 3) ** 2
        ag.remember(state, action, next_state, reward, done)
        ag.learn([[state, action, next_state, reward, done]])
        if done:
            score.append(t+1)
            if t+1 > max_score:
                max_score = t+1
            break

        state = next_state
