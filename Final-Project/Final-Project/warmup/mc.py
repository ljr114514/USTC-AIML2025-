import matplotlib
import numpy as np
import sys
import time
from collections import defaultdict

from envs import BlackjackEnv
import plotting
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    基于给定的 Q 函数和 epsilon 创建一个 ε-贪心策略。
    
    参数:
        Q: 一个字典，映射 状态 -> 动作值。
           每个值是一个长度为 nA 的 numpy 数组
        epsilon: 选择随机动作的概率，0 到 1 之间的浮点数。
        nA: 环境中的动作数量。
    
    返回:
        一个函数，接受观察值作为参数，返回一个长度为 nA 的 numpy 数组，
        表示每个动作的选择概率。
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA  # Initialize the action probabilities
        best_action = np.argmax(Q[observation])      # Find the best action
        A[best_action] += (1.0 - epsilon)            # Add (1 - epsilon) probability to the best action
        return A
    return policy_fn

def mc_first_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    使用 ε-贪心策略的蒙特卡洛控制。
    寻找最优的 ε-贪心策略。
    
    参数:
        env: OpenAI gym 环境
        num_episodes: 要采样的回合数
        discount_factor: 折扣因子 gamma
        epsilon: 选择随机动作的概率，0 到 1 之间的浮点数
    
    返回:
        元组 (Q, policy)
        Q 是一个字典，映射 状态 -> 动作值
        policy 是一个函数，接受观察值作为参数，返回动作概率
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        #########################Implement your code here#########################
        
        # Step 1: Generate an episode: an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):  # 限制步数防止死循环
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # Step 2: Find first-visit index for each (state, action) pair
        G = 0
        # Step 3: Calculate returns backward, update only at first-visit time step
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            
            # 检查这个 (state, action) 是否在之前（t之前的时刻）出现过
            # 如果没出现过，说明时刻 t 是该状态动作对的“首次访问”
            if not any(x[0] == state and x[1] == action for x in episode[0:t]):
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                # 更新 Q 值：平均回报
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
        #########################Implement your code end#########################
    return Q, policy


def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################

        # Step 1: Generate an episode
        episode = []
        state = env.reset()
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # Step 2: Calculate returns for each (state, action) pair (every-visit)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            
            # 无需检查，直接计入平均值
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
        #########################Implement your code end#########################

    return Q, policy

if __name__ == "__main__":
    # First-Visit Monte Carlo
    Q, policy = mc_first_visit(env, num_episodes=10000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    plotting.plot_value_function(V, title="Optimal Value Function", 
        file_name="First_Visit_Value_Function_Episodes_10000")
    
    # Every-Visit Monte Carlo
    # Q, policy = mc_every_visit(env, num_episodes=10000, epsilon=0.1)
    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     V[state] = np.max(actions)
    # plotting.plot_value_function(V, title="Optimal Value Function", 
    #     file_name="Every_Visit_Value_Function_Episodes_10000")