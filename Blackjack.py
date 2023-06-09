import numpy as np
import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="human")

def player_policy(observation):
    score, dealer_score, ace = observation
    if score >= 20:
        return 0
    else:
        return 1

def episode(policy, env):
    states = []
    actions = []
    rewards = []

    observation, info = env.reset()

    while True:
        states.append(observation)

        action = policy(observation)
        actions.append(action)

        observation, reward, terminated, turncated, info = env.step(action)
        rewards.append(reward)

        if terminated:
            break

    return states, actions, rewards

def first_visit_mc(policy, env):
    value_table = {}
    N = {}

    for _ in range(100):
        states, _, rewards = episode(policy, env)
        returns = 0

        for t in range(len(states)):
            R = rewards[t]
            S = states[t]

            returns += R

            if S not in states[:t]:
                if S not in N:
                    N[S] = 1
                    value_table[S] = returns
                else:
                    N[S] += 1
                    value_table[S] += (returns - value_table[S]) / N[S]

        print(f"Value function for state {S}: {value_table[S]}")
    print(f"Maximum Player win chance: {max(value_table.items(), key=lambda x: x[1])}")
    print(f"Maximum Dealer win chance: {min(value_table.items(), key=lambda x: x[1])}")

    return value_table

value = first_visit_mc(player_policy, env)