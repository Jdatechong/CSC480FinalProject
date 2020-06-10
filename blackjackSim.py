import gym
import math
import os.path
import pickle
env = gym.make('Blackjack-v0')

sums = [i for i in range(2, 22)]
dealer_hands = [i for i in range(1,11)]
ace = [True, False]
cards = [i for i in range(1,11)] + [10] + [10] + [10]
poss_actions = [0, 1]
#print(cards)

alpha = 0.4
num_games = 1000000
discount_factor = 0.95
method = 4
if method == 0: # reward = ucb
    wr_file = "training_wr_0.p"
    q_file = "training_q_0.p"
elif method == 1: # reward = win rate
    wr_file = "training_wr_1.p"
    q_file = "training_q_1.p"
elif method == 2: # reward = sum
    wr_file = "training_wr_2.p"
    q_file = "training_q_2.p"
elif method == 3: # reward = ucb  utility factors future states into calculations
    wr_file = "training_wr_3.p"
    q_file = "training_q_3.p"
elif method == 4: # reward = sum  utility factors future states into calculations
    wr_file = "training_wr_4.p"
    q_file = "training_q_4.p"


def init_tables(q_util, win_rates):
    for summ in sums:
        for d_hand in dealer_hands:
            for is_ace in ace:
                for act in poss_actions:
                    q_util[(summ, d_hand, is_ace), act] = 0
                    win_rates[(summ, d_hand, is_ace), act] = (0, 0)
    print(q_util)
    print(win_rates)


def get_next_util(q_util, prev_state):
    #print(prev_state)
    if prev_state[1] == 0:
        #print(q_util[prev_state])
        return q_util[prev_state]
    sum_q = 0
    for i in cards:
        next_state = prev_state[0]
        next_sum = next_state[0] + i
        if next_sum > 21:
            sum_q = sum_q - 1
            continue
        if i == 1:
            next_state = (next_sum, next_state[1], True)
        else:
            next_state = (next_sum, next_state[1], next_state[2])
        #print(next_state)
        max_act_q = max(q_util[(next_state, 0)], q_util[(next_state, 1)])
        sum_q = sum_q + max_act_q
        #print("max_act_q: " + str(max_act_q))

    #print(sum_q/len(cards))
    return sum_q/len(cards)


def get_reward(state, wins_games, num_turns):
    if wins_games[1] == 0:
        return 1
    else:
        if method == 0 or 3:
            return (wins_games[0] / wins_games[1]) + math.sqrt(2) * math.sqrt(math.log2(num_turns)/wins_games[1])
        elif method == 1:
            return wins_games[0] / wins_games[1]
        else:
            return state[0][0]

def get_utility(q_util, state, rs):
    if method == 3 or method == 4:
        q_prime = get_next_util(q_util, state)
        q_util[state] = (1 - alpha) * q_util[state] + alpha * (rs + discount_factor * q_prime)
        #print(state)
        #print(q_util[state])
    else:
        q_util[state] = (1 - alpha) * q_util[state] + alpha * rs
    return q_util[state]


def main():
    q_util = {}
    win_rates = {}
    if os.path.isfile(q_file) and os.path.isfile(wr_file):
        q_util = pickle.load(open(q_file, "rb"))
        win_rates = pickle.load(open(wr_file, "rb"))
        num_turns = 0
        for i in win_rates:
            num_turns = num_turns + win_rates[i][1]
    else:
        num_turns = 0
        init_tables(q_util, win_rates)
    wins = 0

    for i_episode in range(num_games):
        observation = env.reset()
        steps = []
        while True:
            #print(observation)
            hit_state = (observation, 1)
            stick_state = (observation, 0)
            rew_hit = get_reward(hit_state, win_rates[hit_state], num_turns)
            rew_stick = get_reward(stick_state, win_rates[stick_state], num_turns)

            hit_util = get_utility(q_util, hit_state, rew_hit)
            stick_util = get_utility(q_util, stick_state, rew_stick)
            #print("hit: " + str(hit_util) + "stick:" + str(stick_util))
            if hit_util >= stick_util and observation[0] != 21:
                action = 1
            else:
                action = 0
            steps.append((observation, action))
            observation, reward, done, info = env.step(action)
            num_turns = num_turns + 1
            if done or action == 0:
                #print("Finished Game. Reward: " + str(reward))
                for step in steps:
                    #print("step" + str(step))
                    #print(win_rates[step])
                    r_win, r_games = win_rates[step]
                    r_games = r_games + 1
                    if reward == 1:
                        r_win = r_win + 1
                    win_rates[step] = (r_win, r_games)
                    #print(win_rates[step])

                if reward == 1:
                    wins = wins + 1
                break
    #for i in win_rates:
    #    print(str(i) + " " + str(win_rates[i]))
    pickle.dump(win_rates, open(wr_file, "wb"))
    pickle.dump(q_util, open(q_file, "wb"))
    print(wins / num_games)




main()

