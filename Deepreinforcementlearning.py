import numpy as np
import pylab as pl
import networkx as nx

edges = [
    (0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
    (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
    (8, 9), (7, 8), (1, 7), (3, 9)
]

goal = 10
MATRIX_SIZE = 11
gamma = 0.75

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)   # CHANGED: fixed seed for consistency
nx.draw(G, pos, with_labels=True)
pl.show()
M = np.full((MATRIX_SIZE, MATRIX_SIZE), -1)

for (i, j) in edges:
    M[i, j] = 100 if j == goal else 0
    M[j, i] = 100 if i == goal else 0

M[goal, goal] = 100

Q = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

def available_actions(state):
    return np.where(M[state] >= 0)[0]

def sample_next_action(actions):
    return np.random.choice(actions)

def update_q(current_state, action):
    next_max = np.max(Q[action])
    Q[current_state, action] = M[current_state, action] + gamma * next_max

scores = []
for _ in range(1000):
    state = np.random.randint(0, MATRIX_SIZE)
    actions = available_actions(state)
    action = sample_next_action(actions)
    update_q(state, action)
    scores.append(np.sum(Q))

current_state = 0
path = [current_state]

while current_state != goal:
    next_state = np.argmax(Q[current_state])
    path.append(next_state)
    current_state = next_state

print("Most efficient path:")
print(path)
pl.plot(scores)
pl.xlabel("Iterations")
pl.ylabel("Q-value Sum")
pl.show()

police = [2, 4, 5]
drug_traces = [3, 8, 9]

env_police = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
env_drugs = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

def collect_environment(action):
    if action in police:
        return 'p'
    if action in drug_traces:
        return 'd'
    return None

def update_q_env(current_state, action):
    next_max = np.max(Q[action])
    Q[current_state, action] = M[current_state, action] + gamma * next_max
    env = collect_environment(action)
    if env == 'p':
        env_police[current_state, action] += 1
    elif env == 'd':
        env_drugs[current_state, action] += 1

def available_actions_env(state):
    actions = available_actions(state)
    q_values = Q[state, actions]
    if np.any(q_values < 0):
        actions = actions[q_values >= 0]
    return actions if len(actions) > 0 else available_actions(state)

scores = []
for _ in range(1000):
    state = np.random.randint(0, MATRIX_SIZE)
    actions = available_actions_env(state)
    action = sample_next_action(actions)
    update_q_env(state, action)
    scores.append(np.sum(Q))

print("Police matrix:")
print(env_police)
print("\nDrug traces matrix:")
print(env_drugs)

pl.plot(scores)
pl.xlabel("Iterations")
pl.ylabel("Reward gained")
pl.show()
