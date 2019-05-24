import agent
import network
import util
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

def whiten(color, percent):
    # RGB: 0-255, e.g.: #233B6C; percent: b/w 0-1, e.g.: 0.01
    color = np.array([int("0x"+color[1:][2*i:2*i+2], 16) for i in range(3)])
    white = np.array([255, 255, 255])
    vector = white-color
    whiter = color+vector*percent
    return "#"+"".join([f"{int(i):#0{4}x}"[2:] for i in whiter]).upper()

def get_action(host_agent, other_agents, nn):
    # Convert agent states into observation vector
    obs = host_agent.observe(other_agents)[1:]
    obs = np.expand_dims(obs, axis=0)
    # Query the policy based on observation vector
    predictions = nn.predict_p(obs, None)[0]
    raw_action = possible_actions.actions[np.argmax(predictions)]
    action = np.array([host_agent.pref_speed*raw_action[0], util.wrap(raw_action[1] + host_agent.heading_global_frame)])
##    print(f"action of host agent {host_agent.id}: {action}")
    return action

# Load trained network
possible_actions = network.Actions()
num_actions = possible_actions.num_actions # 11
nn = network.NetworkVP_rnn(network.Config.DEVICE, "network", num_actions)
##nn.simple_load("./network_01900000")
##nn.simple_load("network_02360000")
nn.simple_load("network_01653000")

dt = 1 # 1s
##colors = ["red", "green", "orange", "blue"]
colors = ["#FF0000", "#228C22", "#FFA500", "#0000FF"]

# Set current state of host agent
start_x = 2
start_y = 5
goal_x = 3
goal_y = 2
radius = 0.5
pref_speed = 1.2
heading_angle = 0
index = 0
v_x = 0
v_y = 0

plt.scatter(start_x, start_y, s=1000*radius, c=colors[0]) # start of host agent
plt.scatter(goal_x, goal_y, s=1000*radius, facecolors="none", edgecolors=colors[0]) # goal of host agent

host_agent = agent.Agent(start_x, start_y, goal_x, goal_y, radius, pref_speed, heading_angle, index)
host_agent.vel_global_frame = np.array([v_x, v_y])
host_agent.print_agent_info()

# Set current state of other agents
# Sample observation data in a format easily generated from sensors
other_agents_x = [-1,-2,-3]
other_agents_y = [2,3,4]
other_agents_r = [0.5, 0.4, 0.3]
other_agents_vx = [1.0, 0.6, 0.2]
other_agents_vy = [0.0, 0.6, 0.8]
num_other_agents = len(other_agents_x)

# Create Agent objects for each observed dynamic obstacle
other_agents = []
for i in range(num_other_agents):
    x = other_agents_x[i]
    y = other_agents_y[i]
    v_x = other_agents_vx[i]
    v_y = other_agents_vy[i]
    radius = other_agents_r[i]
    
    plt.scatter(x, y, s=1000*radius, c=colors[i+1]) # start of other agents
    
    # dummy info - unobservable states not used by NN, just needed to create Agent object
    heading_angle = np.arctan2(v_y, v_x) 
    pref_speed = np.linalg.norm(np.array([v_x, v_y]))
    goal_x = x + 5.0
    goal_y = y + 5.0

    plt.scatter(goal_x, goal_y, s=1000*radius, facecolors="none", edgecolors=colors[i+1]) # goal of other agents
    
    other_agents.append(agent.Agent(x, y, goal_x, goal_y, radius, pref_speed, heading_angle, i+1))

for i in other_agents:
    i.print_agent_info()

all_agents = [deepcopy(host_agent)]+deepcopy(other_agents)

##print("\n\n")
####for i in range(20): # 20 rounds
####    for j in range(4):
####        ha = all_agents[j]
####        oa = list(range(4))
####        oa.remove(j)
####        oa = [all_agents[k] for k in oa]
####        host_agent_action = get_action(ha, oa, nn)
####        ha.update_state(host_agent_action, dt)
####        plt.scatter(ha.pos_global_frame[0], ha.pos_global_frame[1], s=1000*ha.radius, c=whiten(colors[j], 0.02*(i+1)))
####plt.show()

count = 1
while not all_agents[0].is_at_goal:
    for j in range(4):
        ha = all_agents[j]
        if ha.too_far_away:
            break
        oa = list(range(4))
        oa.remove(j)
        oa = [all_agents[k] for k in oa]
        host_agent_action = get_action(ha, oa, nn)
        ha.update_state(host_agent_action, dt)
        plt.scatter(ha.pos_global_frame[0], ha.pos_global_frame[1], s=1000*ha.radius, c=colors[j])
    count += 1
plt.title("After {} movements, the host agent arrives its goal position".format(count))
plt.show()
