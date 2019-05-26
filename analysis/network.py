import numpy as np
import tensorflow as tf
import time
from pprint import pprint

class Actions:
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
##        the v_pref(=1.0 m/s) and 5 directions
##        array([[ 1.        , -0.52359878],
##               [ 1.        , -0.26179939],
##               [ 1.        ,  0.        ],
##               [ 1.        ,  0.26179939],
##               [ 1.        ,  0.52359878]])
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
##        the 0.5*v_pref(=0.5 m/s) and 3 directions
##        the following:
##               [ 0.5       , -0.52359878]
##               [ 0.5       ,  0.        ]
##               [ 0.5       ,  0.52359878]
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
##        the 0.5*v_pref(=0.5 m/s) and 3 directions
##        the following:
##               [ 0.         -0.52359878]
##               [ 0.          0.        ]
##               [ 0.          0.52359878]
        self.num_actions = len(self.actions)
##        11 actions in total


class NetworkVPCore:
    def __init__(self, device, model_name, num_actions):
        self.device = device            # CPU
        self.model_name = model_name    # "network"
        self.num_actions = num_actions  # 11
        self.nodes = None

        self.graph = tf.Graph()         # init of basic parameters
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph = self.graph,
                    config = tf.ConfigProto(
                        allow_soft_placement = True,
                        log_device_placement = False,
                        gpu_options = tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                vs = tf.global_variables()
                self.nodes = {v.name: v for v in vs}
                self.saver = tf.train.Saver(self.nodes, max_to_keep=0)
    
    def _create_graph_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, Config.NN_INPUT_SIZE], name="X") # column is 75, row is not decided yet, the input structure for later use in `feed_dict` as input
        # shape: (?, 75)
 
    def _create_graph_outputs(self):
        # FCN
        self.fc1 = tf.layers.dense(inputs=self.final_flat, units = 256, use_bias = True, activation=tf.nn.relu, name = "fullyconnected1")
        # input: 256; output: 256
        # shape: (?, 256), kernel: shape: (256, 256), bias: shape: (256,)

        # Cost: p
        self.logits_p = tf.layers.dense(inputs = self.fc1, units = self.num_actions, name = "logits_p", activation = None)
        # input: 256; output: 11
        # shape: (?, 11), kernel: shape: (256, 11), bias: shape: (11,)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        # input: 11; output: 11
        # each number in result + 0.0001 and / 1.0011, then use `argmax` to decide next action
        # shape: (?, 11)

##        # logits_v
##        self.logits_v = tf.layers.dense(inputs = self.fc1, units = 1, name = "logits_v", activation = None)
##        # input: 256; output: 1
##        # shape: (?, 1), kernel: shape: (256, 1), bias: shape: (1,)

        # Cost: v
        # https://github.com/mfe7/cadrl_ros/issues/3
        self.logits_v = tf.squeeze(tf.layers.dense(inputs=self.fc1, units = 1, use_bias = True, activation=None, name = 'logits_v'), axis=[1])

    def predict_p(self, x, audio):
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})
##        return self.sess.run(self.logits_v, feed_dict={self.x: x}) # logits_v

    def predict_p_and_v(self, x):
        # https://github.com/NVlabs/GA3C/blob/master/ga3c/NetworkVP.py
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
        self.saver.restore(self.sess, filename)


class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(self.__class__, self).__init__(device, model_name, num_actions)

    def _create_graph(self):
        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        if Config.USE_REGULARIZATION:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0) # no regularizer, since 0.0 disables the regularizer
        else:
            regularizer = None

        if Config.NORMALIZE_INPUT:
            self.avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype = tf.float32) # shape: (75,)
            self.std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype = tf.float32) # shape: (75,)
            self.x_normalized = (self.x - self.avg_vec) / self.std_vec                 # shape: (?, 75)
        else:
            self.x_normalized = self.x


        if Config.MULTI_AGENT_ARCH == "RNN":
            num_hidden = 64
            max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED # 10
            self.num_other_agents = self.x[:,0]               # number of other agents, the list of first element of each input list of self.x, shape: (?,)
            self.host_agent_vec = self.x_normalized[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
            #                                       :,            1           :                           4+1=5                     :
            # the S vector in paper, approach part, shape: (?, 4)
            self.other_agent_vec = self.x_normalized[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
            #                                        :,                           4+1=5                     :
            # the ~So vector in paper, approach part, there are 10 places for other agents, if there is less than 10 agents, then 0 is used as placeholder, shape: (?, 70)
            self.other_agent_seq = tf.reshape(self.other_agent_vec, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])
            #                                                        -1,     10    ,                    7
            # shape: (?, 10, 7)
##            global cell
##            cell = tf.contrib.rnn.LSTMCell(num_hidden)
##            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(cell, self.other_agent_seq, dtype=tf.float32, sequence_length=self.num_other_agents)
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_hidden), self.other_agent_seq, dtype=tf.float32, sequence_length=self.num_other_agents)
            #  (?, 10, 64)  , LSTMStateTuple                                            (    64    ),   shape: (?, 10, 7) ,                                         shape: (?,)
            # about the LSTM cell:
            # the number of units/the size of hidden state is 64, so the output size of cell is 64
            # the weight of cell is a list of 2 arrays: kernel (71, 256) and bias (256,)
            # about the RNN:
            # `rnn_outputs` is not used (shape: (?, 10, 64)), only use the hidden state (shape: (?, 64))
            self.rnn_output = self.rnn_state.h
            # use rnn_state.h (hidden state), the output is hn in paper, shape: (?, 64)
            self.layer1_input = tf.concat([self.host_agent_vec, self.rnn_output],1, name="layer1_input")
            # the se vector in paper, shape: (?, 4)+(?, 64)=(?, 68)
            self.layer1 = tf.layers.dense(inputs=self.layer1_input, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = "layer1")
            # possible source of 256: https://github.com/NVlabs/GA3C/blob/master/ga3c/NetworkVP.py
            # corresponding paper:    https://arxiv.org/pdf/1611.06256.pdf
            # reference:              https://arxiv.org/pdf/1602.01783.pdf
            # the weight of cell is a list of 2 arrays: kernel (68, 256) and bias (256,)
            # the first FC layer, shape: (?, 256)
##            global w
##            w = cell.get_weights()

        self.layer2 = tf.layers.dense(inputs=self.layer1, units=256, activation=tf.nn.relu, name = "layer2")
        # the second FC layer, shape: (?, 256)
        # the weight of cell is a list of 2 arrays: kernel (256, 256) and bias (256,)
        self.final_flat = tf.contrib.layers.flatten(self.layer2)
        # shape: (?, 256)
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False # not used
    USE_REGULARIZATION  = True  # no effect
    ROBOT_MODE          = True
    EVALUATE_MODE       = True  # equal to PLAY_MODE in class `Agent`, `related to time_remaining_to_reach_goal`

    SENSING_HORIZON     = 8.0   # the visible distance of an agent

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20 # only mentioned in paper, but not used in fact
    MULTI_AGENT_ARCH = "RNN"

    DEVICE                        = "/cpu:0" # Device

    HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between (last 2 terms reversed in order)
    RNN_HELPER_LENGTH = 1              # num other agents
    AGENT_ID_LENGTH = 1                # id
    IS_ON_LENGTH = 1                   # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5])  # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0]) # used for formatting the input
    RNN_HELPER_STD_VECTOR = np.array([1.0]) # used for formatting the input
    IS_ON_AVG_VECTOR = np.array([0.0]) # not used
    IS_ON_STD_VECTOR = np.array([1.0]) # not used

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == "RNN":
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10                                   # used in training of GA3C-CADRL-10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH # 7
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH                # 4
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH  # 1+4+10*7=75
            FIRST_STATE_INDEX = 1 # the cut-off start point, exclude the first element in list

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)]) # shape: (75,)
            #                                                    0;         0, 0, 1, 0.5;         0, 0, 0, 0, 0.5, 0, 1; (repeat for 10 times)
##            array([0,                         RNN_HELPER_AVG_VECTOR
##                   0, 0, 1, 0.5,              HOST_AGENT_AVG_VECTOR
##                   0, 0, 0, 0, 0.5, 0, 1,     OTHER_AGENT_AVG_VECTOR...
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1,
##                   0, 0, 0, 0, 0.5, 0, 1])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)]) # shape: (75,)
            #                                                    1;        5, 3.14, 1, 1;           5, 5, 1, 1, 1, 5, 1; (repeat for 10 times)
##            array([1,                         RNN_HELPER_STD_VECTOR
##                   5, 3.14, 1, 1,             HOST_AGENT_STD_VECTOR
##                   5, 5, 1, 1, 1, 5, 1,       OTHER_AGENT_STD_VECTOR...
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1,
##                   5, 5, 1, 1, 1, 5, 1])
            
    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH # 75+1=76
    NN_INPUT_SIZE = FULL_STATE_LENGTH                               # 75


if __name__ == "__main__": # query speed test
    actions = Actions().actions         # numpy array of 11 actions (speed and orientation)
    num_actions = Actions().num_actions # 11
    nn = NetworkVP_rnn(Config.DEVICE, "network", num_actions)
##    nn.simple_load()
    nn.simple_load("network_01900000")
##    nn.simple_load("network_02360000")
##    nn.simple_load("network_01653000")

    obs = np.zeros((Config.FULL_STATE_LENGTH)) # 75 0's, shape: (75,)
    obs = np.expand_dims(obs, axis=0)          # 75 0's, shape: (1, 75)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0,0] = 10 # num other agents
        obs[0,1] = np.random.uniform(0.5, 10.0) # dist to goal
        obs[0,2] = np.random.uniform(-np.pi, np.pi) # heading to goal
        obs[0,3] = np.random.uniform(0.2, 2.0) # pref speed
        obs[0,4] = np.random.uniform(0.2, 1.5) # radius
        predictions = nn.predict_p(obs, None)[0]
##        predictions = nn.predict_p_and_v(obs)[0]
    t_end = time.time()
    print("avg query time:", (t_end - t_start)/num_queries)
    print("total time:", t_end - t_start)
    action = actions[np.argmax(predictions)]
    print("action:", action)
    pprint(vars(nn))
