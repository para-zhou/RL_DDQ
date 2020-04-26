'''
An SAC Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.
'''

import random, copy, json, pickle
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

from .agent import Agent
from deep_dialog import dialog_config

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


########################################################################################################

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentTD3(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        self.experience_replay_pool = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.running_expereince_pool = None # hold experience from both user and world model

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5
        self.target_update_interval = 1
        self.alpha = 0.2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cur_bellman_err = 0

        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.actor = Actor(self.state_dimension, self.num_actions, self.hidden_size, self.num_actions-1).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.state_dimension, self.num_actions, self.hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)

        self.tau = 0.005
        self.policy_noise = 0.2 * (self.num_actions - 1)
        self.noise_clip = 0.5 * (self.num_actions - 1)
        self.policy_freq = 2
        self.total_it = 0
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return torch.argmax(self.actor(state), 1)


########################################################################################################################

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ SAC: Input state, output action """

        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        #   Create one-hot of acts to represent the current user action
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        #   Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Creat bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Encode last agent act
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        #   Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Encode last agent request slots
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        #   turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        #   One-hot representation of the turn count?
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        #   Representation of KB results (scaled counts)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

        #   Representation of KB results (binary)
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return torch.IntTensor([[random.randint(0, self.num_actions - 1)]])
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.select_action(representation)

    def rule_policy(self):
        """ Rule Policy """

        act_slot_response = {}

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from either environment or world model, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type: (int, ) -> Transition
        # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))

        batch = [random.choice(self.running_expereince_pool) for i in range(batch_size)]
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in range(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=100, episode=1, print_interval=1):
        """ Train SAC with experience buffer that comes from both user and world model interaction."""

        self.running_expereince_pool = list(self.experience_replay_pool) + list(self.experience_replay_pool_from_model)

        for iter_num in range(num_batches):
            for iter_batch in range(int(len(self.running_expereince_pool) / batch_size)):
                self.total_it += 1
                batch = self.sample_from_buffer(batch_size)

                state = torch.FloatTensor(batch.state).to(self.device)
                action = torch.FloatTensor(batch.action).to(self.device)
                reward = torch.FloatTensor(batch.reward).to(self.device)
                next_state = torch.FloatTensor(batch.next_state).to(self.device)
                term = torch.FloatTensor(np.asarray(batch.term, dtype=np.float32)).to(self.device)

                with torch.no_grad():
                    # noise = (torch.randn_like(action, dtype=float) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    next_action = self.actor_target(next_state).gather(1, torch.tensor(batch.action))
                    # next_action = (next_action + noise).clamp(0, 28)
                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + (1 - term) * self.gamma * target_Q
                
                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action)
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                # Delayed policy updates
                if self.total_it % self.policy_freq == 0:
                    # Compute actor losse
                    actor_loss = -self.critic.Q1(state, self.actor(state).gather(1, torch.tensor(batch.action))).mean()

                    self.cur_bellman_err += actor_loss # added by former model

                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if len(self.experience_replay_pool) != 0 and (episode % print_interval == 0):
                print("cur bellman err %.4f, experience replay pool %s, model replay pool %s" % (
                    float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                    len(self.experience_replay_pool), len(self.experience_replay_pool_from_model)))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path,))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print("Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename))

    def reset_dqn_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
