'''
An SAC Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.
'''

import random, copy, json
import pickle
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

from .agent import Agent
from deep_dialog import dialog_config
from deep_dialog.qlearning import DQN
from .sac_model import soft_update, hard_update, QNetwork, GaussianPolicy, DeterministicPolicy, ActorCritic, CategoricalPolicy


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentSAC(Agent):
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
        self.tau = 0.005
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        self.critic = QNetwork(self.state_dimension, self.num_actions, self.hidden_size).to(self.device)
        self.critic_target = QNetwork(self.state_dimension, self.num_actions, self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.policy = GaussianPolicy(self.state_dimension, self.num_actions, self.hidden_size, np.arange(self.num_actions)).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=1e-3)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate == False:
            action, log_prob, _ = self.policy.sample(state)
        else:
            _, log_prob, action = self.policy.sample(state)
        # return action.detach().cpu().numpy()[0]
        return torch.IntTensor([torch.argmax(log_prob).item()])

    def update_parameters(self, batch):
        # SAC update of transition memory batch
        # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))
        # (16, 273) (16, 1) (16, 1) (16, 273) (16, 1)
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.FloatTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        term_batch = torch.FloatTensor(batch.term).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            next_state_action, _ = next_state_action.max(1)
            next_state_action = next_state_action.unsqueeze(1)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target) * (1 - term_batch)
        # (16, 1), (16, 1) <- (16, 273+1)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)
        pi, _ = pi.max(1)
        pi = pi.unsqueeze(1)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.cur_bellman_err += policy_loss.item()

        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)

        # return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

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
            return torch.IntTensor([random.randint(0, self.num_actions - 1)])
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
                batch = self.sample_from_buffer(batch_size)
                self.update_parameters(batch)                

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
