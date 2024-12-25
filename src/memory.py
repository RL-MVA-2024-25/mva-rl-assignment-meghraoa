from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import numpy as np
from torch import nn, optim
import os
from memory import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

config = {
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'buffer_size': 1000000,
    'epsilon_min': 0.05,
    'epsilon_max': 1.0,
    'decay_factor': 0.05,
    'decay_option': 'linear',
    'epsilon_decay_period': 10000,
    'epsilon_delay_decay': 100,
    'batch_size': 2048,
    'gradient_steps': 1,
    'update_target_strategy': 'replace',
    'update_target_freq': 1000,
    'hidden_dim': 1024,
    'criterion': torch.nn.SmoothL1Loss(reduction='none'),
    'double_dqn': True,
    'per': True,
    'alpha': 0.2,
    'beta': 0.6,
    'beta_increment_per_sampling': 5e-6,
    'prior_eps': 1e-6
}

MODEL_PATH = "ddqn_per_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim, nb_neurons=1024):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, nb_neurons),
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.network(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, config=None, training=False):
        if training:
            self.double_dqn = config.get('double_dqn', False)
            self.per = config.get('per', False)
            self.alpha = config.get('alpha', 0.2)
            self.beta = config.get('beta', 0.6)
            self.beta_increment_per_sampling = config.get('beta_increment_per_sampling', 5e-6)
            self.prior_eps = config.get('prior_eps', 1e-6)
            self.decay_factor = config.get('decay_factor', 1)
            self.decay_option = config.get('decay_option', 'linear')
            self.gamma = config.get('gamma', 0.95)
            self.batch_size = config.get('batch_size', 100)
            buffer_size = config.get('buffer_size', int(1e5))

            if self.per:
                self.memory = PrioritizedReplayBuffer(buffer_size, self.alpha, self.beta, self.beta_increment_per_sampling, device)
            else:
                self.memory = ReplayBuffer(buffer_size, device)

            self.epsilon_max = config.get('epsilon_max', 1.0)
            self.epsilon_min = config.get('epsilon_min', 0.01)
            self.epsilon_stop = config.get('epsilon_decay_period', 1000)
            self.epsilon_delay = config.get('epsilon_delay_decay', 20)
            self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
            hidden_dim = config.get('hidden_dim', 128)
            self.model = DQNModel(env.observation_space.shape[0], env.action_space.n, nb_neurons=hidden_dim).to(device)
            self.target_model = deepcopy(self.model).to(device)
            self.criterion = config.get('criterion', nn.MSELoss(reduction='none'))
            lr = config.get('learning_rate', 0.001)
            self.optimizer = config.get('optimizer', optim.Adam(self.model.parameters(), lr=lr))
            self.nb_gradient_steps = config.get('gradient_steps', 1)
            self.update_target_strategy = config.get('update_target_strategy', 'replace')
            self.update_target_freq = config.get('update_target_freq', 20)
            self.update_target_tau = config.get('update_target_tau', 0.005)
    
    def act(self, observation, use_random=False):
        if use_random and np.random.rand() < self.epsilon:
            return env.action_space.sample()
        observation = torch.Tensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.model(observation)
        return action_values.argmax().item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            if self.per:
                X, A, R, Y, D, W, I = self.memory.sample(self.batch_size)
            else:
                X, A, R, Y, D = self.memory.sample(self.batch_size)

            if not self.double_dqn:
                QYmax = self.target_model(Y).max(1)[0].detach()
            else:
                QYmax = self.target_model(Y).gather(1, self.model(Y).argmax(1).unsqueeze(1)).squeeze(1)
        
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss_values = self.criterion(QXA, update.unsqueeze(1))

            if self.per:
                loss = torch.mean(loss_values * W)
            else:
                loss = torch.mean(loss_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.per:
                priorities = loss_values.squeeze().detach().cpu().numpy() + self.prior_eps
                self.memory.update_priorities(
                    indices=I,
                    priorities=priorities
                )

    def train(self, max_episode=100):
        best_score = 0
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        self.epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
                if self.decay_option == 'linear':
                 self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
                elif self.decay_option == 'logistic':
                    self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                            sigmoid(self.decay_factor * (self.epsilon_stop - episode))
                else:
                    raise Exception(f"L'option {self.decay_option} n'est pas reconnue.")

                
            # select epsilon-greedy action
            action = self.act(state, use_random=True)
                
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
                
            # update target network if needed
            if self.update_target_strategy == 'replace' and step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            elif self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
                
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                score = evaluate_HIV(agent=self, nb_episode=1)
                if score > best_score:
                    best_score = score
                    self.save(MODEL_PATH)
                print(
                    f"Episode: {episode:3d}/{max_episode}, "
                    f"Epsilon: {self.epsilon:.2f}, "
                    f"Batch size: {len(self.memory):5d}, "
                    f"Total reward: {episode_cum_reward:13.1f}, "
                    f"Score: {score:13.1f}"
                )
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model = DQNModel(env.observation_space.shape[0], env.action_space.n)
        self.model.load_state_dict(torch.load(f'{os.getcwd()}/{MODEL_PATH}',  map_location=device, weights_only=True))
