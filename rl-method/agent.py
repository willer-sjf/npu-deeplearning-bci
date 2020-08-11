import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, size):
    
        self._storage  = []
        self._maxsize  = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
        
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
        
        
def reward_shape(origin_reward, discount):
    length = len(origin_reward)
    new_reward = np.zeros_like(origin_reward, dtype=float)
    for i in reversed(range(length)):
        new_reward[i] = origin_reward[i] + (discount * new_reward[i+1] if i+1 < length else 0)
    return new_reward
    
class Q_Net(nn.Module):
    
    def __init__(self,v_dim):
        super(Q_Net,self).__init__()
        
        self.v_dim = v_dim
        self.pre_feature = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.first_order_feature= nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.second_order_feature = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(64*3,64)
        self.fc2 = nn.Linear(64  , 1)
    def forward(self, feature, mean_feature, var_feature):
        
        f_pre = self.pre_feature(feature)
        f_fst = self.first_order_feature(mean_feature - feature)
        f_scd = self.second_order_feature(var_feature)
        
        f_merge = torch.cat([f_pre, f_fst, f_scd], 1)
        q = F.relu(self.fc1(f_merge))
        q = self.fc2(q)
        return q


class ADAgent:
    def __init__(self, data, label):
        
        self.gamma = 0.97
        self.epsilon = 0.3
        self.max_drop = 2
        self.env = DropEnv(data, label)
        
        self.eval_net   = Q_Net(data.shape[-1])

        #self.writer = SummaryWriter()
        
        self.interaction_counter = 0
        self.learn_step_counter = 0
        self.step_update = 10
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma = 0.97)

    def select_action(self, state):
        action = -1
        q_max = None
        action_dim = len(state)
        if random.random() < self.epsilon:
            for index, (fea, mean, var) in enumerate(state):
                q = self.eval_net(fea, mean, var)
                if action == -1:
                    action = index
                    q_max = q
                elif q > q_max:
                    action = index
                    q_max = q
        else:
            action = random.randint(0, action_dim)
            q_max  = self.eval_net(state[action])
        return action, q_max
    
    def train(self):
        for _ in range(10):
            self.interaction_counter += 1
            state = self.env.reset()
            q = []
            r = []
            for i in range(self.max_drop):
                action, q_pred = self.select_action(state)
                new_state, reward, done = self.env.step(action)

                q += [q_pred]
                r += [reward]
                if done:
                    #self.summary.add_scalar('epsiode lens', i, self.interaction_counter)
                    print('i')
                    break
                state = new_state

            # r = reward_shape(r, self.gamma)

            self.learn(q, r)
        
    def learn(self, q, r):

        self.learn_step_counter += 1

        self.epsilon *= 1.005
        self.epsilon = min(self.epsilon, 0.97)

        q_score = torch.stack(q ,dim = 0)
        r_score = torch.FloatTensor(r)
        r_score = r_score.reshape(-1, 1)
        
        loss = F.mse_loss(q_score, r_score)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        #self.writer.add_scalar('loss', loss, self.learn_step_counter)
        print('loss : {:.4f}'.format(loss.item()))

