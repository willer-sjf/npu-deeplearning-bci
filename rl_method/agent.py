import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import env

def reward_shape(origin_reward, discount):
    length = len(origin_reward)
    new_reward = np.zeros_like(origin_reward, dtype=np.float32)
    for i in reversed(range(length)):
        new_reward[i] = origin_reward[i] + (discount * new_reward[i+1] if i+1 < length else 0)
    return new_reward

class Q_Net(nn.Module):

    def __init__(self, v_dim, h_dim=64):
        super(Q_Net,self).__init__()

        self.v_dim = v_dim
        self.h_dim = h_dim
        self.pre_feature = nn.Sequential(
            nn.Linear(v_dim, h_dim),
            nn.Tanh(),
        )
        self.first_order_feature= nn.Sequential(
            nn.Linear(v_dim, h_dim),
            nn.Tanh(),
        )
        self.second_order_feature = nn.Sequential(
            nn.Linear(v_dim, h_dim),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(h_dim*3, h_dim)
        self.fc2 = nn.Linear(h_dim  , 1)

    def forward(self, feature, mean_feature=None, var_feature=None):

        if mean_feature == None:
            feature, mean_feature, var_feature = feature.chunk(3, -1)
        f_pre = self.pre_feature(feature)
        f_fst = self.first_order_feature(mean_feature - feature)
        f_scd = self.second_order_feature(var_feature)

        f_merge = torch.cat([f_pre, f_fst, f_scd], -1)
        q = F.relu(self.fc1(f_merge))
        q = self.fc2(q)
        return q

class ReplayBuffer:
    def __init__(self, size):

        self._storage  = []
        self._maxsize  = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, reward, obs_tp1, done):
        data = (obs_t, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, rewards, obses_tp1, dones = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            rewards.append(reward)
            obses_tp1.append(obs_tp1,)
            dones.append(done)
        return obses_t, rewards, obses_tp1, dones

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class ADAgent:
    def __init__(
        self,
        data,
        label,
        reward_model_path,
        gamma=1,
        epsilon=0.3,
        max_drop=7,
        buffer_size=2000,
        batch_size=2,
        train_epoch=1000,
        drop_reward=0.0001
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.max_drop = max_drop
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.check_step = 100

        tdata, vdata, tlabel, vlabel = train_test_split(data, label, test_size=0.2)
        self.env = env.DropEnv(tdata, vdata, tlabel, vlabel, drop_reward, reward_model_path)

        self.device = torch.device('cuda')
        self.eval_net = Q_Net(data.shape[-1])
        self.eval_net_gpu = Q_Net(data.shape[-1]).to(self.device)

        self.val_net = preprocess.get_reward_net(data[0].shape[-1], reward_model_path)
        self.memory = ReplayBuffer(buffer_size)
        self.writer = SummaryWriter("adrl-runs/ADAgent_" + str(datetime.datetime.now()))

        self.interaction_counter = 0
        self.validation_counter = 0
        self.learn_step_counter = 0
        self.step_update = 10
        self.training = True

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.98)

    def select_action(self, state):
        action = -1
        q_max = None
        action_dim = len(state) - 1
        if random.random() < self.epsilon or not self.training:
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
            q_max  = self.eval_net(*state[action])
        return torch.cat(state[action], 0), action, q_max

    def test(self, test_size=25):
        self.validation_counter += 1
        total_init_sim = 0
        total_drop_sim = 0
        total_drop_lens = 0

        i = 0
        self.training = False
        for _ in range(test_size):
            state, init_sim = self.env.reset()
            for i in range(self.max_drop):
                state_tp1, action, q_pred = self.select_action(state)
                new_state, result, done = self.env.step(action)
                if done:
                    break
                state = new_state

            total_init_sim  += init_sim
            total_drop_sim  += result
            total_drop_lens += i

        self.training = True
        self.writer.add_scalar('accuracy/raw', total_init_sim / test_size, self.validation_counter)
        self.writer.add_scalar('accuracy/drop', total_drop_sim / test_size, self.validation_counter)
        self.writer.add_scalar('accuracy/radio(drop_to_raw)', total_drop_sim / total_init_sim, self.validation_counter)
        self.writer.add_scalar('drop_lens/val', total_drop_lens / test_size, self.validation_counter)

    def train(self):
        for _ in range(self.train_epoch):
            self.interaction_counter += 1
            state = self.env.reset()

            i = 0
            for i in range(self.max_drop):
                state_tp1, action, q_pred = self.select_action(state)
                new_state, reward, done = self.env.step(action)

                if done:
                    break
                if i != 0:
                    self.memory.add(state_t, reward, state_tp1, done)
                state = new_state
                state_t = state_tp1

            self.writer.add_scalar('drop_lens/train', i, self.interaction_counter)

            if len(self.memory) >= self.batch_size:
                self.learn()
            if self.interaction_counter % self.check_step == 0:
                self.env.eval()
                self.test()
                self.env.train()

    def learn(self):

        self.learn_step_counter += 1

        self.epsilon *= 1.005
        self.epsilon = min(self.epsilon, 0.98)

        batch_state, batch_reward, batch_next_state, _ = self.memory.sample(self.batch_size)
        batch_state  = torch.stack(batch_state, 0).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1).to(self.device)
        batch_next_state = torch.stack(batch_next_state, 0).to(self.device)

        q_eval = self.eval_net_gpu(batch_state)
        q_next = self.eval_net_gpu(batch_next_state)
        q_target = batch_reward + self.gamma * q_next

        loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.eval_net.load_state_dict(self.eval_net_gpu.state_dict())
        self.writer.add_scalar('loss', loss, self.learn_step_counter)


    def save(self, filename):
        torch.save(self.eval_net.state_dict(), filename + "_Q_net" + str(datetime.datetime.now()))
        torch.save(self.optimizer.state_dict(), filename + "_optimizer_" + str(datetime.datetime.now()))

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename + "_Q_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
