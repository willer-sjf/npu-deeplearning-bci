import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reward_shape(origin_reward, discount):
    length = len(origin_reward)
    new_reward = np.zeros_like(origin_reward, dtype=np.float32)
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
    def __init__(
        self,
        data,
        label,
        gamma=0.97,
        epsilon=0.3,
        max_drop=2,
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.max_drop = max_drop
        self.env = DropEnv(data, label)

        self.eval_net = Q_Net(data.shape[-1])
        #self.writer = SummaryWriter()

        self.interaction_counter = 0
        self.learn_step_counter = 0
        self.step_update = 10

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.97)

    def select_action(self, state):
        action = -1
        q_max = None
        action_dim = len(state) - 1
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
                    print('done')
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

    def save(self, filename):
        torch.save(self.eval_net.state_dict(), filename + "_Q_net" + str(datetime.datetime.now()))
        torch.save(self.optimizer.state_dict(), filename + "_optimizer_" + str(datetime.datetime.now()))

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename + "_Q_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
