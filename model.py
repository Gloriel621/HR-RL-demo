import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Hyperparameters
max_episodes = 10000
mini_batch_size = 50

class PPO(nn.Module):
    def __init__(self, num_employees, num_branches):
        super(PPO, self).__init__()
        self._init_hyperparameters()
        self.data = []
        self.num_employees = num_employees
        self.num_branches = num_branches
             
        self.fc1 = nn.Linear(self.num_employees * self.num_branches, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc_pi = nn.Linear(256, self.num_employees)
        self.fc_pi2 = nn.Linear(256, self.num_branches)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, softmax_dim = 1):
        
        x = x.reshape(-1, self.num_employees * self.num_branches)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def pi2(self, x, softmax_dim = 1):
        
        x = x.reshape(-1, self.num_employees * self.num_branches)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc_pi2(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = x.reshape(-1, self.num_employees * self.num_branches)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, prob_a_lst2, done_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, prob_a2, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            prob_a_lst2.append([prob_a2])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
          
        s, a, r, s_prime, done_mask, prob_a, prob_a2 = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                        torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                        torch.tensor(prob_a_lst2)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, prob_a2
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, prob_a2 = self.make_batch()

        for i in range(self.K_epoch):
            
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            ratio2 = torch.exp(torch.log(pi_a) - torch.log(prob_a2))
                        
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(td_target.detach(), self.v(s))

            surr3 = ratio2 * advantage
            surr4 = torch.clamp(ratio2, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss2 = -torch.min(surr3, surr4) + F.smooth_l1_loss(td_target.detach(), self.v(s))

            
            loss = (loss + loss2) / 2
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def _init_hyperparameters(self):
        self.gamma = 0.98
        self.lmbda = 0.98
        self.K_epoch = 1
        self.eps_clip = 0.2
        self.learning_rate = 0.0002

