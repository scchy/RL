# python3
# AUthor: Scc_hy
# Create Date: 2025-03-26
# Func: random walk sample 
# ======================================================================
from copy import deepcopy
import numpy as np 
import gymnasium as gym 
from tqdm.auto import tqdm 
import pandas as pd 
import matplotlib.pyplot as plt 


class SimpleRandomWalk(gym.Env):
    def __init__(self):
        super(SimpleRandomWalk, self).__init__()
        self.states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.action_space = gym.spaces.Discrete(2, start=0)
        self.observation_space = gym.spaces.Discrete(len(self.states))
        self.start_state = 'D'
        self.current_state = self.start_state
        self.rewards = {'A': 0, 'G': 1}
        
    def step(self, action):
        index = self.states.index(self.current_state)
        if action == 0:  # 左移
            if index > 0:
                self.current_state = self.states[index - 1]
        elif action == 1:  # 右移
            if index < len(self.states) - 1:
                self.current_state = self.states[index + 1]
        
        reward = self.rewards.get(self.current_state, 0)
        done = self.current_state in ['A', 'G']
        return self._state_to_index(self.current_state), reward, done, False, {}
    
    def reset(self, **kwargs):
        self.current_state = self.start_state
        return self._state_to_index(self.current_state), {}

    def render(self, mode='human'):
        print(f"Current state: {self.current_state}")

    def _state_to_index(self, state):
        return self.states.index(state)


def generate_samples(n=100, seq_len=10, print_flag=False):
    # 生成n个训练集，长度为10 
    env = SimpleRandomWalk()
    need_n = 0
    need_train = []
    need_r = []    
    for i in range(10000):
        s, _ = env.reset()
        route = [s]
        r_t = 0
        done = False
        while not done:
            a = env.action_space.sample()
            n_s, r, d, _, _ = env.step(a)
            route.append(n_s)
            r_t += r
            s = n_s
            done = d
        
        if len(route) == seq_len: # 最后一个不算
            need_n += 1
            need_train.append(route)
            need_r.append(r_t)
        if print_flag:
            print(need_n, len(route))
        if need_n >= n:
            break
    return need_train, need_r


def route2arr(route_, time_step=9):
    a = np.zeros((time_step, 7))
    a[np.arange(time_step), np.array(route_)] = 1
    return a


class multiReg:
    def __init__(self, feature_dim=7, learning_rate=0.1):
        self.w = np.random.randn(feature_dim).reshape(-1, 1)
        self.learning_rate = learning_rate

    def predict(self, x):
        return np.matmul(x, self.w)
    
    def fit(self, x, y, batch_size=25, n_rounds=100):
        n = x.shape[0]
        idx = np.arange(n)
        tq_bar =  tqdm(range(n_rounds))
        for round_i in tq_bar:
            tq_bar.set_description(f'[  {round_i + 1} / {n_rounds}]')
            np.random.shuffle(idx) 
            idx_l = [min(i, n) for i in range(0, n + batch_size, batch_size)]
            for st, ed in zip(idx_l[:-1], idx_l[1:]):
                batch_x = x[st:ed, ...]
                batch_y = y[st:ed, ...]
                # x @ w
                p_t = self.predict(batch_x) 
                # (p_t - z)
                e = p_t.reshape(batch_x.shape[0], -1) - batch_y
                # \sum_{t=1}^T (p_t - z) @ x
                grad = np.matmul(batch_x.transpose(0, 2, 1), e[..., np.newaxis]).mean(axis=0)
                self.w -= self.learning_rate *  grad


class multiTDReg:
    def __init__(self, feature_dim=7, learning_rate=0.1, time_step=9, lmbda=0.9, incre_update=True):
        self.w = np.random.randn(feature_dim).reshape(-1, 1)
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.time_step = time_step
        self.incre_update = incre_update

    def predict(self, x):
        return np.matmul(x, self.w)

    def get_nabla(self, batch_x):
        lmbda = self.lmbda
        e_t_list = [0]
        for t in range(self.time_step):
            e_t_1 = batch_x[:, t, :] + e_t_list[-1] * lmbda
            e_t_list.append(e_t_1)
        return np.stack(e_t_list[1:], axis=1)
    
    def fit(self, x, y, batch_size=25, n_rounds=100):
        n = x.shape[0]
        idx = np.arange(n)
        tq_bar =  tqdm(range(n_rounds))
        for round_i in tq_bar:
            tq_bar.set_description(f'[  {round_i + 1} / {n_rounds}]')
            np.random.shuffle(idx) 
            idx_l = [min(i, n) for i in range(0, n + batch_size, batch_size)]
            for st, ed in zip(idx_l[:-1], idx_l[1:]):
                batch_x = x[st:ed, ...]
                batch_y = y[st:ed, ...]
                p_t = self.predict(batch_x) 
                p_t_c = np.concatenate([p_t, batch_y[..., np.newaxis]], axis=1)
                e =  p_t_c[:, :-1, :] - p_t_c[:, 1:, :] #  -(p_{t+1} -p_t)  # b, time_step, 1
                # e \sum_{k=1}^t lmbda^{t-k} \nabla_w P_k 
                nabla = self.get_nabla(batch_x) # b, time_step, 7
                # print(f'{nabla.shape=} {e.shape=}')
                if not self.incre_update:
                    grad = np.matmul(nabla.transpose(0, 2, 1), e).mean(axis=0)
                    self.w -= self.learning_rate *  grad
                    continue

                # incrementally update:
                for t in range(self.time_step):
                    grad_t = nabla[:, t, :] * e[:, t, :]
                    self.w -= self.learning_rate *  grad_t.mean(axis=0).reshape(-1, 1)


def generate_arr(n=100, seq_len=10):
    tr, r = generate_samples(n, seq_len=seq_len)
    tr_x = np.stack([route2arr(x_i[:-1], time_step=seq_len-1) for x_i in tr])
    tr_y = np.array(r).reshape(-1, 1)
    return tr_x, tr_y


np.random.seed(202503)
batch_size=10
train_seq_pairs = [
    (10, 4), (90, 6), (150, 8), (200, 10), 
    (150, 12), (100, 14), (100, 16), (100, 18), (100, 20),
]
tr_x_arr_list = []
tr_y_arr_list = []
for (n_seq, seq_l) in train_seq_pairs:
    tr_x_i, tr_y_i = generate_arr(n=n_seq, seq_len=seq_l)
    tr_x_arr_list.append(tr_x_i)
    tr_y_arr_list.append(tr_y_i)

te_x, te_y = generate_arr(n=40, seq_len=10)
te_x1, te_y1 = generate_arr(n=40, seq_len=18)
# -----------------------------------------
# find best alpha with lambda=0.9
n_rounds_list = [1, 1, 5, 5, 7, 7, 10, 10, 15, 15]
test_res = []
for alpha in np.linspace(0, 0.35, 20):
    for n_rounds_i in n_rounds_list:
        m_reg = multiReg(7, alpha)
        m_td_reg = multiTDReg(7, alpha, time_step=9, lmbda=0.9)
        for r_ in range(n_rounds_i):
            for tr_x, tr_y in zip(tr_x_arr_list, tr_y_arr_list):
                m_td_reg.time_step = tr_x.shape[1] 
                m_reg.fit(tr_x, tr_y, batch_size=batch_size, n_rounds=1)
                m_td_reg.fit(tr_x, tr_y, batch_size=batch_size, n_rounds=1)
        mse_te = np.sqrt(np.mean(
            (m_reg.predict(te_x).mean(axis=1) - te_y)**2 / 2 + 
            (m_reg.predict(te_x1).mean(axis=1) - te_y1)**2 / 2) 
        )
        td_mse_te = np.sqrt(np.mean(
            (m_td_reg.predict(te_x).mean(axis=1) - te_y)**2 / 2 + 
            (m_td_reg.predict(te_x1).mean(axis=1) - te_y1)**2 / 2) 
        )
        # print(f'[{test_i=}]:\nmultiReg Train mse={mse_tr:.3f} Test mse={mse_te:.3f}\nmultiTDReg Train mse={td_mse_tr:.3f} Test mse={td_mse_te:.3f}')
        test_res.append([alpha, mse_te, td_mse_te])

res_df = pd.DataFrame(np.array(test_res), columns=['alpha', 'mse_te', 'td_mse_te' ])
res_ = res_df.groupby('alpha').mean()
print('res_:\n', res_)
plt.plot(range(res_.shape[0]-1), res_.mse_te.values[1:], label='mse_te')
plt.plot(range(res_.shape[0]-1), res_.td_mse_te.values[1:], label='td_mse_te')
plt.xticks(range(res_.shape[0]-1), np.round( res_.index.values[1:], 3))
plt.xlabel("$alpha$")
plt.ylabel('RMSE')
plt.ylim([0.38, min(0.65, res_.mse_te.values[1:].max())])
plt.title(f"RMSE: Two Method of Multi-Step Learning Result in Different $alpha$\n1000 samples (batch_size=10)\nEach $alpha$ run 10 times with different epochs({n_rounds_list})")
plt.legend()
plt.show()

test_res2 = []
n_rounds_list2 = [200, 200, 50, 50, 70, 70, 100, 100, 150, 150]
for alpha in np.linspace(0, 0.5, 20):
    for n_i in n_rounds_list2:
        m_reg2 = multiReg(7, alpha)
        m_td_reg2 = multiTDReg(7, alpha, time_step=9, lmbda=0.9)
        for r_ in range(n_rounds_i):
            for tr_x, tr_y in zip(tr_x_arr_list, tr_y_arr_list):
                m_td_reg2.time_step = tr_x.shape[1] 
                m_reg2.fit(tr_x, tr_y, batch_size=batch_size, n_rounds=1)
                m_td_reg2.fit(tr_x, tr_y, batch_size=batch_size, n_rounds=1)

        mse_te2 = np.sqrt(np.mean(
            (m_reg2.predict(te_x).mean(axis=1) - te_y)**2 / 2 + 
            (m_reg2.predict(te_x1).mean(axis=1) - te_y1)**2 / 2) 
        )
        td_mse_te2 = np.sqrt(np.mean(
            (m_td_reg2.predict(te_x).mean(axis=1) - te_y)**2 / 2 + 
            (m_td_reg2.predict(te_x1).mean(axis=1) - te_y1)**2 / 2) 
        )
        # print(f'[{test_i=}]:\nmultiReg Train mse={mse_tr:.3f} Test mse={mse_te:.3f}\nmultiTDReg Train mse={td_mse_tr:.3f} Test mse={td_mse_te:.3f}')
        test_res2.append([alpha, n_i, mse_te2, td_mse_te2])


res_df = pd.DataFrame(np.array(test_res2), columns=['alpha', 'n_i', 'mse_te', 'td_mse_te' ])
res_ = res_df.groupby('alpha').mean()
print('res_:\n', res_)
plt.plot(range(res_.shape[0]-1), res_.mse_te.values[1:], label='mse_te')
plt.plot(range(res_.shape[0]-1), res_.td_mse_te.values[1:], label='td_mse_te')
plt.xticks(range(res_.shape[0]-1), np.round( res_.index.values[1:], 3))
plt.xlabel("$alpha$")
plt.ylabel('RMSE')
plt.ylim([0.38, min(0.65, res_.mse_te.values[1:].max())])
plt.title(f"RMSE: Two Method of Multi-Step Learning Result in Different $alpha$\n1000 samples (batch_size=10)\nEach $alpha$ run 10 times with different epochs({n_rounds_list2})")
plt.legend()
plt.show()


# -----------------------------------------
# find best lambda
test_lambda_res = []
for alpha in np.linspace(0, 0.6, 10):
    for lmbda in [0.1,  0.5,  0.9, 1.0]:
        m_td_reg = multiTDReg(7, alpha, time_step=9, lmbda=lmbda)
        for r_ in range(1):
            for tr_x, tr_y in zip(tr_x_arr_list, tr_y_arr_list):
                m_td_reg.time_step = tr_x.shape[1] 
                m_td_reg.fit(tr_x, tr_y, batch_size=batch_size, n_rounds=1)
        tr_n = 0
        se_sum = 0
        for tr_x, tr_y in zip(tr_x_arr_list, tr_y_arr_list):
            tr_n += tr_x.shape[0]
            se_sum += np.sum((m_td_reg.predict(tr_x).mean(axis=1) - tr_y)**2)

        td_mse_te = np.sqrt(se_sum/tr_n)
        test_lambda_res.append([alpha, lmbda, td_mse_te])



res_lambda_df = pd.DataFrame(np.array(test_lambda_res), columns=['alpha', 'lmbda', 'td_mse_te'])
res_ = res_lambda_df.groupby(['alpha', 'lmbda']).mean().reset_index()
for lb, tmp_df in res_.groupby('lmbda'):
    plt.plot(range(tmp_df.shape[0]), tmp_df.td_mse_te.values, marker='o', label=f'$\lambda$={lb:.2f}')
    plt.xticks(range(tmp_df.shape[0]), np.round(tmp_df.alpha.values, 3))
    plt.xlabel("$alpha$")
    plt.ylabel('RMSE')
    plt.title(f"RMSE: TDReg Method of Multi-Step Learning Result in Different $lmbda$\n1000 samples (batch_size=10)")
    plt.legend()

plt.ylim([0.4, min(1, res_.td_mse_te.values[1:].max())])
plt.show()




 
 
 
