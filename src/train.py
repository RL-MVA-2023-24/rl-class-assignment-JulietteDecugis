from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self) -> None:
        S, A, R, S2, D = collect_samples()
        self.trainX = [S, A, R, S2, D]
        self.nb_samples = S.shape[0]
        self.nb_actions = 4
        self.Q = RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features = 8)
        self.Q.fit(np.append(S,A, axis = 1), R)
        self.gamma = 0.9
        self.path = None

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(range(4), 1)
        else:
            Qsa = []
            for a in range(self.nb_actions):
                sa = np.append(observation, a).reshape(1, -1)
                Qsa.append(self.Q.predict(sa))
            return np.argmax(Qsa)
    
    def rf_fqi(self, S, A, R, S2, D):
        n = 200
        SA = np.append(S, A, axis = 1)
        for iter in range(n):
            Q2 = np.zeros((self.nb_samples,self.nb_actions))
            for a in range(self.nb_actions):
                A2 = a * np.ones((S.shape[0],1))
                S2A2 = np.append(S2, A2, axis = 1)
                Q2[:, a] = self.Q.predict(S2A2)
            maxQ2 = np.max(Q2, axis=1)
            value = R + self.gamma*(1-D)*maxQ2
            Q = RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features = 8)
            Q.fit(SA, value)
            self.Q = Q

    def save(self, path):
        joblib.dump(self.Q, path)
        self.path = path

    def load(self):
        if self.path:
            self.Q = joblib.load(self.path)
        else:
            S, A, R, S2, D = self.trainX
            self.rf_fqi(S, A, R, S2, D)

def collect_samples(horizon = 200):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in range(horizon):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
        else:
            s = s2

    # conver to arrays
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D