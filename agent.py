import random
from collections import deque

from keras import Sequential
from keras.layers import Dense


class DeepAgent:
    def __init__(self):
        self.model = self.construct_model()
        self.memory = deque(maxlen=100_000)
        self.epsilon = 1
        self.epsilon_decay_high = 0.99995
        self.epsilon_decay_low = 0.99999
        self.lr = 0.3
        self.discount_factor = 0.95

    @staticmethod
    def construct_model():
        model = Sequential()
        model.add(Dense(16, activation='relu', input_shape=(4,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def learn(self, train_data, verbose=0):
        for state, action, next_state, reward, done in train_data:
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            q = self.model.predict(state)[0]
            if done:
                q[action] = reward
            else:
                q[action] = (1 - self.lr) * q[action] + \
                            self.lr * (reward + self.discount_factor * max(self.model.predict(next_state)[0]))

            self.model.fit(state, q.reshape(1, -1), verbose=verbose)

    def learn_from_memory(self, verbose=0, k=100):
        if k == 0:
            train_data = self.memory
        else:
            train_data = random.choices(self.memory, k=100)
        self.learn(train_data, verbose=verbose)

    def act(self, state):
        if random.random() > self.epsilon:
            action = self.model.predict(state.reshape(1, -1)).argmax()
        else:
            action = random.randint(0, 1)

        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay_high
        else:
            self.epsilon *= self.epsilon_decay_low

        return action
