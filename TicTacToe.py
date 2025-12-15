import numpy as np
import pickle
BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1  

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])

        if abs(diag_sum1) == 3:
            self.isEnd = True
            return 1 if diag_sum1 == 3 else -1

        if abs(diag_sum2) == 3:
            self.isEnd = True
            return 1 if diag_sum2 == 3 else -1

        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print(f"Rounds {i}")

            while not self.isEnd:
               
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                self.p1.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                self.p2.addState(self.getHash())
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

    def play2(self):
        while not self.isEnd:
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                print("Computer wins!" if win == 1 else "Tie!")
                self.reset()
                break
            positions = self.availablePositions()
            p2_action = self.p2.chooseAction(positions)
            self.updateState(p2_action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                print("Human wins!" if win == -1 else "Tie!")
                self.reset()
                break
    def showBoard(self):
        for i in range(BOARD_ROWS):
            print("-------------")
            out = "| "
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + " | "
            print(out)
        print("-------------")

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(BOARD_COLS * BOARD_ROWS))

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            return positions[np.random.choice(len(positions))]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                value = self.states_value.get(self.getHash(next_board), 0)
                if value >= value_max:
                    value_max = value
                    action = p
            return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            self.states_value.setdefault(st, 0)
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open('policy_' + self.name, 'wb') as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, file):
        with open(file, 'rb') as f:
            self.states_value = pickle.load(f)

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Row (0-2): "))
            col = int(input("Col (0-2): "))
            if (row, col) in positions:
                return (row, col)

if __name__ == "__main__":
    p1 = Player("p1")
    p2 = Player("p2")
    st = State(p1, p2)
    print("Training...")
    st.play(5000)
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")
    p2 = HumanPlayer("human")
    st = State(p1, p2)
    c = 'y'
    while c == 'y':
        st.play2()
        c = input("Continue? (y/n): ")
