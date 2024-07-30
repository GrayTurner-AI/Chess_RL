import chess
import agent
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, lr_finder
from pytorch_lightning.loggers import TensorBoardLogger


class Game:
    def __init__(self):
        self.args = {'C': 1.41, 'max_search':5, 'training':False}
        self.agent = agent.Agent(self.args)
        
    def play(self):

        board = chess.Board()

        while True: 
            
            while board.turn == chess.WHITE:
                print(board)
                print()

                try:
                    move = str(input('move: '))
                    if move == 'q':
                        return
                    board.push_san(move)
                except:
                    print('invalid move')
            
            if board.is_game_over():
                print('Gameover')
                break

            move = self.agent.choose_move(board)
            board.push(move)

            if board.is_game_over():
                print('Gameover')
                break



class selfPlay:
    def __init__(self):
        self.args = {'C': 1.41, 'max_search':100, 'training':True}
        self.agent = agent.Agent(self.args)
        self.simulations_per_update = 500
        self.batch_size = 256
        self.epochs = 100
        self.upgrade_iterations = 10

        self.train_session()
 

    def runGame(self):

        board = chess.Board()
        while True: 
            white_move = self.agent.choose_move(board)
            board.push(white_move)
        
            if board.is_game_over():
                if board.outcome().winner == chess.WHITE:
                    return 1
                else:
                    return 0

            black_move = self.agent.choose_move(board)
            board.push(black_move)

            if board.is_game_over():
                if board.outcome().winner == chess.BLACK:
                    return -1
                else:
                    return 0

                
    def dataGeneration(self):
        self.target_values =[]
        self.target_policies = []
        self.bitboards = []

        for _ in range(self.simulations_per_update):
            result = self.runGame()
            self.agent.memory.store_result(result)
            bitboards, t_vals, t_pols = self.agent.memory.pop()

            self.bitboards.extend(bitboards)
            self.target_values.extend(t_vals)
            self.target_policies.extend(t_pols)
            self.agent.memory.clear()

            print('Game gen {} of {}'.format(_+1, self.simulations_per_update))
        xtrain, xtest, valtrain, valtest, poltrain, poltest = train_test_split(np.array(self.bitboards), np.array(self.target_values), np.array(self.target_policies), test_size=0.2)

        return xtrain, xtest, valtrain, valtest, poltrain, poltest

    def train_once(self, data):
        training = (data[0], data[2], data[4])
        validation = (data[1], data[3], data[5])

        trainset = self.Training_Data(training)
        valset = self.Training_Data(validation)

        trainloader = DataLoader(trainset,
                          batch_size=self.batch_size,
                          shuffle=True)
        
        valloader = DataLoader(valset,
                          batch_size=self.batch_size,
                          shuffle=False)
        
        torch.set_float32_matmul_precision('medium')

        logger = TensorBoardLogger(save_dir='logs/model',
                            name='logger')
        
        trainer = pl.Trainer(max_epochs=self.epochs,
                      callbacks=[EarlyStopping(monitor='validation_loss',
                                                check_on_train_epoch_end=False),
                                 lr_finder.LearningRateFinder()], 
                      logger = logger)
        
        trainer.fit(self.agent.model, trainloader, valloader)


    def train_session(self):
        for _ in range(self.upgrade_iterations):
            self.agent.model.cuda()
            print('Model{}'.format(_+1))

            data = self.dataGeneration()
            self.train_once(data)
            self.agent.memory.clear()

        torch.save(self.agent.model, 'saved_models/FullyTrained')


    class Training_Data(Dataset):
        def __init__(self, data):
            self.bitboards, self.target_values, self.target_policies = data[0], data[1], data[2]

        def __len__(self):
            return len(self.target_values)
        
        def __getitem__(self, idx):
            return torch.tensor(self.bitboards[idx], dtype = torch.float32).to('cuda'), torch.tensor(self.target_values[idx], dtype=torch.float32).to('cuda'), torch.tensor(self.target_policies[idx], dtype=torch.float32).to('cuda')
    


if __name__ == '__main__':
    #selfPlay()
    Game().play()
