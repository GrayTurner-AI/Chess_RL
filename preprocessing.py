import chess.pgn 
import numpy as np
import torch
from torch.utils.data import Dataset
from mask import Mask 

class Data:
    def __init__(self, datapoints, bitboard):
        self.datapoints = datapoints
        self.bitboard = bitboard
        self.labeler = {'1/2-1/2':0, 
                        '1-0':1, 
                        '0-1':-1}
        self.mask = Mask()

        self.process()


    def load_data(self):
        data = open('dataset\lichess_db_standard_rated_2024-06.pgn')
        return data 
    
    def process(self):
        values = []
        boards = []
        
        data = self.load_data()
        n_games = 0
        for _ in range(self.datapoints):
            game = chess.pgn.read_game(data)
            board = chess.Board()

            if len([move for move in game.mainline_moves()]) == 0:
                continue

            n_games += 1
            if game.headers['Result'] not in self.labeler.keys():
                continue

            label = self.labeler[game.headers['Result']]

            for move in game.mainline_moves():
                board.push(move)

                if board.turn == chess.BLACK:
                    copy = board.mirror()
                    copy_label = -label
                    
                    boards.append(self.bitboard(copy))
                    values.append(copy_label)

                else: 

                    boards.append(self.bitboard(board))
                    values.append(label)

        X = torch.tensor(np.array(boards, dtype=np.float32))
        vals = torch.tensor(np.array(values, np.float32).reshape(-1,1))
        

        np.savez('dataset/'+str(n_games)+'games', x=X, vals=vals, policies=None)



    def mirrorMove(self, move):
        """
        Mirrors a move vertically.

        Args:
            move (chess.Move) the move to be flipped

        Returns:
            (chess.Move) the mirrored move
        """

        from_square = move.from_square
        to_square = move.to_square

        new_from_square = chess.square_mirror( from_square )
        
        new_to_square = chess.square_mirror( to_square )

        return chess.Move( new_from_square, new_to_square )
    
    



def ResNet_bitboard(board):
        #This is the 3d board 
        bitboard = np.zeros((16, 8, 8))
        
        #Here we add the pieces view on matrix 
        for piece in chess.PIECE_TYPES:
            #White pieces first 
            for square in board.pieces(piece, chess.WHITE):
                idx = np.unravel_index(square, (8,8))
                bitboard[piece-1][7-idx[0]][idx[1]] = 1
                

            #Black pieces
            for square in board.pieces(piece, chess.BLACK):
                idx = np.unravel_index(square, (8,8))
                bitboard[piece+5][7-idx[0]][idx[1]] = 1

        bitboard[12] = np.ones((8,8)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((8,8))
        bitboard[13] = np.ones((8,8)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((8,8))

        bitboard[14] = np.ones((8,8)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((8,8))
        bitboard[15] = np.ones((8,8)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((8,8))

        #16, 8, 8
        return bitboard


class Training_Data(Dataset):
    def __init__(self, file):
        self.data = np.load(file, allow_pickle=True)

        self.dataset, self.values, self.policies = self.data['x'], self.data['vals'], self.data['policies']
            
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        if self.policies == None:
            return self.dataset[idx], self.values[idx], []
        
        return self.dataset[idx], self.values[idx], self.policies[idx]
    


#Data(10000, ResNet_bitboard)