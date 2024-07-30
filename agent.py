import numpy as np 
import math
import chess
import pytorch_lightning as pl
import torch
import net

from mask import Mask
import preprocessing

class Agent:
    def __init__(self, args):
        self.args = args 
        self.mask = Mask()
        self.memory = Memory()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.model = self.load_model().cuda()
        except:
            self.model = net.ResNet(128, 12, 1e-3).cuda()

        self.mcts = MCTS(self, self.args)

    def choose_move(self, board):
        best_move = self.mcts.search(board)
        return best_move
    
    def predict(self, board):
        bb = torch.tensor(self.bitboard(board), dtype=torch.float32).to(self.device).unsqueeze(0)
        
        value, policy = self.model.forward(bb)
       
        return value.squeeze(), policy.squeeze()
    

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
    
    
    
    def decodePolicyOutput(self, board, policy):
        """
        Decode the policy output from the neural network.

        Args:
            board (chess.Board) the board
            policy (numpy.array) the policy output

        """

        move_probabilities = np.zeros(200)
        count = 0
        for idx, move in enumerate(board.legal_moves):
            mirrored_move = move
            if not board.turn:
                mirrored_move = self.mirrorMove(move)
            planeIdx, rankIdx, fileIdx = self.mask.moveToIdx(mirrored_move)
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            move_probabilities[ idx ] = policy[moveIdx]

            count += 1
        return move_probabilities[:count]
    
    def load_model(self):
        '''Load a previously trained model'''
        checkpoint = torch.load('saved_models/model')
        state_dict = checkpoint['state_dict']

        model = net.ResNet(128, 12, 1e-3).cuda()
        model.load_state_dict(state_dict)

        return model
    

    def bitboard(self, board):
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
        
        


class Node:
    def __init__(self, board, parent = None, parent_action = None):
        self.board = board.copy()
        self.parent = parent 
        self.parent_action = parent_action

        self.n_visits = 0 
        self.value = 0
        self.policy = 0
        self.policy_masked = 0

        self.legal_moves = list(self.board.legal_moves)
        self.children = []
        
    def is_terminal(self):
        return self.board.is_game_over()
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def update(self):
        self.n_visits += 1

    def expand(self):
        if not self.is_terminal():
            for move in self.legal_moves:
                copy = self.board.copy()
                copy.push(move)
                self.children.append(Node(copy, parent=self, parent_action=move))

        else: 
            if self.board.is_checkmate():
                if self.board.turn == chess.WHITE:
                    self.value = 1
                




class MCTS:
    def __init__(self, agent, args):
        '''board is current game state, args is a dictionary with the C and max_search values'''
        self.C, self.max_search, self.training = args['C'], args['max_search'], args['training']
        self.agent = agent

    def rootSearch(self, board):
        for child in self.root.children:
            if child.board == board:
                self.root = child
                return
        self.createRoot(board)
    
    def createRoot(self, board):
        self.root = Node(board)
        root_val, root_pol = self.agent.predict(self.root.board)
        root_masked_pol = self.agent.decodePolicyOutput(self.root.board, root_pol)
        self.root.value = root_val
        self.root.policy_masked = root_masked_pol
        self.root.policy = root_pol


    def search(self, board):
        try:
            self.root
            self.rootSearch(board)

        except:
            #Define first node
            self.createRoot(board)

        root = self.root

        #Run MCTS!
        for _ in range(self.max_search):

            #Selection 
            node = self.select(root)

            #Expansion 
            node.expand()
            for child in node.children:
                child_board = child.board

                #Use the net to find policy and node values
                value, policy_raw = self.agent.predict(child_board)
                policy_masked = self.agent.decodePolicyOutput(child_board, policy_raw)
                
                #update node values and polciies 
                child.value = value 
                child.policy_masked = policy_masked
                child.policy = policy_raw

            #Back prop 
            self.backpropagate(node)

        self.agent.memory.store_move(root)
        selected_move = self.move(root)
        #Search children to update root 
        for child in root.children:
            if child.parent_action == selected_move:
                self.root = child
        return selected_move

    def ucb(self, child):
        raw_pol = torch.tensor(child.parent.policy_masked)
        
        clean_pol = torch.softmax(raw_pol, dim=-1)
        
        move_list = child.parent.legal_moves 
        child_idx = move_list.index(child.parent_action)

        prob = clean_pol[child_idx]
        return child.value + self.C * prob * (math.sqrt(child.parent.n_visits) / (child.n_visits+1))
    
    def backpropagate(self, node):
        while node != None:
            node.update()
            node = node.parent

    def select(self, root):
        node = root
        while not node.is_leaf():

            best_val = -np.inf 
            best_child = None
            for child in node.children:
                value = self.ucb(child)

                if value > best_val:
                    best_child = child 
                    best_val = value

            node = best_child
        return node



    def move(self, root):
        if not self.training:
            best_val = -np.inf 
            best_child = None
            for child in root.children:
                value = self.ucb(child)

                if value > best_val:
                    best_child = child 
                    best_val = value
            return best_child.parent_action
        
        policy = torch.tensor(root.policy_masked)

        legal_moves = list(root.board.legal_moves)

        probs = torch.softmax(policy, dim=-1)
        
        return np.random.choice(legal_moves, p = probs.cpu().detach().numpy())


class Memory:
    def __init__(self):
        self.mask = Mask()
        self.raw_boards = []
        self.target_values = []
        self.target_policies = []
        self.bitboards = []

    def clear(self):
        self.raw_boards = []
        self.target_values = []
        self.target_policies = []
        self.bitboards = []


    def store_move(self, root):
        board = root.board
    
        target_policy = np.array([child.n_visits/(root.n_visits-1) for child in root.children])
        target_policy = self.encodeLegalMovePolicy(root.board, target_policy)
        bb = preprocessing.ResNet_bitboard(board)

        self.target_policies.append(target_policy)
        self.raw_boards.append(board)
        self.bitboards.append(bb)
        
    def store_result(self, value):
        for board in self.raw_boards:

            if board.turn == chess.WHITE:
                self.target_values.append(value)
            
            else:
                self.target_values.append(-value)


    def pop(self):
        return self.bitboards, self.target_values, self.target_policies
    

    def encodeLegalMovePolicy(self, board, policy):
        """
        Decode the policy output from the neural network.

        Args:
            board (chess.Board) the board
            policy (numpy.array) the policy output

        """

        move_probabilities = np.zeros(4608)

        for idx, move in enumerate(board.legal_moves):
            if not board.turn:
                move = self.mirrorMove(move)
            planeIdx, rankIdx, fileIdx = self.mask.moveToIdx(move)
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            move_probabilities[ moveIdx ] = policy[idx]
            
        return move_probabilities
    
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
    

