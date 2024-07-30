import chess 
import numpy as np 


class Mask:

    def moveToIdx(self, move):
        """
        Maps a legal move to an index in (72, 8, 8)
        Each of the 72 planes represents a different direction
        and distance: rook and bishop directions with distance (64 planes)
        and 8 horse directions.
        The location in the plane specifies the start square.
        
        Args:
            move (chess.Move) the move to be encoded.

        Returns:
            (int) the plane the move maps to
            from_rank (int) the moves starting rank
            from_file (int) the moves starting file
        """

        from_rank = chess.square_rank( move.from_square )
        from_file = chess.square_file( move.from_square )
        
        to_rank = chess.square_rank( move.to_square )
        to_file = chess.square_file( move.to_square )

        if from_rank == to_rank and from_file < to_file:
            directionPlane = 0
            distance = to_file - from_file
            directionAndDistancePlane = directionPlane + distance
        elif from_rank == to_rank and from_file > to_file:
            directionPlane = 8
            distance = from_file - to_file
            directionAndDistancePlane = directionPlane + distance
        elif from_file == to_file and from_rank < to_rank:
            directionPlane = 16
            distance = to_rank - from_rank
            directionAndDistancePlane = directionPlane + distance
        elif from_file == to_file and from_rank > to_rank:
            directionPlane = 24
            distance = from_rank - to_rank
            directionAndDistancePlane = directionPlane + distance
        elif to_file - from_file == to_rank - from_rank and to_file - from_file > 0:
            directionPlane = 32
            distance = to_rank - from_rank
            directionAndDistancePlane = directionPlane + distance
        elif to_file - from_file == to_rank - from_rank and to_file - from_file < 0:
            directionPlane = 40
            distance = from_rank - to_rank
            directionAndDistancePlane = directionPlane + distance
        elif to_file - from_file == -(to_rank - from_rank) and to_file - from_file > 0:
            directionPlane = 48
            distance = to_file - from_file
            directionAndDistancePlane = directionPlane + distance
        elif to_file - from_file == -(to_rank - from_rank) and to_file - from_file < 0:
            directionPlane = 56
            distance = from_file - to_file
            directionAndDistancePlane = directionPlane + distance
        elif to_file - from_file == 1 and to_rank - from_rank == 2:
            directionAndDistancePlane = 64
        elif to_file - from_file == 2 and to_rank - from_rank == 1:
            directionAndDistancePlane = 65
        elif to_file - from_file == 2 and to_rank - from_rank == -1:
            directionAndDistancePlane = 66
        elif to_file - from_file == 1 and to_rank - from_rank == -2:
            directionAndDistancePlane = 67
        elif to_file - from_file == -1 and to_rank - from_rank == 2:
            directionAndDistancePlane = 68
        elif to_file - from_file == -2 and to_rank - from_rank == 1:
            directionAndDistancePlane = 69
        elif to_file - from_file == -2 and to_rank - from_rank == -1:
            directionAndDistancePlane = 70
        elif to_file - from_file == -1 and to_rank - from_rank == -2:
            directionAndDistancePlane = 71

        return directionAndDistancePlane, from_rank, from_file
    
    def legalMoveMask(self, board):
        mask = np.zeros((72, 8, 8), dtype=np.int32)

        for move in board.legal_moves:
            planeIdx, rankIdx, fileIdx = self.moveToIdx(move)
            mask[planeIdx, rankIdx, fileIdx] = 1



