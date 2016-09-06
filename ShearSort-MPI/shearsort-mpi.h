#pragma once

const int MATRIX_DIM = 8;
const int MATRIX_CELLS_COUNT = MATRIX_DIM*MATRIX_DIM;
const int CELLS_PER_PROC = 1; // Each node take care of only one cell
const int NEIGHBOR_DISTANCE = 1;
const int CART_DIM = 2;
const int MASTER_RANK = 0;

enum MatrixPassDirection {
	COLLS = 0,
	ROWS  = 1,
};

enum CommDirection {
	RECEIVING = 0,
	SENDING = 1,
	NO_COMM = -1
};

enum SortDirection {
	ASCENDING = 0,
	DESCENDING = 1
};

void          GenerateMatrix(int* matrix);
int           ShearSort(int receivedNum, MPI_Comm comm);
int           OddEvenSort(int* coord, int storedNum, MatrixPassDirection direction, MPI_Comm comm);
int           ExchangeBetweenNeighbors(int storedNum, CommDirection commDirection, SortDirection sortDirection, int neighborRank, MPI_Comm comm);
bool          isGreater(int num1, int num2, SortDirection sortDirection);
CommDirection GetCommDirection(int* coord, int iteration, MatrixPassDirection direction);
SortDirection GetSortDirection(int* coord, MatrixPassDirection direction);
void          PrintInitial(int *matrix);
void          PrintResult(int* matrix);
void          PrintLine(int* line, int count, bool isForward);
void          RegularPrint(int* matrix, bool isFlatView);
void          ReversePrint(int* matrix);