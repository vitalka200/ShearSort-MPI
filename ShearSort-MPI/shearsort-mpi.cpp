#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "shearsort-mpi.h"

int main(int argc, char* argv[]) {

	int numberOfWorkers, currentId;
	double startTime, stopTime;
	int* matrix = new int[MATRIX_CELLS_COUNT] { 0 };
	int receivedNum;
	int dims[CART_DIM] = { MATRIX_DIM, MATRIX_DIM };
	int periods[CART_DIM] = { 0, 0 }; // we don't want ciclic topolgy
	int reorder = 0; // we don't want that MPI reorder nodes
	MPI_Comm calcComm;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentId);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfWorkers);

	if (numberOfWorkers != MATRIX_CELLS_COUNT)
	{
		printf("Program requires %d nodes\n", MATRIX_CELLS_COUNT);
		MPI_Finalize();
		exit(1);
	}

	if (currentId == MASTER_RANK)
	{
		// Prepare initial matrix
		GenerateMatrix(matrix);
		PrintInitial(matrix);
		startTime = MPI_Wtime(); // start time counter on master before sending numbers
	}

	// Distribute CELLS_PER_PROC==1 elements to each node
	MPI_Scatter(matrix, CELLS_PER_PROC, MPI_INT, &receivedNum, CELLS_PER_PROC, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
	
	// Create MPI Cartesian topology with 2 dimmentions
	MPI_Cart_create(MPI_COMM_WORLD, CART_DIM, dims, periods, reorder, &calcComm);

	// Sort matrix. Complexity O(2*log2(n)+1)
	receivedNum = ShearSort(receivedNum, calcComm);

	// Receive sorted data
	MPI_Gather(&receivedNum, CELLS_PER_PROC, MPI_INT, matrix, CELLS_PER_PROC, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

	if (currentId == MASTER_RANK)
	{
		PrintResult(matrix);
		stopTime = MPI_Wtime();
		printf("============ Result ============\n");
		printf("================================\n");
		printf("Number of elements n^2: %d\nExecution Time: %f\n", MATRIX_CELLS_COUNT, stopTime - startTime);
	}
	MPI_Finalize();

	delete[] matrix;
	return 0;
}

void GenerateMatrix(int* matrix)
{
	srand(time(NULL));

	for (int i = 0; i < MATRIX_CELLS_COUNT; i++)
		matrix[i] = rand() % MATRIX_CELLS_COUNT; 
}

int ShearSort(int receivedNum, MPI_Comm comm)
{
	int rank;
	int coord[2]; 
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, CART_DIM, coord);
	
	
	int totalIterations = (int)ceil(log2((double)MATRIX_DIM)) + 1;
	
	for (int i = 0; i <= totalIterations; i++)
	{
		// Rows pass
		receivedNum = OddEvenSort(coord, receivedNum, ROWS, comm);
		// Columns pass
		receivedNum = OddEvenSort(coord, receivedNum, COLLS, comm);
	}
	
	return receivedNum;
}

int OddEvenSort(int* coord, int storedNum, MatrixPassDirection passDirection, MPI_Comm comm)
{
	int neighbor1, neighbor2, neighborRankForExchange;
	CommDirection commDirection;
	SortDirection sortDirection = GetSortDirection(coord, passDirection);
	MPI_Cart_shift(comm, passDirection, NEIGHBOR_DISTANCE, &neighbor1, &neighbor2);
	
	for (int i = 0; i < MATRIX_DIM; i++)
	{
		commDirection = GetCommDirection(coord, i, passDirection);
		neighborRankForExchange = commDirection == SENDING ? neighbor2 : neighbor1;

		if (neighborRankForExchange != MPI_PROC_NULL) // Exchange only if we in bounds
			storedNum = ExchangeBetweenNeighbors(storedNum, commDirection, sortDirection, neighborRankForExchange, comm);
	}
	return storedNum;
}

int ExchangeBetweenNeighbors(int storedNum, CommDirection commDirection, SortDirection sortDirection, int neighborRank, MPI_Comm comm)
{
	MPI_Status status;
	int result = storedNum;
	if (commDirection == SENDING) // Sending side
	{
		MPI_Send(&storedNum, CELLS_PER_PROC, MPI_INT, neighborRank, 0, comm);
		MPI_Recv(&result, CELLS_PER_PROC, MPI_INT, neighborRank, 0, comm, &status);
	}
	else // Receiving side. Make the actual check/sort
	{
		int received;
		MPI_Recv(&received, CELLS_PER_PROC, MPI_INT, neighborRank, 0, comm, &status);

		if (isGreater(storedNum, received, sortDirection))
		{
			result = received;
			received = storedNum;
		}
		MPI_Send(&received, CELLS_PER_PROC, MPI_INT, neighborRank, 0, comm);
	}
	return result;
}

bool isGreater(int num1, int num2, SortDirection sortDirection)
{
	return sortDirection == ASCENDING ? num1 < num2 : num1 > num2;
}

CommDirection GetCommDirection(int* coord, int iteration, MatrixPassDirection direction)
{
	// if position even and iteration even we are sending, otherwise receiving
	// if position odd and iteration odd we are sending, otherwise receiving
	return (iteration % 2 == coord[direction] % 2) ? SENDING : RECEIVING;
}

SortDirection GetSortDirection(int* coord, MatrixPassDirection direction)
{
	// Even Row  ASCENGING
	// Odd Row   DESCENDING
	// Coll always ASCENGING;
	if (direction == COLLS) return ASCENDING;
	return (SortDirection)(coord[0] % 2);
}

void PrintInitial(int *matrix)
{
	printf("======== Initial Numbers =======\n");
	RegularPrint(matrix, true);
	printf("\n================================\n");
	RegularPrint(matrix, false);
	printf("\n================================\n");
	fflush(stdout);
}

void PrintResult(int* matrix)
{
	printf("======== Sorted Numbers ========\n");
	RegularPrint(matrix, false);
	printf("\n================================\n");
	ReversePrint(matrix);
	printf("\n================================\n");

}

void ReversePrint(int* matrix)
{
	printf("========== Flat View ===========\n");
    bool evenLinesSwapper = MATRIX_CELLS_COUNT % 2 == 0 ? true : false;
	for (int i = MATRIX_CELLS_COUNT - MATRIX_DIM; i >= 0; i -= MATRIX_DIM*2)
	{
        PrintLine(matrix + i, MATRIX_DIM, evenLinesSwapper);
        // take care of odd matrix dimentions
        if (i - MATRIX_DIM > 0) { PrintLine(matrix + i - MATRIX_DIM, MATRIX_DIM, !evenLinesSwapper); }
	}
}


void PrintLine(int* line, int count, bool isForward)
{
	
	if (isForward)
	{
		for (int i = 0; i < count; i++) { printf("%3d ", line[i]); }
	}
	else
	{
		for (int i = count - 1; i >= 0; i--) { printf("%3d ", line[i]); }
	}
}

void RegularPrint(int* matrix, bool isFlatView)
{
	if (isFlatView) { printf("========== Flat View ===========\n"); }
	else { printf("========= Matrix View ==========\n"); }
	for (int i = 0; i < MATRIX_CELLS_COUNT; i += MATRIX_DIM)
	{
		PrintLine(matrix + i, MATRIX_DIM, true);
		if (!isFlatView) { printf("\n"); }
	}
}