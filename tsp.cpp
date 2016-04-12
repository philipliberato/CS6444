// prod-cons.c

// Code skeleton orignally from:
// http://www.hpc.cam.ac.uk/using-clusters/compiling-and-development/parallel-programming-mpi-example

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <climits>
#include <stack>
using namespace std;

#define WORKTAG    1
#define DIETAG     2
#define NUM_CITIES 5		 	// change depending on city file
char filename[] = "city5.txt";	// change depending on city file

void master();
void worker();
void inCities();
void printCities();

int myrank, ntasks, best;
double elapsed_time;
int cities[NUM_CITIES][NUM_CITIES];

stack<string> mystack; //NEED TO SYNCHRONIZE


int main(int argc, char *argv[]) {
	// Initialize
	MPI_Init(&argc, &argv);
	// Make sure everyone got here
	MPI_Barrier(MPI_COMM_WORLD);

	// Get current time
	elapsed_time = - MPI_Wtime();
	// Get myrank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// Get total number of tasks
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	// Load city data
	inCities();

	// Set best to 'infinity'
	best = INT_MAX;

	// Determine if master or worker
	if (myrank == 0) {
		master();
	} else {
		worker();
	}

	// Cleanup
	MPI_Finalize();
	return 0;
}

//*****************************************//
//    Stuff for master (rank 0) to do      //
//*****************************************//
void master() {
	int rank, work, result;
	MPI_Status status;

//-----------Set up the Queue-----------//
	// Probably don't need to synchronize initially lol
	for (int i = 2; i <= NUM_CITIES; ++i) {
		string result;          // string which will contain the result
		ostringstream convert;   // stream used for the conversion
		convert << i;      // insert the textual representation of 'Number' in the characters in the stream
		result = convert.str(); // set 'Result' to the contents of the stream

		mystack.push("1"+result);
	}

	// Print queue for testing, also it's gone now lol
	/*
	std::cout << "Popping out elements...";
	while (!mystack.empty())
	{
	 std::cout << ' ' << mystack.top();
	 mystack.pop();
	}
	std::cout << '\n';
	*/
	

	
//------------Seed workers--------------//
	for (rank = 1; rank < ntasks; ++rank) {
		// Get next thing of work
		// Probs don't need to synchronize?
		string work = mystack.top();
		mystack.pop();

		MPI_Send(work.c_str(), /* message buffer */
		work.length(),              /* one data item */
		MPI_CHAR,        /* data item is an integer */
		rank,           /* destination process rank */
		WORKTAG,        /* user chosen message tag */
		MPI_COMM_WORLD);/* always use this */

	}

	printf("Seeded workers\n");


	int count = 0;
//----Recieve more requests and respond-----//
	while (!mystack.empty()) { //queue is not empty
		// Get next thing of work
		// NEED TO SYNCHRONIZE
		string work = mystack.top();
		mystack.pop();

		// Recieve request
		MPI_Recv(&result,/* message buffer */
		1,               /* one data item */
		MPI_INT,      /* of type double real */
		MPI_ANY_SOURCE,  /* receive from any sender */
		MPI_ANY_TAG,     /* any type of message */
		MPI_COMM_WORLD,  /* always use this */
		&status);        /* received message info */

		printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);
		// Potentially update best
		if (result < best) {
			best = result;
		}

		// Respond with more work
		MPI_Send(work.c_str(), work.length(), MPI_CHAR, status.MPI_SOURCE,
		WORKTAG, MPI_COMM_WORLD);

		count++;
	}

	printf("Telling workers to exit\n");

//--------Recieve final reqests---------//
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		printf("Received final request from %d\n",status.MPI_SOURCE);
	}

//--------Tell workers to die----------//
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);

		printf("Sent dietag to %d\n",rank);
	}

	printf("Best: %d\n",best);

	// Get elapsed time
	elapsed_time += MPI_Wtime();
	printf("RESULT: \n %f\n",elapsed_time);
	fflush(stdout);
}


//*****************************************//
//   Stuff for worker (rank != 0) to do    //
//*****************************************//
void worker() {
	int result = 0;
	int work;
	MPI_Status status;
	
//----------Continuously do work-----------//
	for (;;) {
		// Recieve initial bit of work
		MPI_COMM_WORLD.Probe(source, 1, status);
		MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);

		// Check for die tag
		if (status.MPI_TAG == DIETAG) {
			return;
		}

//-----------DO WORK-----------//
		// WOW


//-----------Share answer???-----------//
		// Only send back computed result if it's better than best
		// Otherwise just send best back
		if (result > best) {
			result = best;
		}

		// Send a request for more work
		MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		//printf("Rank %d requesting more work\n",myrank);
	}
	fflush(stdout);
}


//*****************************************//
//           Helper Functions              //
//*****************************************//
// Read the cities input file into an array
void inCities() {
	ifstream inputFile;
	inputFile.open(filename);
	for (int row = 0; row < NUM_CITIES; row++) {
	    for (int col = 0; col < NUM_CITIES; col++) {   
	    	inputFile >> cities[row][col];
	    }   
	}
}

// Print the cities array for debugging
void printCities() {
	cout << "Rank: " << myrank << endl;
	for (int row = 0; row < NUM_CITIES; row++) {
	    for (int col = 0; col < NUM_CITIES; col++) {   
	    	cout << cities[row][col] << " ";
	    }   
	    cout << endl;
	}
}
