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
#include <vector>
#include <algorithm>
#include <set>
#include <iterator>
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
vector<int> all_cities;

stack< vector<int> > mystack; //NEED TO SYNCHRONIZE


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

	// Set up all cities vector
	for (int i = 1; i <= NUM_CITIES; i++) {
		all_cities.push_back(i);
	}

	// Print for testing
	cout << "ALL CITIES: ";
	for (int i = 0; i < NUM_CITIES; i++) {
		cout << all_cities[i] << " ";
	}
	cout << endl;

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
	int rank, result;
	MPI_Status status;

//-----------Set up the Queue-----------//
	// Probably don't need to synchronize initially lol
	for (int i = 2; i <= NUM_CITIES; ++i) {
		vector<int> tmp(NUM_CITIES,0);
		tmp[0] = 1;
		tmp[1] = i;
		mystack.push(tmp);
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
		
		vector<int> work = mystack.top();
		mystack.pop();
		
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		int tmpwork = 0;
		cout << endl;

		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES,     /* one data item */
		MPI_INT,        /* data item is an integer */
		rank,           /* destination process rank */
		WORKTAG,        /* user chosen message tag */
		MPI_COMM_WORLD);/* always use this */

	}

	printf("Seeded workers\n");


	int count = 0;
	
	bool empty = mystack.empty();
	
	vector<int> message(NUM_CITIES+1,0);
	vector<int> work(NUM_CITIES,0);

//----Recieve more requests and respond-----//
	while (!empty) { //queue is not empty

		message.resize(NUM_CITIES+1);

		// Recieve request
		MPI_Recv(&message[0],/* message buffer */
		NUM_CITIES+1,               /* one data item */
		MPI_INT,      /* of type double real */
		MPI_ANY_SOURCE,  /* receive from any sender */
		MPI_ANY_TAG,     /* any type of message */
		MPI_COMM_WORLD,  /* always use this */
		&status);        /* received message info */

		result = message.back();
		message.pop_back();


		printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);
		cout << "Message: ";
		for (int i = 0; i < message.size(); i++) {
			cout << message[i] << " ";
		}
		cout << endl;

		vector<int> tmpwork(message);

		// Check if current sum is already worse than best
		if (result > best) {
			
		}
		else {
//-----------Spawn more work-----------//
			vector<int> diff;
			set_difference(all_cities.begin(),all_cities.end(), tmpwork.begin(), tmpwork.end(),inserter(diff,diff.end()));

			cout << "DIFF: ";
			for (int i = 0; i < diff.size(); i++) {
				cout << diff[i] << " ";
			}
			cout << endl;


//-----------Share answer???-----------//

			// if all values in 'work' are not 0, send result
			if (diff.size() == 0) {
				if (result < best) {
					best = result;
				}
			}
			// else send INT_MAX
			else { // ALSO add jobs to queue
				// Generate work
				for (int i = 0; i < diff.size(); i++) {
					vector<int> new_work(tmpwork);
					new_work[NUM_CITIES-diff.size()] = diff[i];

					
					mystack.push(new_work);
					
					// Print
					cout << "NEW WORK: ";
					for (int j =0; j < new_work.size(); j++) {
						cout << new_work[j] << " ";
					}
					cout << endl;
					
				}
			}
		}
		// Potentially update best
		/*if (result < best) {
			best = result;
		}*/


		// idk where this goes
		empty = mystack.empty();

		// Get next thing of work
		work = mystack.top();
		mystack.pop();
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;

		// Respond with more work
		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES,     /* one data item */
		MPI_INT,        /* data item is an integer */
		status.MPI_SOURCE, /* destination process rank */
		WORKTAG,        /* user chosen message tag */
		MPI_COMM_WORLD);/* always use this */

		count++;
		
	}

	printf("Telling workers to exit\n");

//--------Recieve final reqests---------//
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);
		// Potentially update best
		if (result < best) {
			best = result;
		}

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
	// Load city data
	inCities();

	int result = INT_MAX;
	int sum = 0;
	vector<int> work;
	MPI_Status status;
	
//----------Continuously do work-----------//
	for (;;) {
		// Reset result
		sum = 0;

		// Reset work
		work.resize(NUM_CITIES);
		// Recieve work
		MPI_Recv(&work[0], NUM_CITIES, MPI_INT, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);

		// Check for die tag
		if (status.MPI_TAG == DIETAG) {
			return;
		}

		// Print for testing
		cout << "Rank: " << myrank << " received: ";
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;

//-----------DO WORK-----------//
		// COMPUTE SUM
		for (int i = 0; i < NUM_CITIES; i++) {
			if ((i+1) < NUM_CITIES) {
				
				int val1 = work[i];
				int val2 = work[i+1];
				
				//cout << "VALS: (" << val1 << ", " << val2 << ")\n";
				if (val1 != 0 && val2 != 0) {
					sum += cities[val1-1][val2-1];
				}
			}
		}

		vector<int> message(work);
		message.push_back(sum);

		// Send a request for more work
		MPI_Send(&message[0], NUM_CITIES+1, MPI_INT, 0, 0, MPI_COMM_WORLD);

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
