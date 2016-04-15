// This version pre-generates the search space,
// then prunes it

// Whereas the other version generated one layer of the tree at a time,
// one branch at a time

// tsp.cpp

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
#include <queue>
#include <functional>
#include <math.h>
using namespace std;

#define WORKTAG    1
#define DIETAG     2
#define SHARETAG   3
#define NUM_CITIES 17		 	// change depending on city file
#define NUM_CITIES_1 NUM_CITIES-1
char filename[] = "city17.txt";	// change depending on city file

void master();
void worker();
void inCities();
void printCities();
void branch(vector<int> work);

int myrank, ntasks;
double elapsed_time;
int cities[NUM_CITIES][NUM_CITIES];
vector<int> all_cities;

stack< vector<int> > mystack;
//priority_queue<vector<int>, vector<vector<int> >, Compare > mystack;


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
	for (int i = 2; i <= NUM_CITIES; i++) {
		all_cities.push_back(i);
	}

	// Print for testing
	/*cout << "ALL CITIES: ";
	for (int i = 0; i < NUM_CITIES; i++) {
		cout << all_cities[i] << " ";
	}
	cout << endl;*/

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
	int rank, local_best;
	MPI_Status status;
	int best = INT_MAX;
	vector<int> best_path(NUM_CITIES_1,0);

//-----------Set up the Queue-----------//
	// First layer, can leave off the last city because of reversals
	//for (int i = NUM_CITIES_1; i > 1; i--) {
	for (int i = 2; i < NUM_CITIES; i++) {
		vector<int> tmp(NUM_CITIES_1,0);
		tmp[0] = i;
		// Second layer
		for (int j = 2; j <= NUM_CITIES; j++) {
			if (i != j) {
				tmp[1] = j;
			
				// Third layer
				for (int k = 2; k <= NUM_CITIES; k++) {
					if (k != j && k != i) {
						tmp[2] = k;

						// Print for testing
						// cout << "init: ";
						// for (int j = 0; j < tmp.size(); j++) {
						// 	cout << tmp[j] << " ";
						// }
						// cout << endl;

						mystack.push(tmp);
					}
				}
			}
		}
	}
		
//------------Seed workers--------------//
	vector<int> work(NUM_CITIES_1,0);
	int pending_jobs = 0;

	for (rank = 1; rank < ntasks; ++rank) {
		// Get next thing of work
		work = mystack.top();
		mystack.pop();

		// Add current best to back
		work.push_back(INT_MAX);
		
		// Print for testing
		/*cout << "Work: ";
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;*/

		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES,        /* length */
		MPI_INT,           /* data item is an integer */
		rank,              /* destination process rank */
		WORKTAG,           /* user chosen message tag */
		MPI_COMM_WORLD);   /* always use this */

		pending_jobs++;

		// this 'if' won't happen now that we generate more inital tasks than workers
		// but its here for testing (and using smaller amounts of workers)
		if (mystack.empty()) {
			break;
		}
	}

	printf("Seeded workers\n");


//----Recieve more requests and respond-----//	
	vector<int> message(NUM_CITIES,0);

	while (!mystack.empty() || pending_jobs != 0) { //queue is not empty

		message.resize(NUM_CITIES);

		// Recieve request
		MPI_Recv(&message[0],/* message buffer */
		NUM_CITIES,          /* length */
		MPI_INT,             /* type int */
		MPI_ANY_SOURCE,  	 /* receive from any sender */
		MPI_ANY_TAG,     	 /* any type of message */
		MPI_COMM_WORLD,   	 /* always use this */
		&status);        	 /* received message info */

		pending_jobs--;

		// Fetch what this worker's local best was
		local_best = message.back();
		message.pop_back();

		//printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);


		// Potentially update global best
		if (local_best < best) {
			best = local_best;
			best_path = message;
			cout << "UPDATED BEST: " << best << " [ ";
			for (int i = 0; i < best_path.size(); i++) {
				cout << best_path[i] << " ";
			}
			cout << "]" << endl;
		}

		// idk where this goes
		if (mystack.empty()) {
			if (pending_jobs == 0) {
				break;
			}
			else {
				continue;
			}
		}

		// Get next thing of work
		work = mystack.top();
		mystack.pop();

		// Add current best to back
		work.push_back(best);
		
		// Print for testing
		// cout << "Work: ";
		// for (int i = 0; i < work.size(); i++) {
		// 	cout << work[i] << " ";
		// }
		// cout << endl;

		// Respond with more work
		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES,        /* one data item */
		MPI_INT,           /* data item is an integer */
		status.MPI_SOURCE, /* destination process rank */
		WORKTAG,           /* user chosen message tag */
		MPI_COMM_WORLD);   /* always use this */

		pending_jobs++;
	}

	printf("Telling workers to exit\n");

//--------Tell workers to die----------//
	for (rank = 1; rank < ntasks; ++rank) {
		work.resize(NUM_CITIES);
		MPI_Send(&work[0], NUM_CITIES, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);

		//printf("Sent dietag to %d\n",rank);
	}

	printf("Best: %d\n",best);

	cout << "Best path: 1 ";
	for (int i = 0; i < best_path.size(); i++) {
		cout << best_path[i] << " ";
	}
	cout << "1 " << endl;

	// Get elapsed time
	elapsed_time += MPI_Wtime();
	printf("RESULT: \n %f\n",elapsed_time);
	fflush(stdout);
}

// Globals for worker
int local_best = INT_MAX;
vector<int> local_best_path;
stack< vector<int> > localstack;
int lb; // lower bound
int potential_best; // local bests recieved from other nodes

//*****************************************//
//   Stuff for worker (rank != 0) to do    //
//*****************************************//
void worker() {
	// Load city data
	inCities();
	// A copy of cities to manipulate for lower bound computation
	int cities_copy[NUM_CITIES][NUM_CITIES];

	// Partial sum
	int sum = 0;
	// Next work item
	vector<int> work;
	// MPI Status wow no way
	MPI_Status status;

//----------Continuously recv work-----------//
	for (;;) {
		// Reset work
		work.resize(NUM_CITIES);

		// Recieve work
		MPI_Recv(&work[0], NUM_CITIES, MPI_INT, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);

		// Check for die tag
		if (status.MPI_TAG == DIETAG) {
			return;
		}

		// Update local best
		local_best = work.back();
		work.pop_back();

		// BRANCH on 'work'
		branch(work);

//-----------DO WORK-----------//
		// BOUND
		while (!localstack.empty()) {
			// Get next thing of work
			work = localstack.top();
			localstack.pop();

			// cout << "Rank: " << myrank << "  work: ";
			// for (int i = 0; i < work.size(); i++) {
			// 	cout << work[i] << " ";
			// }
			// cout << endl;

			// Reset result
			sum = 0;
			// Re-copy cities
			for (int i = 0; i < NUM_CITIES; i++) {
				for (int j = 0; j < NUM_CITIES; j++) {
					if (i == j) {
						cities_copy[i][j] = INT_MAX;
					}
					else {
						cities_copy[i][j] = cities[i][j];
					}
				}
			}

			// Compute partial sum
			// assume beginss and ends with 1
			// beginning
			sum += cities[0][work[0]-1];

			// mark edge as used in cities
			cities_copy[0][work[0]-1] = INT_MAX;
			cities_copy[work[0]-1][0] = INT_MAX;

			// only do end edge if it's not a zero
			if (work[NUM_CITIES_1-1] != 0) {
				sum += cities[0][work[NUM_CITIES_1-1]-1];
			}
			// figure out how many zeroes there are and where they start
			int zeroes = 0;
			int zero_idx = 0;
			
			// loop over the rest
			for (int i = 0; i < NUM_CITIES_1-1; i++) {
				int val1 = work[i];
				int val2 = work[i+1];
				// make sure neither is a zero
				if (val1 != 0 && val2 != 0) {
					sum += cities[val1-1][val2-1];

					// mark this edge as being used in copy of "cities"
					cities_copy[val1-1][val2-1] = INT_MAX;
					cities_copy[val2-1][val1-1] = INT_MAX;
				}
				else if (val1 != 0 && val2 == 0) {
					zero_idx = i;
					zeroes++;
				}
				else {
					zeroes++;
				}
			}

			// If partial sum is < local best
			if (sum < local_best) {
				// Compute lower bound
				vector<int> work_copy(work);

				// compute the lower bound by adding up the best edges
				// for all the edges that are left over
				lb = sum;
				// for each zero
				for (int i = 0; i < zeroes; i++) {
					int val1 = work_copy[zero_idx+i];
					int least = INT_MAX; // min of row val1

					int val2 = 0;

					for (int j = 0; j < NUM_CITIES_1; j++) {
						int next = cities_copy[val1-1][j];
						if (next < least) {
							val2 = j+1;
							least = next;
						}
					}

					// set edge (val1,val2_idx) as INF
					cities_copy[val1-1][val2-1] = INT_MAX;
					cities_copy[val2-1][val1-1] = INT_MAX;

					// write to work_copy
					work_copy[zero_idx+i+1] = val2;

					lb += least;
				}

				// If lower bound < local best,
				// potentially update local best, and BRANCH on 'work'
				if (lb < local_best) {
		//-----------Branch on 'work'-----------//
					branch(work);
				}
				// else don't branch on 'work'
			}
			// else don't branch on 'work'
		} // END WHILE

		// Send back local best path and local best, asking for more work
		vector<int> message(local_best_path);
		message.push_back(local_best);

		// Send a request for more work
		MPI_Send(&message[0], NUM_CITIES, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	fflush(stdout);
}


//*****************************************//
//           Helper Functions              //
//*****************************************//
// Branch function
void branch(vector<int> work) {

	vector<int> tmpwork(work);
	vector<int> tmpworksorted(tmpwork);

	vector<int> diff(NUM_CITIES_1,0);
	vector<int>::iterator it;
	sort(tmpworksorted.begin(),tmpworksorted.end());
	it = set_difference(all_cities.begin(), all_cities.end(), tmpworksorted.begin(), tmpworksorted.end(),diff.begin());

	diff.resize(it-diff.begin());

	// Print for testing
	// cout << "DIFF: ";
	// for (int i = 0; i < diff.size(); i++) {
	// 	cout << diff[i] << " ";
	// }
	// cout << endl;

	// If full candidate solution, update local best
	if (diff.size() == 0) {
		local_best = lb;
		local_best_path = work;
	}
	// If only 2 entries are left, we can just fill them in
	// And push 2 more work pieces
	else if (diff.size() == 2) {
		// Generate work
		vector<int> new_work(tmpwork);
		new_work[NUM_CITIES_1-2] = diff[0];
		new_work[NUM_CITIES_1-1] = diff[1];
		
		localstack.push(new_work);

		vector<int> new_work2(tmpwork);
		new_work2[NUM_CITIES_1-2] = diff[1];
		new_work2[NUM_CITIES_1-1] = diff[0];
		
		localstack.push(new_work2);
	}	
	else {
		// Generate work
		for (int i = 0; i < diff.size(); i++) {
			vector<int> new_work(tmpwork);
			new_work[NUM_CITIES_1-diff.size()] = diff[i];
			
			localstack.push(new_work);
			
			// Print for testing
			// cout << "generated: ";
			// for (int j =0; j < new_work.size(); j++) {
			// 	cout << new_work[j] << " ";
			// }
			// cout << endl;
		}
	}
}


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

