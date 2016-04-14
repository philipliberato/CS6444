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
using namespace std;

#define WORKTAG    1
#define DIETAG     2
#define NUM_CITIES 17		 	// change depending on city file
#define NUM_CITIES_1 NUM_CITIES-1
char filename[] = "city17.txt";	// change depending on city file

void master();
void worker();
void inCities();
void printCities();

int myrank, ntasks, best;
double elapsed_time;
int cities[NUM_CITIES][NUM_CITIES];
vector<int> all_cities;


// custom class for comparing vector<ints>
class Compare
{
public:
    bool operator() (vector<int> a, vector<int> b) {
		// count zeroes in a and b
		int a_zeroes, b_zeroes = 0;
		
		for (int i = 0; i < a.size(); i++) {
			if (a[i] == 0) {
				a_zeroes++;
			}
		}
		for (int i = 0; i < b.size(); i++) {
			if (b[i] == 0) {
				b_zeroes++;
			}
		}

		if (a_zeroes > b_zeroes) {
			return true;
		}
		else if (a_zeroes < b_zeroes) {
			return false;
		}
		else { //same number of zeroes
			//return a > b;
			return false;
		}
		//return (a<b);
	}
};

//stack< vector<int> > mystack;
priority_queue<vector<int>, vector<vector<int> >, Compare > mystack;


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
	vector<int> best_path(NUM_CITIES_1,0);

//-----------Set up the Queue-----------//
	for (int i = 2; i <= NUM_CITIES; ++i) {
		vector<int> tmp(NUM_CITIES_1,0);
		tmp[0] = i;
		mystack.push(tmp);
	}
		

	vector<int> work(NUM_CITIES_1,0);
	int pending_jobs = 0;

//------------Seed workers--------------//
	for (rank = 1; rank < ntasks; ++rank) {
		// Get next thing of work
		work = mystack.top();
		mystack.pop();
		
		// Print for testing
		/*cout << "Work: ";
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;*/

		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES_1,      /* length */
		MPI_INT,           /* data item is an integer */
		rank,              /* destination process rank */
		WORKTAG,           /* user chosen message tag */
		MPI_COMM_WORLD);   /* always use this */

		pending_jobs++;

		if (mystack.empty()) {
			break;
		}
	}

	printf("Seeded workers\n");

	
	vector<int> message(NUM_CITIES,0);
	vector<int> diff;

//----Recieve more requests and respond-----//
	while (!mystack.empty() || pending_jobs != 0) { //queue is not empty

		message.resize(NUM_CITIES);

		// Recieve request
		MPI_Recv(&message[0],/* message buffer */
		NUM_CITIES,        /* length */
		MPI_INT,             /* type int */
		MPI_ANY_SOURCE,  	 /* receive from any sender */
		MPI_ANY_TAG,     	 /* any type of message */
		MPI_COMM_WORLD,   	 /* always use this */
		&status);        	 /* received message info */

		pending_jobs--;

		result = message.back();
		message.pop_back();

		//printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);


		vector<int> tmpwork(message);
		vector<int> tmpworksorted(tmpwork);

		// Print for testing
		/*cout << result << "  tmpwork: ";
		for (int i = 0; i < tmpwork.size(); i++) {
			cout << tmpwork[i] << " ";
		}
		cout << endl;*/

		// Check if current sum is already worse than best
		if (result > best) {
			
		}
		else {
			//cout << "Spawn more work\n";
//-----------Spawn more work-----------//
			vector<int> diff(NUM_CITIES_1,0);
			vector<int>::iterator it;
			sort(tmpworksorted.begin(),tmpworksorted.end());
			//set_difference(all_cities.begin(), all_cities.end(), tmpwork.begin(), tmpwork.end(),inserter(diff,diff.end()));
			it = set_difference(all_cities.begin(), all_cities.end(), tmpworksorted.begin(), tmpworksorted.end(),diff.begin());

			diff.resize(it-diff.begin());

			// Print for testing
			/*cout << "DIFF: ";
			for (int i = 0; i < diff.size(); i++) {
				cout << diff[i] << " ";
			}
			cout << endl;*/


			// If all values in 'work' are not 0, send result
			if (diff.size() == 0) {
				// maybe update best
				if (result < best) {
					best = result;
					best_path = tmpwork;
					cout << "UPDATED BEST: " << best << endl;
				}
			}
			// If only 2 entries are left, we can just fill them in
			// And push 2 more work pieces
			else if (diff.size() == 2) {
				// Generate work
				vector<int> new_work(tmpwork);
				new_work[NUM_CITIES_1-2] = diff[0];
				new_work[NUM_CITIES_1-1] = diff[1];
				
				mystack.push(new_work);

				vector<int> new_work2(tmpwork);
				new_work2[NUM_CITIES_1-2] = diff[1];
				new_work2[NUM_CITIES_1-1] = diff[0];
				
				mystack.push(new_work2);
			}	
			else {
				// Generate work
				for (int i = 0; i < diff.size(); i++) {
					vector<int> new_work(tmpwork);
					new_work[NUM_CITIES_1-diff.size()] = diff[i];
					
					mystack.push(new_work);
					
					// Print for testing
					/*cout << "NEW WORK: ";
					for (int j =0; j < new_work.size(); j++) {
						cout << new_work[j] << " ";
					}
					cout << endl;*/
					
				}
			}
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
		
		// Print for testing
		/*cout << "Work: ";
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;*/

		// Respond with more work
		MPI_Send(&work[0], /* message buffer */
		NUM_CITIES_1,     /* one data item */
		MPI_INT,        /* data item is an integer */
		status.MPI_SOURCE, /* destination process rank */
		WORKTAG,        /* user chosen message tag */
		MPI_COMM_WORLD);/* always use this */

		pending_jobs++;
	}

	printf("Telling workers to exit\n");

//--------Recieve final reqests---------//
	/*for (rank = 1; rank < ntasks; ++rank) {
		MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		printf("Result: %d from Rank: %d\n",result,status.MPI_SOURCE);
		// Potentially update best
		if (result < best) {
			best = result;
		}

		printf("Received final request from %d\n",status.MPI_SOURCE);
	}*/

//--------Tell workers to die----------//
	for (rank = 1; rank < ntasks; ++rank) {
		work.resize(NUM_CITIES_1);
		MPI_Send(&work[0], NUM_CITIES_1, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);

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


//*****************************************//
//   Stuff for worker (rank != 0) to do    //
//*****************************************//
void worker() {
	// Load city data
	inCities();

	int sum = 0;
	vector<int> work;
	MPI_Status status;
	
//----------Continuously do work-----------//
	for (;;) {
		// Reset result
		sum = 0;

		// Reset work
		work.resize(NUM_CITIES_1);
		// Recieve work
		MPI_Recv(&work[0], NUM_CITIES_1, MPI_INT, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);

		// Check for die tag
		if (status.MPI_TAG == DIETAG) {
			return;
		}

		// Print for testing
		/*cout << "Rank: " << myrank << " received: ";
		for (int i = 0; i < work.size(); i++) {
			cout << work[i] << " ";
		}
		cout << endl;*/

//-----------DO WORK-----------//
		// COMPUTE SUM
		
		// assume beginss and ends with 1
		// beginning
		sum += cities[0][work[0]-1];
		// only do end edge if it's not a zero
		if (work[NUM_CITIES_1-1] != 0) {
			sum += cities[0][work[NUM_CITIES_1-1]-1];
		}
		
		// loop over the rest
		for (int i = 0; i < NUM_CITIES_1-1; i++) {
			int val1 = work[i];
			int val2 = work[i+1];
			// make sure neither is a zero
			if (val1 != 0 && val2 != 0) {
				sum += cities[val1-1][val2-1];
			}
		}


		vector<int> message(work);
		message.push_back(sum);

		// Send a request for more work
		MPI_Send(&message[0], NUM_CITIES, MPI_INT, 0, 0, MPI_COMM_WORLD);

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

