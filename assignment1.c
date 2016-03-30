#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#ifdef PTHREAD
//cache_alignment 64 byte
#	include <stdint.h>
#	include <pthread.h>
#endif

#ifdef OPENMP
#	include <omp.h>
#	include <stdint.h>
#endif

// for Debug purpose
#ifdef DEBUG_ENABLED
#	define dprintf(...) {			\
		printf("DEBUG] ");			\
		printf(__VA_ARGS__);		\
	}
#else
#	define dprintf(...) (0)
#endif

#define NANOSECOND 1000000000.0
#define ELAPSED(a, b) ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / NANOSECOND)

#define true 1
#define false 0

void do_solve();

double *_A, *_B, *_origA, *_origB;
int n, p;
int current = -1;

#define A(a, b) (_A[a * n + b])
#define B(i) (_B[i])
#define origA(a, b) (_origA[a * n + b])
#define origB(i) (_origB[i])

// for debug-purpose
#ifdef DEBUG_ENABLED
void print_result()
{
	for (int i = 0; i < n; i++)
	{
		dprintf("  %03d: ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%f ", A(i, j));
		}
		printf("%f\n", B(i));
	}
}
#endif

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("usage: ./assignment1 matrix_size thread_num");
		return -1;
	}

	int i, j;
	char *ptr;

	errno = 0;
	n = strtol(argv[1], &ptr, 10);
	if (errno > 0 || *ptr != '\0' || n < 1 || n > INT_MAX) { return -1; }
	p = strtol(argv[2], &ptr, 10);
	if (errno > 0 || *ptr != '\0' || p < 1 || p > INT_MAX) { return -1; }

	//_A = malloc(n * (n + 1) * sizeof(double));
	errno = posix_memalign((void **)&_A, 0x40, n * n * sizeof(double));
	if (errno > 0) { return -1; }
	errno = posix_memalign((void **)&_B, 0x40, n * sizeof(double));
	if (errno > 0) { return -1; }

	_origA = malloc(n * n * sizeof(double));
	_origB = malloc(n * sizeof(double));

	// time elapse
	struct timespec start, finish;
	double elapsed_time, total_time;

	clock_gettime(CLOCK_MONOTONIC, &start);

	// set array
	struct drand48_data rand_buffer;
//	srand(time(NULL));
//	srand48_r(time(NULL), &rand_buffer);
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n + 1; j++)
		{
			//drand48_r(&rand_buffer, &A(i, j));
			A(i, j) = (double)rand() / (RAND_MAX);
		}
		//drand48_r(&rand_buffer, &B(i));
		B(i) = (double)rand() / (RAND_MAX);
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	total_time = ELAPSED(start, finish);

	memcpy(_origA, _A, sizeof(double) * n * n);
	memcpy(_origB, _B, sizeof(double) * n);

	#ifdef DEBUG_ENABLED
		// print original array
		dprintf("original:\n");
		print_result();
	#endif

	clock_gettime(CLOCK_MONOTONIC, &start);
	do_solve();
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_time = ELAPSED(start, finish);

	#ifdef DEBUG_ENABLED
		// print result array
		dprintf("result:\n");
		print_result();
	#endif

	dprintf("final check:\n");
	double diff = 0.0;
	for (i = 0; i < n; i++)
	{
		double check_value = 0.0;
		for (j = 0; j < n; j++)
		{
			check_value += origA(i, j) * B(j);
		}
		dprintf("calculated: %.32lf, orig: %.32lf\n", check_value, origB(i));
		diff += (check_value - origB(i)) * (check_value - origB(i));
	}

	printf( "\n"
			"error (L2 norm): %.32lf\n"
			"matrix gen. time: %.32lf\n"
			"gaussian elim. time: %.32lf\n"
			, sqrt(diff), total_time, elapsed_time);

	total_time += elapsed_time;
	printf("\nTotal time: %.32lf\n", total_time);

	return 0;
}

#ifdef NAIVE
void do_solve()
{
	int i, j;

	// - 1st phase -
	while (++current < n)
	{
		int pivot_line = -1;
		double pivot = -DBL_MAX;

		// -- search pivot --
		for (i = current; i < n; i++)
		{
			if (pivot < A(i, current))
			{
				pivot = A(i, current);
				pivot_line = i;
			}
		}
		dprintf("find pivot %.16f in %d\n\n", pivot, pivot_line);

		// -- switch pivot --
		if (current != pivot_line)
		{
			double temp;
			for (j = current; j < n; j++)
			{
				temp = A(current, j);
				A(current, j) = A(pivot_line, j);
				A(pivot_line, j) = temp;
			}

			temp = B(current);
			B(current) = B(pivot_line);
			B(pivot_line) = temp;

			#ifdef DEBUG_ENABLED
				print_result();
				dprintf("switch pivot\n");
			#endif
		}


		// -- set pivot to 1 --
		for (j = current + 1; j < n; j++)
			A(current, j) /= pivot;

		B(current) /= pivot;
		A(current, current) = 1;

		#ifdef DEBUG_ENABLED
			print_result();
			dprintf("set pivot to 1\n");
		#endif

		// -- set other to 0 --
		for (i = current + 1; i < n; i++)
		{
			double target = -A(i, current);
			for (j = current; j < n; j++)
			{
				A(i, j) += target * A(current, j);
			}
			B(i) += target * B(current);
		}

		#ifdef DEBUG_ENABLED
			print_result();
			dprintf("set other to 0\n");
		#endif
	}
	// - 1st phase end -

	// - 2nd phase -
	while (--current > 0)
	{
		// -- set other to 0 --
		for (i = 0; i < current; i++)
		{
			double target = -A(i, current);
			// doesn't need
			// A(i, current) = 0;
			B(i) += target * B(current);
		}
	}
	// - 2nd phase end -
}
#endif

#if defined(PTHREAD) || defined(OPENMP)
struct Thread_data
{
	double pivot;		/* 8 bytes */
	int pivot_line;		/* 4 bytes */
	uint8_t pad[52];	/* 52 bytes */
}	*thread_data;		/* = 64 byte aligned */
#endif

#ifdef PTHREAD

int pivot_line;
double pivot;

struct
{
	pthread_mutex_t count_lock;
	pthread_cond_t count_cond;
	int done_count;
}	c_barrier;

void *do_pthread(void *data)
{
	int tid = (intptr_t)data;
	int i, j;

	// align 64byte
	int align = (64 / sizeof(double));
	int block_count = n / align;

	dprintf("tid #%d is start...\n", tid);
/*
	// initialize - search local pivot
	thread_data[tid].pivot = -DBL_MAX;
	for (i = tid; i < n; i += p)
	{
		if (thread_data[tid].pivot < A(i, 0))
		{
			thread_data[tid].pivot = A(i, 0);
			thread_data[tid].pivot_line = i;
		}
	}
*/
	// - 1st phase -
	/*** current increase AFTER while statement ***/
	while (current < n - 1)
	{
		// --- search *next* local pivot ---
		thread_data[tid].pivot_line = -1;
		thread_data[tid].pivot = -DBL_MAX;
		for (i = (current + 1) + tid; i < n; i += p)
		{
			if (thread_data[tid].pivot < A(i, (current + 1)))
			{
				thread_data[tid].pivot = A(i, (current + 1));
				thread_data[tid].pivot_line = i;
			}
		}

		// barrier
		pthread_mutex_lock(&(c_barrier.count_lock));
			c_barrier.done_count++;
			if (c_barrier.done_count == p)
			{
				c_barrier.done_count = 0;

				// --- collect pivot ---
				pivot = -DBL_MAX;
				for (i = 0; i < p; i++)
				{
					if (pivot < thread_data[i].pivot)
					{
						pivot = thread_data[i].pivot;
						pivot_line = thread_data[i].pivot_line;
					}
				}

				#ifdef DEBUG_ENABLED
					print_result();
					dprintf("find pivot %.16f in %d\n\n", pivot, pivot_line);
				#endif

				// increase current
				current++;
				pthread_cond_broadcast(&(c_barrier.count_cond));
			}
			else
			{
				while (pthread_cond_wait(&(c_barrier.count_cond), &(c_barrier.count_lock)) != 0);
			}
		pthread_mutex_unlock(&(c_barrier.count_lock));

		// block operation - operate row block
		for (int block = tid; block <= block_count; block += p)
		{
			int start_block = block * align;
			int end_block = (block + 1) * align;
			if (end_block < current) continue;
			if ( __builtin_expect(start_block < current, false) ) start_block = current;

			// -- switching pivot --
			if ( __builtin_expect(current != pivot_line, true) ) // usually true
			{
				double temp;

				for (j = start_block; j < end_block && j < n; j++)
				{
					temp = A(current, j);
					A(current, j) = A(pivot_line, j);
					A(pivot_line, j) = temp;
				}

				#ifdef DEBUG_ENABLED
					print_result();
					dprintf("switching finished\n\n");
				#endif
			}

			// -- set pivot to 1 --
			for (j = start_block; j < end_block && j < n; j++)
			{
				A(current, j) /= pivot;
			}
			#ifdef DEBUG_ENABLED
				print_result();
				dprintf("set pivot to 1\n\n");
			#endif
		}

		// barrier
		pthread_mutex_lock(&(c_barrier.count_lock));
			c_barrier.done_count++;
			if (c_barrier.done_count == p)
			{
				c_barrier.done_count = 0;

				// switching & set on B
				if ( __builtin_expect(current != pivot_line, true) ) // usually true
				{
					double temp;
					temp = B(current);
					B(current) = B(pivot_line);
					B(pivot_line) = temp;
				}
				B(current) /= pivot;

				#ifdef DEBUG_ENABLED
					print_result();
					dprintf("switch and divided on B\n\n");
				#endif

				pthread_cond_broadcast(&(c_barrier.count_cond));
			}
			else
			{
				while (pthread_cond_wait(&(c_barrier.count_cond), &(c_barrier.count_lock)) != 0);
			}
		pthread_mutex_unlock(&(c_barrier.count_lock));

		// -- set other to 0 --
		for (i = (current + 1) + tid; i < n; i += p)
		{
			double target = -A(i, current);

			// doesn't need
			 A(i, current) = 0;
			for (j = (current + 1); j < n; j++)
			{
				A(i, j) += target * A(current, j);
			}
			B(i) += target * B(current);
		}
		#ifdef DEBUG_ENABLED
			print_result();
			dprintf("set to 0 finish\n\n");
		#endif
	}
	current = n;

	// - 2nd phase -
	/*** current decrease AFTER while statement ***/
	while (current > 1)
	{
		// -- decrese current & do barrier --
		pthread_mutex_lock(&(c_barrier.count_lock));
			c_barrier.done_count++;
			if (c_barrier.done_count == p)
			{
				c_barrier.done_count = 0;
				current--;

				if (p > current) p = current;
				pthread_cond_broadcast(&(c_barrier.count_cond));
			}
			else
			{
				while (pthread_cond_wait(&(c_barrier.count_cond), &(c_barrier.count_lock)) != 0);
			}
		pthread_mutex_unlock(&(c_barrier.count_lock));

		// exit if size is too small
		if (tid >= current) pthread_exit(NULL);

		// -- back substitution --
		for (i = tid; i < current; i += p)
		{
			double target = -A(i, current);
			// doesn't need
			// A(i, current) = 0;
			B(i) += target * B(current);
		}
	}

	pthread_exit(NULL);
}

void do_solve()
{
	pthread_t threads[p];
	pthread_attr_t thread_attr;

	pthread_attr_init(&thread_attr);

	//int err = posix_memalign((void **)&thread_data, 0x40, p * sizeof(struct Thread_data));
	//if (err > 0) return ;
	thread_data = malloc(p * sizeof(struct Thread_data));

	// barrier init
	c_barrier.done_count = 0;
	pthread_mutex_init(&(c_barrier.count_lock), NULL);
	pthread_cond_init(&(c_barrier.count_cond), NULL);

	for (int i = 0; i < p; i++)
	{
		void *data = (void *)(uintptr_t)(i);
		pthread_create(&threads[i], &thread_attr, do_pthread, data);
	}

	for (int i = 0; i < p; i++)
	{
		pthread_join(threads[i], NULL);
	}
	dprintf("Ended.\n");
}
#endif

#ifdef OPENMP
void do_solve()
{
	int i, j;
	int tid;

	int err = posix_memalign((void **)&thread_data, 0x40, p * sizeof(struct Thread_data));
	if (err > 0) return ;

	// - 1st phase -
	while (++current < n)
	{
		int pivot_line = -1;
		double pivot = -DBL_MAX;

		// -- search pivot --
		#pragma omp parallel num_threads(p) default(none) private(i, tid) shared(thread_data, current, n, _A)
		{
			tid = omp_get_thread_num();
			thread_data[tid].pivot_line = -1;
			thread_data[tid].pivot = -DBL_MAX;

			#pragma omp for
			for (i = current; i < n; i++)
			{
				if (thread_data[tid].pivot < A(i, current))
				{
					thread_data[tid].pivot = A(i, current);
					thread_data[tid].pivot_line = i;
				}
			}
		}
		for (i = 0; i < p; i++)
		{
			if (pivot < thread_data[i].pivot)
			{
				pivot = thread_data[i].pivot;
				pivot_line = thread_data[i].pivot_line;
			}
		}
		dprintf("find pivot %.16f in %d\n", pivot, pivot_line);

		// -- switch pivot --
		if (current != pivot_line)
		{
			double temp;
			#pragma omp parallel num_threads(p) default(none) private(j, temp) shared(current, pivot_line, n, _A)
			{
				#pragma omp for
				for (j = current; j < n; j++)
				{
					temp = A(current, j);
					A(current, j) = A(pivot_line, j);
					A(pivot_line, j) = temp;
				}
			}
			temp = B(current);
			B(current) = B(pivot_line);
			B(pivot_line) = temp;
		}

		// -- set pivot to 1 --
		#pragma omp parallel num_threads(p) default(none) private(j) shared(current, pivot, n, _A)
		{
			#pragma omp for
			for (j = current + 1; j < n; j++)
				A(current, j) /= pivot;
		}
		B(current) /= pivot;
		A(current, current) = 1;

		// -- set other to 0 --
		#pragma omp parallel num_threads(p) default(none) private(i, j) shared(current, n, _A, _B)
		{
			#pragma omp for
			for (i = current + 1; i < n; i++)
			{
				double target = -A(i, current);
				for (j = current; j < n; j++)
				{
					A(i, j) += target * A(current, j);
				}
				B(i) += target * B(current);
			}
		}
		#ifdef DEBUG_ENABLED
			print_result();
		#endif
	}
	// - 1st phase end -

	// - 2nd phase -
	while (--current > 0)
	{
		// -- set other to 0 --
		#pragma omp parallel num_threads(p) default(none) private(i) shared(current, n, _A, _B)
		{
			#pragma omp for
			for (i = 0; i < current; i++)
			{
				double target = -A(i, current);
				// doesn't need
				// A(i, current) = 0;
				B(i) += target * B(current);
			}
		}
	}
	// - 2nd phase end -
}
#endif