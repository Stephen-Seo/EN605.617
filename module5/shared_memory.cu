#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short int u16;
typedef unsigned int u32;

#define NUM_ELEMENTS 256

#define MAX_NUM_LISTS 16

void checkError() {
	cudaError_t errorValue = cudaGetLastError();
	if (errorValue != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(errorValue));
	}
}

__host__ void cpu_sort(u32 * const data, const u32 num_elements)
{
	static u32 cpu_tmp_0[NUM_ELEMENTS];
	static u32 cpu_tmp_1[NUM_ELEMENTS];

	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for(u32 i=0; i<num_elements; i++)
		{
			const u32 d = data[i];
			if((d & bit_mask) > 0)
			{
				cpu_tmp_1[base_cnt_1] = d;
				base_cnt_1++;
			}
			else
			{
				cpu_tmp_0[base_cnt_0] = d;
				base_cnt_0++;
			}
		}

		// Copy data back to the source
		// First the zero list, then the one list
		for(u32 i=0; i<base_cnt_0; i++)
		{
			data[i] = cpu_tmp_0[i];
		}
		for(u32 i = 0; i<base_cnt_1; i++)
		{
			data[base_cnt_0+i] = cpu_tmp_1[i];
		}
	}
}

__device__ void radix_sort(u32 * const sort_tmp,
				const u32 tid,
				u32 * const sort_tmp_0,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<NUM_ELEMENTS; i+=MAX_NUM_LISTS)
		{
			const u32 elem = sort_tmp[i+tid];
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1+tid] = elem;
				base_cnt_1+=MAX_NUM_LISTS;
			}
			else
			{
				sort_tmp_0[base_cnt_0+tid] = elem;
				base_cnt_0+=MAX_NUM_LISTS;
			}
		}

		__syncthreads();
		
		// Copy data back to source - first the zero list
		for(u32 i=0;i<base_cnt_0;i+=MAX_NUM_LISTS)
		{
			sort_tmp[i+tid] = sort_tmp_0[i+tid];
		}
		
		//Copy data back to source - then the one list
		for(u32 i=0;i<base_cnt_1; i+=MAX_NUM_LISTS)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}

		__syncthreads();
	}
}

__device__ void radix_sort2(u32 * const sort_tmp,
				const u32 tid,
				u32 * const sort_tmp_0,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<NUM_ELEMENTS / MAX_NUM_LISTS; ++i)
		{
			const u32 elem = sort_tmp[i * MAX_NUM_LISTS + tid];
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1 * MAX_NUM_LISTS + tid] = elem;
				++base_cnt_1;
			}
			else
			{
				sort_tmp_0[base_cnt_0 * MAX_NUM_LISTS + tid] = elem;
				++base_cnt_0;
			}
		}

		__syncthreads();

		// Copy data back to source - first the zero list
		for(u32 i = 0; i < base_cnt_0; ++i)
		{
			sort_tmp[i * MAX_NUM_LISTS + tid] =
					sort_tmp_0[i * MAX_NUM_LISTS + tid];
		}
		
		//Copy data back to source - then the one list
		for(u32 i = 0; i < base_cnt_1; ++i)
		{
			sort_tmp[(base_cnt_0 + i) * MAX_NUM_LISTS + tid] =
					sort_tmp_1[i * MAX_NUM_LISTS + tid];
		}

		__syncthreads();
	}
}

u32 find_min(const u32 * const src_array,
		u32 * const list_indexes,
		const u32 num_lists,
		const u32 num_elements_per_list)
{
	u32 min_val = 0xFFFFFFF;
	u32 min_idx = 0;
	// Iterate over each of the lists
	for(u32 i=0; i<num_lists; i++)
	{
		// If the current list ahs already been emptied
		// then ignore it
		if(list_indexes[i] < num_elements_per_list)
		{
			const u32 src_idx = i + (list_indexes[i] * num_lists);

			const u32 data = src_array[src_idx];
	
			if(data <= min_val)
			{
				min_val = data;
				min_idx = i;
			}
		}
	}
	list_indexes[min_idx]++;
	return min_val;
}

void merge_array(const u32 * const src_array,
			u32 * const dest_array,
			const u32 num_lists,
			const u32 num_elements)
{
	const u32 num_elements_per_list = (num_elements / num_lists);

	unsigned int list_indexes[MAX_NUM_LISTS];

	for(u32 list=0; list < MAX_NUM_LISTS; list++)
	{
		list_indexes[list] = 0;
	}

	for(u32 i=0; i<num_elements; i++)
	{
		dest_array[i] = find_min(src_array,
					list_indexes,
					num_lists,
					num_elements_per_list);
	}
}

__device__ void copy_data_to_shared(const u32 * const data,
									u32 * const sort_tmp,
									const u32 tid)
{
	// Copy data into temp store
	for(u32 i = 0; i<NUM_ELEMENTS / MAX_NUM_LISTS; ++i)
	{
		//printf("%d: put into %d\n", tid, i * MAX_NUM_LISTS + tid);
		sort_tmp[i * MAX_NUM_LISTS + tid] = data[i * MAX_NUM_LISTS + tid];
	}
	__syncthreads();
}


// Uses a single thread for merge
__device__ void merge_array1(const u32 * const src_array,
							u32 * const dest_array,
							const u32 tid)
{
	__shared__ char list_indexes[MAX_NUM_LISTS];

	// Multiple threads
	list_indexes[tid] = 0;
	__syncthreads();

	// Single threaded
	if(tid == 0)
	{
		for (unsigned int dest_idx = 0; dest_idx < NUM_ELEMENTS; ++dest_idx) {
			unsigned int min = 0xFFFFFFFF;
			unsigned int idx = 0;

			for(unsigned int i = 0; i < MAX_NUM_LISTS; ++i) {
				if (list_indexes[i] >= MAX_NUM_LISTS) {
					continue;
				}
				unsigned int temp_idx = list_indexes[i] * MAX_NUM_LISTS + i;
				if (src_array[temp_idx] < min) {
					min = src_array[temp_idx];
					idx = i;
				}
			}

			dest_array[dest_idx] = min;
			++list_indexes[idx];
		}
	}
}

__global__ void gpu_sort_array_array(u32 * const data)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ u32 sort_tmp[NUM_ELEMENTS];
	__shared__ u32 sort_tmp_0[NUM_ELEMENTS];
	__shared__ u32 sort_tmp_1[NUM_ELEMENTS];

	copy_data_to_shared(data, sort_tmp, tid);

	radix_sort2(sort_tmp, tid, sort_tmp_0, sort_tmp_1);

	//copy_data_to_shared(sort_tmp, data, tid);

	merge_array1(sort_tmp, data, tid);
}

// Uses multiple threads for merge
// Deals with multiple identical entries in the data
__device__ void merge_array6(const u32 * const src_array,
								u32 * const dest_array,
								const u32 tid)
{
	const u32 num_elements_per_list = (NUM_ELEMENTS / MAX_NUM_LISTS);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	list_indexes[tid] = 0;

	//Wait for list_indexes[tid] to be cleared
	__syncthreads();

	//Iterate over all elements
	for(u32 i=0; i<NUM_ELEMENTS; i++)
	{
		//Create a value shared with other threads
		__shared__ u32 min_val;
		__shared__ u32 min_tid;

		// Use a temp register for work purposes
		u32 data;

		//If the current list has not already been
		//emptied then read from it, else ignore it
		if(list_indexes[tid] < num_elements_per_list)
		{
			//Work out from the list_index, the index into
			// the linear array
			const u32 src_idx = tid + (list_indexes[tid] * MAX_NUM_LISTS);

			//Read the data from the list for the given
			// thread
			data = src_array[src_idx];
		}
		else
		{
			data = 0xFFFFFFFF;
		}

		//Have thread zero clear the min values
		if(tid == 0)
		{
			// Write a very large value so the first
			// thread wins with the min
			min_val = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}

		// Wait for all threads
		__syncthreads();

		// Have every thread try to store it's value into
		// min_val. Only the thread with the lowest value
		// will win.
		atomicMin(&min_val, data);

		//Make sure all threads have taken their turn
		__syncthreads();

		// If this thread was the one with the minimum
		if(min_val == data)
		{
			// Check for equal values
			// Lowest tid wins, and does the write
			atomicMin(&min_tid, tid);
		}

		// Make sure all threads have taken their turn.
		__syncthreads();

		// If this thread has the lowest tid
		if(tid == min_tid)
		{
			// Increment the list pointer for this thread
			list_indexes[tid]++;

			// Store the winning value
			dest_array[i] = data;
		}
	}
}

// Uses multiple threads for reduction type merge
__device__ void merge_array5(const u32 * const src_array,
								u32 * const dest_array,
								const u32 num_lists,
								const u32 num_elements,
								const u32 tid)
{
	const u32 num_elements_per_list = (num_elements / num_lists);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	__shared__ u32 reduction_val[MAX_NUM_LISTS];
	__shared__ u32 reduction_idx[MAX_NUM_LISTS];

	//Clear the working sets
	list_indexes[tid] = 0;
	reduction_val[tid] = 0;
	reduction_idx[tid] = 0;
	__syncthreads();

	for(u32 i=0; i<num_elements; i++)
	{
		// We need (num_lists / 2) active threads
		u32 tid_max = num_lists >> 1;

		u32 data;

		// If the current list has already been
		// emptied then ignore it
		if(list_indexes[tid] < num_elements_per_list)
		{
			const u32 src_idx = tid + (list_indexes[tid] * num_lists);

			data = src_array[src_idx];
		}
		else
		{
			data = 0xFFFFFFFF;
		}

		reduction_val[tid] = data;
		reduction_idx[tid] = tid;

		__syncthreads();

		while(tid_max != 0)
		{
			if(tid < tid_max)
			{
				const u32 val2_idx = tid + tid_max;

				const u32 val2 = reduction_val[val2_idx];

				if(reduction_val[tid] > val2)
				{
					reduction_val[tid] = val2;
					reduction_idx[tid] = reduction_idx[val2_idx];
				}
			}
			tid_max >>= 1;

			__syncthreads();
		}
		if(tid == 0)
		{
			list_indexes[reduction_idx[0]]++;

			dest_array[i] = reduction_val[0];
		}

		__syncthreads();
	}
}

#define REDUCTION_SIZE 8
#define REDUCTION_SIZE_BIT_SHIFT 3
#define MAX_ACTIVE_REDUCTIONS ((MAX_NUM_LISTS) / REDUCTION_SIZE)

__device__ void merge_array9(const u32 * const src_array,
								u32 * const dest_array,
								const u32 num_lists,
								const u32 num_elements,
								const u32 tid)
{
	u32 data = src_array[tid];

	const u32 s_idx = tid >> REDUCTION_SIZE_BIT_SHIFT;

	const u32 num_reductions = num_lists >> REDUCTION_SIZE_BIT_SHIFT;
	const u32 num_elements_per_list = (num_elements / num_lists);

	__shared__ u32 list_indexes[MAX_NUM_LISTS];
	list_indexes[tid] = 0;

	for(u32 i=0; i<num_elements; i++)
	{
		__shared__ u32 min_val[MAX_ACTIVE_REDUCTIONS];
		__shared__ u32 min_tid;

		if(tid < num_lists)
		{
			min_val[s_idx] = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}

		__syncthreads();

		atomicMin(&min_val[s_idx], data);

		if(num_reductions > 0)
		{
			__syncthreads();

			if(tid < num_reductions)
			{
				atomicMin(&min_val[0], min_val[tid]);
			}

			__syncthreads();
		}

		if(min_val[0] == data)
		{
			atomicMin(&min_tid, tid);
		}

		__syncthreads();

		if(tid == min_tid)
		{
			list_indexes[tid]++;

			dest_array[i] = data;

			if(list_indexes[tid] < num_elements_per_list)
			{
				data = src_array[tid + (list_indexes[tid] * num_lists)];
			}
			else
			{
				data = 0xFFFFFFFF;
			}
		}
		__syncthreads();
	}
}

void execute_host_functions()
{

}

int cuda_practice_qsort_comp(const void *a, const void *b) {
	const unsigned int *a_int = (const unsigned int *)a;
	const unsigned int *b_int = (const unsigned int *)b;

	if (*a_int < *b_int) {
		return -1;
	} else if (*a_int > *b_int) {
		return 1;
	} else {
		return 0;
	}
}

void execute_gpu_functions()
{
	u32 *d = NULL;
	unsigned int idata[NUM_ELEMENTS];
	unsigned int odata[NUM_ELEMENTS];
	unsigned int expected_data[NUM_ELEMENTS];
	srand(time(NULL));
	for (unsigned int i = 0; i < NUM_ELEMENTS; i++){
		idata[i] = rand() % NUM_ELEMENTS;
	}

	cudaMalloc((void** ) &d, sizeof(unsigned int) * NUM_ELEMENTS);
	
	cudaMemcpy(d, idata, sizeof(unsigned int) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

	//Call GPU kernels
	gpu_sort_array_array<<<1, MAX_NUM_LISTS>>>(d);

	cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	checkError();
	
	cudaMemcpy(odata, d, sizeof(int) * NUM_ELEMENTS, cudaMemcpyDeviceToHost);

	memcpy(expected_data, idata, sizeof(unsigned int) * NUM_ELEMENTS);
	qsort(expected_data, NUM_ELEMENTS, sizeof(unsigned int), cuda_practice_qsort_comp);

	for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%3u: Input value: %3u, device output: %3u, expected: %3u\n",
				i, idata[i], odata[i], expected_data[i]);
	}


	cudaFree((void* ) d);
	cudaDeviceReset();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	//execute_host_functions();
	execute_gpu_functions();

	return 0;
}

// vim: noet: tabstop=4: cindent
