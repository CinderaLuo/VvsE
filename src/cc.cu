#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"
#include "src/vwarp.cu"

// print info about CC values
void report_cc_values(const int * const values, int n) {
	int * c = (int *)calloc(n, sizeof(int));
	int cc = 0;
	for (int i = 0; i < n; i++) {
		int r = values[i];
		if (c[r] == 0) cc++;
		c[r]++;
	}
	printf("Number of Connected Components: %d\n", cc);
	int k = 0;
	printf("\tID\tRoot\tN\n");
	for (int i = 0; i < n; i++) {
		if (c[i] != 0)
			printf("\t%d\t%d\t%d\n", k++, i, c[i]);
		if (k > 20) {
			printf("\t...\n");
			break;
		}
	}
	free(c);
}

// check if arrays v1 & v2 have the same first n elements (no boundary check)
static void check_values(const int * const v1, const int * const v2, int n) {
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i]) {
			printf("Check Fail\n");
			return;
		}
	}
	printf("Check PASS\n");
}

// MFset function - Reset
static void Reset(int s[], int n) {
	for (int i = 0; i < n; i++) s[i] = i;
}
// MFset function - Find
static int Find(int s[], int x) {
	int root = x;
	// find the root
	while (s[root] != root)
		root = s[root];
	// merge the path
	while (s[x] != root) {
		int t = s[x];
		s[x] = root;
		x = t;
	}
	// return root
	return root;
}
// MFset function - Union
static void Union(int s[], int x, int y) {
	int root1 = Find(s, x);
	int root2 = Find(s, y);
	// keep the smaller root
	if (root1 < root2)
		s[root2] = root1;
	else if (root1 > root2)
		s[root1] = root2;
}
// CC on CPU
static void cpu_cc_vertex_uf(
		const int vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values
		) {
	timer_start();
	// Initializing values
	Reset(values, vertex_num);
	// MFset calculating
	for (int i = 0; i < vertex_num; i++)
		for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++)
			Union(values, i, edge_dest[k]);
	// Calculating final values
	for (int i = 0; i < vertex_num; i++)
		Find(values, i);
	printf("\t%.2f\tcpu_cc_vertex_uf\n", timer_stop());
}


static void cpu_cc_vertex(
		const int vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values)
{
	timer_start();
	Reset(values,vertex_num);
	int flag=1;
	int *farray=(int *)calloc(vertex_num, sizeof(int));
	if(farray== NULL)
	{
		perror("Out of Memory");
		exit(1);
	}
	memset(farray,0,vertex_num*sizeof(int));
	int step=1;
	do
	{

		memset(farray,0,vertex_num*sizeof(int));
		for (int i = 0; i < vertex_num; ++i)
		{
			for (int k=vertex_begin[i];k<vertex_begin[i+1];k++)
			{
				if(values[i]!=values[edge_dest[k]])
				{
					farray[i]=1;
					if(values[i]<values[edge_dest[k]])
						values[edge_dest[k]]=values[i];
					else
						values[i]=values[edge_dest[k]];
				}
			}
		}
		int m;
		for (m= 0; m < vertex_num; ++m)
		{
			if(farray[m]==1)
				break;
		}
		if(m==vertex_num)
			flag=0;
		step++;
	}while(flag==1);

	printf("\t%.2f\tcpu_cc_vertex\n", timer_stop());
}

static  void cpu_cc_edge(
		const int edge_num,
		const int vertex_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values)
{
	timer_start();
	Reset(values,vertex_num);
	timer_start();
	Reset(values,vertex_num);
	int flag=1;
	int *farray=(int *)calloc(vertex_num, sizeof(int));
	memset(farray,0,sizeof(int)*vertex_num);
	int ite=0;
	while(flag==1)
	{
		memset(farray,0,sizeof(int)*vertex_num);
		for (int i = 0; i < edge_num; ++i)
		{
			int src=edge_src[i];
			int  dest=edge_dest[i];
			if(values[src]!=values[dest])
			{
				farray[src]=1;
				if(values[src]<values[dest])
					values[src]=values[dest];
				else
					values[src]=values[dest];
			}
		}
		int m;
		for (m= 0; m < vertex_num; ++m)
		{
			if(farray[m]==1)
				break;
		}
		if(m==vertex_num)
			flag=0;
	}
	free(farray);
	ite++;
}
// CC Kernel on edges without inner loops
static __global__ void kernel_edge (
		int const edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	int i=index;
	// proceeding loop
	if( i< edge_num) {
		int src = edge_src[i];
		int dest = edge_dest[i];
		if (values[src] != values[dest]) {
			flag = 1;
			// combined to the smaller Component ID
			if (values[src] > values[dest])
				values[src] = values[dest];
			else
				values[dest] = values[src];
		}
	}
	if (flag == 1) *continue_flag = 1;
}
// CC Kernel on edges with inner loops
static __global__ void kernel_edge_loop (
		int const edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	// proceeding loop
	for (int i = index; i < edge_num; i += n) {
		int src = edge_src[i];
		int dest = edge_dest[i];
		if (values[src] != values[dest]) {
			flag = 1;
			// combined to the smaller Component ID
			if (values[src] > values[dest])
				values[src] = values[dest];
			else
				values[dest] = values[src];
		}
	}
	if (flag == 1) *continue_flag = 1;
}

// CC algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_cc_edge(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (edge_num+255)/256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_edge<<<bn, tn>>>(
				edge_num,
				dev_edge_src,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_edge\tStep=%d\t",
			execTime, timer_stop(), step);
}

// CC algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_cc_edge_loop(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = 204;
	int tn = 128;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_edge_loop<<<bn, tn>>>(
				edge_num,
				dev_edge_src,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_edge_loop\tStep=%d\t",
			execTime, timer_stop(), step);
}

static __global__ void kernel_vertex (
		int const vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// thread index of this thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// proceed
	if (i < vertex_num) {
		int new_value = values[i];
		int flag = 0;
		// find the best new_value (smallest)
		for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
			int dest_value = values[edge_dest[e]];
			if (dest_value != new_value)
				flag = 1;
			if (dest_value < new_value)
				new_value = dest_value;
		}
		// update values
		if (flag) {
			values[i] = new_value;
			for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
				values[edge_dest[e]] = new_value;
			}
			*continue_flag = 1;
		}
	}
}

static __global__ void kernel_vertex_loop (
		int const vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	// proceeding loop
	// proceed
	for (int i=index;i < vertex_num;i=i+n) {
		int new_value = values[i];
		int flag = 0;
		// find the best new_value (smallest)
		for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
			int dest_value = values[edge_dest[e]];
			if (dest_value != new_value)
				flag = 1;
			if (dest_value < new_value)
				new_value = dest_value;
		}
		// update values
		if (flag) {
			values[i] = new_value;
			for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
				values[edge_dest[e]] = new_value;
			}
			*continue_flag = 1;
		}
	}
}

// virtual warp by myself 
static __global__ void kernel_vertex_vwarp (
		int const vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
#define VWARP_SIZE 2
	// thread index of this thread
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int n=blockDim.x*gridDim.x;
	int vwarp_num = n / VWARP_SIZE;
	int vwarp_id = index / VWARP_SIZE;
	int vwarp_tid = index % VWARP_SIZE;
	// offset of each vwarp for cache indexing (vwarp_offset + vwarp_tid == threadIdx.x)
	int vwarp_offset = threadIdx.x / VWARP_SIZE * VWARP_SIZE;
	// proceed
	for(int i=vwarp_id;i<vertex_num;i+=vwarp_num) {
		int new_value = values[i];
		// find the best new_value (smallest)
		for (int e = vertex_begin[i]+vwarp_tid; e < vertex_begin[i + 1]; e+=VWARP_SIZE) {
			int dest_value = values[edge_dest[e]];
			if(dest_value !=new_value)
			{
				*continue_flag = 1;
				if (dest_value > new_value)           
					values[edge_dest[e]] =values[i];
				if (dest_value < new_value)
					values[i] = values[edge_dest[e]];
			}    

		}
	}

}

//virtual warp similar to Totem
template<int VWARP_WIDTH1,int VWARP_BATCH1>
static  __global__ void kernel_vertex_vwarp_totem (
		int const vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	if(THREAD_GLOBAL_INDEX>=
			vwarp_thread_count(vertex_num,VWARP_WIDTH1,VWARP_BATCH1)){return ;}
	__shared__ bool finish_block;
	finish_block=true;
	__syncthreads();

	int start_vertex=vwarp_block_start_vertex(VWARP_WIDTH1,VWARP_BATCH1)+
		vwarp_warp_start_vertex(VWARP_WIDTH1,VWARP_BATCH1);
	int end_vertex=start_vertex+
		vwarp_warp_batch_size(vertex_num,VWARP_WIDTH1,VWARP_BATCH1);
	int warp_offset=vwarp_thread_index(VWARP_WIDTH1);
	// proceed
	for(int i=start_vertex;i<end_vertex;i++) {
		int new_value = values[i];
		// find the best new_value (smallest)
		const int nbr_count=vertex_begin[i+1]-vertex_begin[i];
		const int *edge=edge_dest+vertex_begin[i];
		for (int e =warp_offset; e < nbr_count; e+=VWARP_WIDTH1)
		{
			int nbr=edge[e];
			int dest_value = values[nbr];
			if(dest_value!=new_value)
			{
				finish_block=false;
				if (dest_value < new_value) 
					values[i]=values[nbr];
				if (dest_value > new_value)
					values[nbr]=values[i];
			}   

		}
	}
	//__syncthreads();
	//if(!finish_block&&THREAD_GLOBAL_INDEX==0) *continue_flag=1;
	if(!finish_block) *continue_flag=1;
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_vertex<<<bn, tn>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_vertex\tStep=%d\t",
			execTime, timer_stop(), step);
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex_vwarp(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;

	const int threads=MAX_THREADS_PER_BLOCK;
	dim3 blocks;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		kernel_configure(vwarp_thread_count(g->vertex_num,VWARP_WIDTH,BATCH_SIZE),
				blocks,threads);
		CudaTimerBegin();
		kernel_vertex_vwarp<<<blocks,threads>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_vertex_vwarp\tStep=%d\t",
			execTime, timer_stop(), step);
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex_vwarp_totem(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;

	const int threads=MAX_THREADS_PER_BLOCK;
	dim3 blocks;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		kernel_configure(vwarp_thread_count(g->vertex_num,VWARP_WIDTH,BATCH_SIZE),
				blocks,threads);
		CudaTimerBegin();
		kernel_vertex_vwarp_totem<VWARP_WIDTH,BATCH_SIZE><<<blocks,threads>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_vertex_vwarp_totem\tStep=%d\t",
			execTime, timer_stop(), step);
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex_loop(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_vertex_loop<<<bn, tn>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 1000);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_cc_vertex_loop\tStep=%d\t",
			execTime, timer_stop(), step);
}



// experiments of BFS on Graph g with Partition Table t and partitions
void cc_experiments(const Graph * const g) {

	printf("-------------------------------------------------------------------\n");

	int * value_cpu = (int *) calloc(g->vertex_num, sizeof(int));
	int * value_gpu = (int *) calloc(g->vertex_num, sizeof(int));
	if (value_cpu == NULL || value_gpu == NULL) {
		perror("Out of Memory for values");
		exit(1);
	}

	printf("\tTime\tTotal\tTips\n");

	cpu_cc_vertex_uf(g->vertex_num, g->vertex_begin, g->edge_dest, value_cpu);
	// report_cc_values(value_cpu, g->vertex_num);

	//    cpu_cc_edge(g->edge_num,g->vertex_num,g->edge_src,g->edge_dest,value_cpu);
	cpu_cc_vertex(g->vertex_num,g->vertex_begin,g->edge_dest,value_cpu);

	gpu_cc_edge(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);
	gpu_cc_edge_loop(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);


	gpu_cc_vertex(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);

	gpu_cc_vertex_vwarp(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);
	gpu_cc_vertex_vwarp_totem(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);

	gpu_cc_vertex_loop(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);


	free(value_cpu);
	free(value_gpu);
}

