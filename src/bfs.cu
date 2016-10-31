#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"

#define SOURCE_VERTEX 3

// print info about bfs values
void print_bfs_values(const int * const values, int const size) {
    int visited = 0;
    int step = 0;
    int first = 0;
    // get the max step and count the visited
    for (int i = 0; i < size; i++) {
        if (values[i] != 0) {
            visited++;
            if (values[i] > step) step = values[i];
            if (values[i] == 1) first = i;
        }
    }
    // count vertices of each step
    if (step == 0) return;
    int * m = (int *) calloc(step + 1, sizeof(int));
    for (int i = 0; i < size; i++) {
        m[values[i]]++;
    }
    // print result info
    printf("\tSource = %d, Step = %d, Visited = %d\n", first, step, visited);
    printf("\tstep\tvisit\n");
    for (int i = 1; i <= step; i++) {
        printf("\t%d\t%d\n", i, m[i]);
    }
    free(m);
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

static void cpu_bfs_vertex(
    const int vertex_num,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values,
    const int first_vertex
) {

    timer_start();
    // for simplicity, use a large but simple queue instead of a small full functional queue)
    int * queue = (int *)calloc(vertex_num, sizeof(int));
    if (queue == NULL) {
        perror("Out of memory");
        exit(1);
    }
    // the position to put next enqueue element & get next dequeue element
    int incount = 0;
    int outcount = 0;
    // initialization
    memset(values, 0, vertex_num * sizeof(int));
    values[first_vertex] = 1;
    queue[incount++] = first_vertex;

    int step = 0;
    while (incount > outcount) {
        // dequeue the vertex to be visited
        int v = queue[outcount++];
        step = values[v];
        for (int e = vertex_begin[v]; e < vertex_begin[v + 1]; e++) {
            int dest = edge_dest[e];
            if (values[dest] == 0) {
                // enqueue the vertex will be visited
                values[dest] = step + 1;
                queue[incount++] = dest;
            }
        }
    }
    printf("\t%.2f\tcpu_bfs_vertex\tStep=%d\n", timer_stop(), step);
    free(queue);
}

static void cpu_bfs_edge(
    const int edge_num,
	const int vertex_num,
    const int * const edge_src,
    const int * const edge_dest,
    int * const values,
    const int first_vertex
) {

    timer_start();
    // for simplicity, use a large but simple queue instead of a small full functional queue)
    int * queue = (int *)calloc(vertex_num, sizeof(int));
    if (queue == NULL) {
        perror("Out of memory");
        exit(1);
    }
    // the position to put next enqueue element & get next dequeue element
    int incount = 0;
    int outcount = 0;
    // initialization
    memset(values, 0, vertex_num * sizeof(int));
    values[first_vertex] = 1;
    queue[incount++] = first_vertex;

    int step = 0;
    while (incount > outcount) {
        // dequeue the vertex to be visited
        int v = queue[outcount++];
        step = values[v];
        for (int e=0; e < edge_num; e++) {
            int src = edge_src[e];
            int dest= edge_dest[e];
            if (src ==v && values[dest] == 0) {
                // enqueue the vertex will be visited
                values[dest] = step + 1;
                queue[incount++] = dest;
            }
        }
    }
    printf("\t%.2f\tcpu_bfs_edge\tStep=%d\n", timer_stop(), step);
    free(queue);
}

// BFS kernel run on edges with inner loop
static __global__ void kernel_edge_loop(
    const int edge_num,
    const int * const edge_src,
    const int * const edge_dest,
    int * const values,
    const int step,
    int * const continue_flag
) {
    // total thread number & thread index of this thread
    int n = blockDim.x * gridDim.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // step counter
    int curStep = step;
    int nextStep = curStep + 1;
    // continue flag for each thread
    int flag = 0;
    // proceeding loop
    for (int i = index; i < edge_num; i += n) {
        if (values[edge_src[i]] == curStep && values[edge_dest[i]] == 0) {
            values[edge_dest[i]] = nextStep;
            flag = 1;
        }
    }
    // update flag
    if (flag == 1) *continue_flag = 1;
}

// BFS kernel run on edges without inner loop
static __global__ void kernel_edge(
        const int edge_num,
        const int * const edge_src,
        const int * const edge_dest,
        int * const values,
        const int step,
        int * const continue_flag
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // step counter
    int curStep = step;
    int nextStep = curStep + 1;
    // proceeding loop
    if(index < edge_num) {
        if (values[edge_src[index]] == curStep && values[edge_dest[index]] == 0) {
            values[edge_dest[index]] = nextStep;
            *continue_flag=1;
		}
	}
}
// BFS algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_bfs_edge(
		const Graph * const g,
		int * const values,
		int const first_vertex
		) {
	int step = 1, flag = 0;
	float bfsTime = 0.0;
	timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    Auto_Utility();
    // Allocate GPU buffer
    CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferZero(int, dev_value, vertex_num);
    CudaBufferZero(int, dev_continue_flag, 1)
    // Set Source Vertex Value (Little Endian)
    CudaMemset(dev_value + first_vertex, 1, 1);
    // Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch Kernel for this Iteration
		CudaTimerBegin();
		kernel_edge<<<(edge_num + 255) / 256, 256>>>
			(
			 edge_num,
			 dev_edge_src,
			 dev_edge_dest,
			 dev_value,
			 step,
			 dev_continue_flag
			);
		bfsTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tgpu_bfs_edge\tstep=%d\t", bfsTime, timer_stop(), step - 1);
}

// BFS algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_bfs_edge_loop(
    const Graph * const g,
    int * const values,
    int const first_vertex
) {
    int step = 1, flag = 0;
    float bfsTime = 0.0;
    timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    Auto_Utility();
    // Allocate GPU buffer
    CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferZero(int, dev_value, vertex_num);
    CudaBufferZero(int, dev_continue_flag, 1)
    // Set Source Vertex Value (Little Endian)
    CudaMemset(dev_value + first_vertex, 1, 1);
    // Main Loop
    do {
        // Clear Flag
        CudaMemset(dev_continue_flag, 0, sizeof(int));
        // Launch Kernel for this Iteration
        CudaTimerBegin();
        kernel_edge_loop<<<208, 128>>>
        (
            edge_num,
            dev_edge_src,
            dev_edge_dest,
            dev_value,
            step,
            dev_continue_flag
        );
        bfsTime += CudaTimerEnd();
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tgpu_bfs_edge_loop\tstep=%d\t", bfsTime, timer_stop(), step - 1);
}



// BFS kernel run on vertices with inner loop
static __global__ void kernel_vertex_loop(
    const int vertex_num,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values,
    const int step,
    int * const continue_flag
) {
    // total thread number & thread index of this thread
    int n = blockDim.x * gridDim.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // step counter
    int curStep = step;
    int nextStep = curStep + 1;
    // continue flag for each thread
    int flag = 0;
    // proceeding loop
    for (int i = index; i < vertex_num; i += n) {
        if (values[i] == curStep) {
            for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
                int dest = edge_dest[k];
                if (values[dest] == 0) {
                    values[dest] = nextStep;
                    flag = 1;
                }
            }
        }
    }
    if (flag) *continue_flag = 1;
}

// BFS kernel run on vertices without inner loop
static __global__ void kernel_vertex(
    const int vertex_num,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values,
    const int step,
    int * const continue_flag
) {
    // total thread number & thread index of this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // proceed
    if (i < vertex_num) {
        if (values[i] == step) {
            for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
                int dest = edge_dest[k];
                if (values[dest] == 0) {
                    values[dest] = step + 1;
                    *continue_flag = 1;
                }
            }
        }
    }
}

// BFS algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_bfs_vertex(
    const Graph * const g,
    int * const values,
    int const first_vertex
) {
    int step = 1, flag = 0;
    float bfsTime = 0.0;
    timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    Auto_Utility();
    // Allocate GPU buffer
    CudaBufferCopy(int, dev_vertex_begin, vertex_num + 1, g->vertex_begin);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferZero(int, dev_value, vertex_num);
    CudaBufferZero(int, dev_continue_flag, 1)
    // Set Source Vertex Value (Little Endian)
    CudaMemset(dev_value + first_vertex, 1, 1);
    // Main Loop
    do {
        // Clear Flag
        CudaMemset(dev_continue_flag, 0, sizeof(int));
        // Launch Kernel for this Iteration
        CudaTimerBegin();
        kernel_vertex<<<(vertex_num + 255) / 256, 256>>>
        (
            vertex_num,
            dev_vertex_begin,
            dev_edge_dest,
            dev_value,
            step,
            dev_continue_flag
        );
        bfsTime += CudaTimerEnd();
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tgpu_bfs_vertex\tstep=%d\t", bfsTime, timer_stop(), step - 1);
}

// BFS algorithm on graph g, not partitioned, run on vertices with inner loop
static void gpu_bfs_vertex_loop(
    const Graph * const g,
    int * const values,
    int const first_vertex
) {
    int step = 1, flag = 0;
    float bfsTime = 0.0;
    timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    Auto_Utility();
    // Allocate GPU buffer
    CudaBufferCopy(int, dev_vertex_begin, vertex_num + 1, g->vertex_begin);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferZero(int, dev_value, vertex_num);
    CudaBufferZero(int, dev_continue_flag, 1)
    // Set Source Vertex Value (Little Endian)
    CudaMemset(dev_value + first_vertex, 1, 1);
    // Main Loop
    do {
        // Clear Flag
        CudaMemset(dev_continue_flag, 0, sizeof(int));
        // Launch Kernel for this Iteration
        CudaTimerBegin();
        kernel_vertex_loop<<<208, 256>>>
        (
            vertex_num,
            dev_vertex_begin,
            dev_edge_dest,
            dev_value,
            step,
            dev_continue_flag
        );
        bfsTime += CudaTimerEnd();
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tgpu_bfs_vertex_loop\tstep=%d\t", bfsTime, timer_stop(), step - 1);
}


// experiments of BFS on Graph g with Partition Table t and partitions
void bfs_experiments(const Graph * const g) {

    // partition on the Graph
    printf("-------------------------------------------------------------------\n");

    int * value_cpu = (int *) calloc(g->vertex_num, sizeof(int));
    int * value_gpu = (int *) calloc(g->vertex_num, sizeof(int));
    if (value_cpu == NULL || value_gpu == NULL) {
        perror("Out of Memory for values");
        exit(1);
    }

    printf("\tTime\tTotal\tTips\n");
    
    cpu_bfs_vertex(g->vertex_num, g->vertex_begin, g->edge_dest, value_cpu, SOURCE_VERTEX);
 //   print_bfs_values(value_cpu, g->vertex_num);

//    cpu_bfs_edge(g->edge_num,g->vertex_num, g->edge_src, g->edge_dest, value_cpu, SOURCE_VERTEX);
 // print_bfs_values(value_cpu, g->vertex_num);
 
    gpu_bfs_edge(g, value_gpu, SOURCE_VERTEX);
    check_values(value_cpu, value_gpu, g->vertex_num);
    gpu_bfs_edge_loop(g, value_gpu, SOURCE_VERTEX);
    check_values(value_cpu, value_gpu, g->vertex_num);

    gpu_bfs_vertex(g, value_gpu, SOURCE_VERTEX);
    check_values(value_cpu, value_gpu, g->vertex_num);
    gpu_bfs_vertex_loop(g, value_gpu, SOURCE_VERTEX);
    check_values(value_cpu, value_gpu, g->vertex_num);
    
    free(value_cpu);
    free(value_gpu);
}


