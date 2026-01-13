enum OpCode {
  //%extra_op_codes%
};

//%extra_op_structs%

union Payload {
  //%extra_op_payloads%
};

struct Task {
  OpCode op;
  int range;
  int remaining;
  int in_dep_a_stride;
  int in_dep_a_base;
  int in_dep_b_stride;
  int in_dep_b_base;
  int in_dep_c_stride;
  int in_dep_c_base;
  int out_dep_stride;
  int out_dep_base;
  const float *source_ptrs[3];
  float *out_ptr;
  Payload payload;
};

struct SMEvent {
  unsigned long long start;
  unsigned long long stop;
  int event;
};

//%constants%

__device__ __noinline__ int eval_expression(int expression, int const_z) {
  switch (expression) {
    //%expr_fns%
  }
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

//%extra_op_functions%

__device__ __forceinline__ void nanosleep(unsigned int cycles) {
  asm volatile("nanosleep.u32 %0;" ::"r"(cycles));
}

__device__ inline void mutex_lock(int *m) {
  while (atomicCAS(m, 0, 1) != 0) {
    nanosleep(64);
  }
  __threadfence();
}
__device__ inline void mutex_unlock(int *m) {
  __threadfence();
  atomicExch(m, 0);
}

struct NextTask {
  int current;
  int task_idx;
};
__device__ inline bool fetch_next_task(Task *tasks, int num_tasks, int *head,
                                       NextTask *out, int *queue_lock) {
  mutex_lock(queue_lock);

  int idx = *head;
  if (idx >= num_tasks) {
    mutex_unlock(queue_lock);
    return false;
  }

  out->task_idx = idx;

  // Check if we need to reset this remaining counter (-1 signals there are no
  // remaining tasks here to run)
  if (tasks[idx].remaining == -1) {
    tasks[idx].remaining = eval_expression(tasks[idx].range, 0) - 1;
  }

  out->current = tasks[idx].remaining;

  // Decrement the remaining count
  tasks[idx].remaining--;

  // If this is the last iteration in this task, advance the head
  if (out->current == 0) {
    atomicAdd(head, 1);
  }

  mutex_unlock(queue_lock);
  return true;
}

__device__ inline void record_event(SMEvent *__restrict__ timings, int *event_idx, int event_type) {
  if (*event_idx < 1000) {
    unsigned long long now = read_globaltimer();
    if (*event_idx > 0) { // record the end of the previous op
      timings[*event_idx - 1].stop = now;
    }
    timings[*event_idx].start = now;
    timings[*event_idx].stop = 0ull;
    timings[*event_idx].event = event_type;
    (*event_idx)++;
  }
}

extern "C" {
__global__ void worker_kernel(Task *__restrict__ tasks, int num_tasks,
                              int *__restrict__ head, int *__restrict__ ready,
                              int *__restrict__ queue_lock,
                              SMEvent *__restrict__ timings,
                              unsigned long long *__restrict__ start_times) {
  __shared__ NextTask nt;
  __shared__ int done;
  int dep_out = 0;
  int recorded_event = 0;
  timings += blockIdx.x * 1000;
  if (threadIdx.x == 0) {
    start_times[blockIdx.x] = read_globaltimer();
  }
  while (true) {
    if (threadIdx.x == 0) {
      done = !fetch_next_task(tasks, num_tasks, head, &nt, queue_lock);
    }
    __syncthreads();
    if (done)
      break;

    const Task *t = &tasks[nt.task_idx];

    // Check for input dependencies
    if (threadIdx.x == 0) {
      __threadfence();

      record_event(timings, &recorded_event, 0); // Record issue start

      int dep_a = eval_expression(t->in_dep_a_base, 0) +
                  eval_expression(t->in_dep_a_stride, nt.current);
      int dep_b = eval_expression(t->in_dep_b_base, 0) +
                  eval_expression(t->in_dep_b_stride, nt.current);
      int dep_c = eval_expression(t->in_dep_c_base, 0) +
                  eval_expression(t->in_dep_c_stride, nt.current);
      dep_out = eval_expression(t->out_dep_base, 0) +
                eval_expression(t->out_dep_stride, nt.current);
      // Increment the output barrier to signal an op is in-flight
      // TODO: This should be done while the mutex is still held. Technically
      // there is a bit of time where we can have a race condition!
      atomicAdd(&ready[dep_out], 1);
      record_event(timings, &recorded_event, 1); // Record wait start

      // Wait on input dependencies
      while (atomicAdd(&ready[dep_a], 0) > 0)
        nanosleep(64);
      while (atomicAdd(&ready[dep_b], 0) > 0)
        nanosleep(64);
      while (atomicAdd(&ready[dep_c], 0) > 0)
        nanosleep(64);

      __threadfence();
      record_event(timings, &recorded_event, t->op + 2); // Record op start
    }
    __syncthreads();
    switch (t->op) {
      //%extra_op_calls%
    }
    __syncthreads();
    // Arrive at output barrier
    if (threadIdx.x == 0) {
      __threadfence();
      atomicSub(&ready[dep_out], 1);
    }
  }

  if (threadIdx.x == 0 && recorded_event > 0) {
    timings[recorded_event - 1].stop = read_globaltimer();
  }
}
}
