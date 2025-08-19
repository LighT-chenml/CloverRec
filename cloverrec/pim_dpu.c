#include <alloc.h>
#include <stdlib.h>
#include <mram.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <vmutex.h>

// Tasklet
#define TASKLET_NUM 16

BARRIER_INIT(my_barrier, TASKLET_NUM);

// MRAM Management

#define MRAM_METADATA_SIZE (1024 * 1024 * 4)
#define MRAM_BUFFER_SIZE (64 * 1024 * 1024 - MRAM_METADATA_SIZE)

__mram uint64_t emb_dim;
__mram uint8_t buffer[MRAM_BUFFER_SIZE];

VMUTEX_INIT(my_vmutex, (1 << 8) * 10, 1);

void apply_emb()
{
    int tasklet_id = me();

    uint32_t num_indices = *((__mram_ptr uint32_t *)buffer + 1);
    uint32_t num_index_groups = *((__mram_ptr uint32_t *)buffer + 2);

    uint32_t num_group_per_tasklet = (num_index_groups  + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start_group = num_group_per_tasklet * tasklet_id;
    uint32_t end_group = start_group + num_group_per_tasklet;
    if (end_group > num_index_groups)
        end_group = num_index_groups;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 2 * 1024 * 1024);
    __mram_ptr float *result_base = (__mram_ptr float *)(buffer + 1 * 1024 * 1024);

    for (int i = start_group; i < end_group; ++i)
    {
        __mram_ptr float *sum = result_base + i * emb_dim;
        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) = 0.0;
    }

    barrier_wait(&my_barrier);

    for (int i = 0; i < num_indices; ++i)
    {
        uint32_t gid = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2);
        uint32_t index = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2 + 1);
        uint32_t result_index = *((__mram_ptr uint32_t *)buffer + 4 + gid);

        if (result_index >= start_group && result_index < end_group)
        {
            __mram_ptr float *ev = emb_base + index * emb_dim;
            __mram_ptr float *sum = result_base + result_index * emb_dim;

            for (int j = 0; j < emb_dim; ++j)
                *(sum + j) += *(ev + j);
        }
    }
}

void apply_emb_dim_parallel()
{
    int tasklet_id = me();

    uint32_t num_indices = *((__mram_ptr uint32_t *)buffer + 1);
    uint32_t num_index_groups = *((__mram_ptr uint32_t *)buffer + 2);

    uint32_t num_dims_per_tasklet = (emb_dim + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start_dim = num_dims_per_tasklet * tasklet_id;
    uint32_t end_dim = start_dim + num_dims_per_tasklet;
    if (end_dim > emb_dim)
        end_dim = emb_dim;

    uint32_t num_group_per_tasklet = (num_index_groups  + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start_group = num_group_per_tasklet * tasklet_id;
    uint32_t end_group = start_group + num_group_per_tasklet;
    if (end_group > num_index_groups)
        end_group = num_index_groups;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 2 * 1024 * 1024);
    __mram_ptr float *result_base = (__mram_ptr float *)(buffer + 1 * 1024 * 1024);

    for (int i = start_group; i < end_group; ++i)
    {
        __mram_ptr float *sum = result_base + i * emb_dim;
        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) = 0.0;
    }

    barrier_wait(&my_barrier);

    for (int i = 0; i < num_indices; ++i)
    {
        uint32_t gid = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2);
        uint32_t index = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2 + 1);
        uint32_t result_index = *((__mram_ptr uint32_t *)buffer + 4 + gid);

        __mram_ptr float *ev = emb_base + index * emb_dim;
        __mram_ptr float *sum = result_base + result_index * emb_dim;

        for (int j = start_dim; j < end_dim; ++j)
            *(sum + j) += *(ev + j);
    }
}

void apply_emb_lock()
{
    int tasklet_id = me();

    uint32_t num_indices = *((__mram_ptr uint32_t *)buffer + 1);
    uint32_t num_index_groups = *((__mram_ptr uint32_t *)buffer + 2);

    uint32_t num_indices_per_tasklet = (num_indices + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start = num_indices_per_tasklet * tasklet_id;
    uint32_t end = start + num_indices_per_tasklet;
    if (end > num_indices)
        end = num_indices;

    uint32_t num_group_per_tasklet = (num_index_groups  + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start_group = num_group_per_tasklet * tasklet_id;
    uint32_t end_group = start_group + num_group_per_tasklet;
    if (end_group > num_index_groups)
        end_group = num_index_groups;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 2 * 1024 * 1024);
    __mram_ptr float *result_base = (__mram_ptr float *)(buffer + 1 * 1024 * 1024);

    for (int i = start_group; i < end_group; ++i)
    {
        __mram_ptr float *sum = result_base + i * emb_dim;
        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) = 0.0;
    }

    barrier_wait(&my_barrier);

    for (int i = start; i < end; ++i)
    {
        uint32_t gid = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2);
        uint32_t index = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2 + 1);
        uint32_t result_index = *((__mram_ptr uint32_t *)buffer + 4 + gid);

        __mram_ptr float *ev = emb_base + index * emb_dim;
        __mram_ptr float *sum = result_base + result_index * emb_dim;

        vmutex_lock(&my_vmutex, result_index);

        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) += *(ev + j);

        vmutex_unlock(&my_vmutex, result_index);
    }
}

void apply_emb_ideal()
{
    int tasklet_id = me();

    uint32_t num_indices = *((__mram_ptr uint32_t *)buffer + 1);
    uint32_t num_index_groups = *((__mram_ptr uint32_t *)buffer + 2);

    uint32_t num_indices_per_tasklet = (num_indices + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start = num_indices_per_tasklet * tasklet_id;
    uint32_t end = start + num_indices_per_tasklet;
    if (end > num_indices)
        end = num_indices;

    uint32_t num_group_per_tasklet = (num_index_groups  + TASKLET_NUM - 1) / TASKLET_NUM;
    uint32_t start_group = num_group_per_tasklet * tasklet_id;
    uint32_t end_group = start_group + num_group_per_tasklet;
    if (end_group > num_index_groups)
        end_group = num_index_groups;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 2 * 1024 * 1024);
    __mram_ptr float *result_base = (__mram_ptr float *)(buffer + 1 * 1024 * 1024);

    for (int i = start_group; i < end_group; ++i)
    {
        __mram_ptr float *sum = result_base + i * emb_dim;
        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) = 0.0;
    }

    barrier_wait(&my_barrier);

    for (int i = start; i < end; ++i)
    {
        uint32_t gid = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2);
        uint32_t index = *((__mram_ptr uint32_t *)buffer + 4 + num_index_groups + i * 2 + 1);
        uint32_t result_index = *((__mram_ptr uint32_t *)buffer + 4 + gid);

        __mram_ptr float *ev = emb_base + index * emb_dim;
        __mram_ptr float *sum = result_base + result_index * emb_dim;

        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) += *(ev + j);
    }
}

int main()
{
    uint32_t fun_type = *((__mram_ptr uint32_t *)buffer);
    if (fun_type == 0)
    {
        apply_emb_dim_parallel();
        // apply_emb();
        // apply_emb_lock();
        // apply_emb_ideal();
    }
    return 0;
}