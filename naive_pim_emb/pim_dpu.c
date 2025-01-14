#include <alloc.h>
#include <stdlib.h>
#include <mram.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>

// Tasklet
#define TASKLET_NUM 16

// WRAM Management

#define STACK_SIZE (256)
#define WRAM_BUFFER_SIZE (2048)

__dma_aligned uint8_t wram_buffer[WRAM_BUFFER_SIZE];

// MRAM Management

#define MRAM_METADATA_SIZE (128 + 1024 * 1024)
#define MRAM_BUFFER_SIZE (64 * 1024 * 1024 - MRAM_METADATA_SIZE)

__mram uint64_t emb_dim;
__mram uint32_t num_emb;
__mram uint8_t buffer[MRAM_BUFFER_SIZE];

void load_emb()
{
    int tasklet_id = me();

    num_emb = (uint32_t)(*((__mram_ptr float *)buffer + 1));

    uint32_t num_emb_per_tasklet = num_emb / TASKLET_NUM;
    uint32_t start = num_emb_per_tasklet * tasklet_id;
    uint32_t end = start + num_emb_per_tasklet;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 16 * 1024 * 1024);
    __mram_ptr float *input_start_pos = ((__mram_ptr float *)buffer + 2 + start * emb_dim);
    __mram_ptr float *input_end_pos = ((__mram_ptr float *)buffer + 2 + end * emb_dim);
    __mram_ptr float *emb_start_pos = emb_base + start * emb_dim;

    while (input_start_pos < input_end_pos)
    {
        uint64_t trans_size = (input_end_pos - input_start_pos) * 4;

        if (trans_size > WRAM_BUFFER_SIZE)
            trans_size = WRAM_BUFFER_SIZE;

        mram_read((__mram_ptr void *)buffer, wram_buffer, trans_size);
        mram_write(wram_buffer, (__mram_ptr void *)emb_start_pos, trans_size);

        input_start_pos += trans_size / 4;
        emb_start_pos += trans_size / 4;
    }
}

void apply_emb()
{
    int tasklet_id = me();

    uint32_t num_indices = *((__mram_ptr uint32_t *)buffer + 1);
    num_indices -= 4;
    num_indices /= 2;

    uint32_t num_index_groups = *((__mram_ptr uint32_t *)buffer + 2);

    uint32_t num_group_per_tasklet = num_index_groups / TASKLET_NUM;
    uint32_t start = num_group_per_tasklet * tasklet_id;
    uint32_t end = start + num_group_per_tasklet;
    if (end > num_index_groups)
        end = num_index_groups;

    __mram_ptr float *emb_base = (__mram_ptr float *)(buffer + 16 * 1024 * 1024);
    __mram_ptr float *result_base = (__mram_ptr float *)(buffer + 8 * 1024 * 1024);

    for (int i = start; i < end; ++i)
    {
        __mram_ptr float *sum = result_base + i * emb_dim;
        for (int j = 0; j < emb_dim; ++j)
            *(sum + j) = 0.0;
    }

    // printf("%d num_indices %u num_index_groups %u num_group_per_tasklet %u start %u end %u\n", tasklet_id, num_indices, num_index_groups, num_group_per_tasklet, start, end);

    for (int i = 0; i < num_indices; ++i)
    {
        uint32_t gid = *((__mram_ptr uint32_t *)buffer + 4 + i * 2);
        if (gid >= start && gid < end)
        {
            uint32_t index = *((__mram_ptr uint32_t *)buffer + 4 + i * 2 + 1);
            __mram_ptr float *ev = emb_base + index * emb_dim;
            __mram_ptr float *sum = result_base + gid * emb_dim;

            // printf("%d gid %u index %u\n", tasklet_id, gid, index);

            for (int j = 0; j < emb_dim; ++j)
                *(sum + j) += *(ev + j);
        }
    }
}

int main()
{
    mram_read(buffer, wram_buffer, 8);
    uint32_t fun_type = *((uint32_t *)wram_buffer);
    if (fun_type == 0)
        load_emb();
    else if (fun_type == 1)
        apply_emb();
    return 0;
}