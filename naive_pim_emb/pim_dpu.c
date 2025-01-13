#include <alloc.h>
#include <stdlib.h>
#include <mram.h>
#include <stdint.h>
#include <defs.h>

// Tasklet
#define TASKLET_NUM 16

// WRAM Management

#define STACK_SIZE (256)
#define WRAM_BUFFER_SIZE (1024 * 3 - STACK_SIZE)

uint8_t *wram_buffer;

// MRAM Management

#define MRAM_METADATA_SIZE (128)
#define MRAM_BUFFER_SIZE (64 * 1024 * 1024 - MRAM_METADATA_SIZE)

__mram uint64_t emb_dim;
__mram uint32_t num_emb;
__mram uint8_t buffer[MRAM_BUFFER_SIZE];

void load_emb()
{
    num_emb = buffer[1];
    int tasklet_id = me();
    uint32_t num_per_tasklet = num_emb / TASKLET_NUM;
    uint32_t start = num_per_tasklet * tasklet_id;
    uint32_t end = start + num_per_tasklet;

    float *emb_base = (float *)(buffer + 16 * 1024 * 1024);
    float *input_start_pos = ((float *)buffer + 1 + start * emb_dim);
    float *input_end_pos = ((float *)buffer + 1 + end * emb_dim);
    float *emb_start_pos = emb_base + start * emb_dim;

    while (input_start_pos < input_end_pos)
    {
        uint64_t trans_size = (input_end_pos - input_start_pos) * 4;
        if (trans_size > WRAM_BUFFER_SIZE) trans_size = WRAM_BUFFER_SIZE;
        mram_read((__mram_ptr void *)input_end_pos, wram_buffer, trans_size);
        mram_write(wram_buffer, (__mram_ptr void *)emb_start_pos, trans_size);
        input_start_pos += trans_size / 4;
        emb_start_pos += trans_size / 4;
    }
}

void apply_emb()
{
    
}

int main()
{
    buddy_init(WRAM_BUFFER_SIZE);
    wram_buffer = buddy_alloc(WRAM_BUFFER_SIZE);

    mram_read(buffer, wram_buffer, 8);
    uint32_t fun_type = *((uint32_t *)wram_buffer);
    if (fun_type == 0) load_emb();
    else if (fun_type == 1) apply_emb();

    buddy_free(wram_buffer);
    return 0;
}