#include <alloc.h>
#include <stdlib.h>
#include <mram.h>
#include <stdint.h>

// WRAM Management

#define WRAM_BUFFER_SIZE (4096)

// MRAM Management

#define MRAM_BUFFER_SIZE (64 * 1024 * 1024)

__mram uint8_t buffer[MRAM_BUFFER_SIZE];

int main()
{
    buddy_init(WRAM_BUFFER_SIZE);
    uint8_t *wram_buffer = buddy_alloc(32);
    mram_read(buffer, wram_buffer, 16);
    float *a = (float *)wram_buffer;
    float *b = a + 1;
    float *c = b + 1;
    (*c) = *(a) + *(b);
    mram_write(wram_buffer, buffer, 16);
    buddy_free(wram_buffer);
    return 0;
}