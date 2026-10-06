/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MEM_H__
#define __MEM_H__

#include <stddef.h>

extern struct pi_device ram;

void open_fs();
void mem_init();
struct pi_device *get_ram_ptr();

// Request slots per tensor/direction for asynchronous L3 copies (see Deeploy/Targets/GAP9/DMA/L3Dma.py).
#ifndef GAP9_L3_REQ_SLOTS
#define GAP9_L3_REQ_SLOTS 3
#endif
// PSRAM page-wrap guard (see gap9_ram_copy_2d in dory_mem.c). 1 KB is stricter than the
// 2 KB wrap measured on the EVK; every 2 KB boundary is also a 1 KB one.
#define GAP9_RAM_PAGE 1024u
#define GAP9_RAM_MIN_HEAD 8u
#include "pmsis.h"
int gap9_ram_copy_2d(uint32_t ext, void *loc, uint32_t size, uint32_t stride, uint32_t len, int ext2loc,
                     pi_cl_ram_req_t *req);
void *ram_malloc(size_t size);
void ram_free(void *ptr, size_t size);
void ram_read(void *dest, void *src, size_t size);
void ram_write(void *dest, void *src, size_t size);
void *cl_ram_malloc(size_t size);
void cl_ram_free(void *ptr, size_t size);
void cl_ram_read(void *dest, void *src, size_t size);
void cl_ram_write(void *dest, void *src, size_t size);
size_t load_file_to_ram(const void *dest, const char *filename);
size_t load_file_to_local(const void *dest, const char *filename);

#endif // __MEM_H__
