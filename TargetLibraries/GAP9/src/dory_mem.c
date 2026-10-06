/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dory_mem.h"
#include "bsp/bsp.h"
// #include "bsp/flash.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
// #include "bsp/ram.h"
#include "pmsis.h"

#ifdef USE_HYPERFLASH
#include "bsp/flash/hyperflash.h"
typedef struct pi_hyperflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_hyperflash_conf_init(conf)
#elif defined USE_SPIFLASH
#include "bsp/flash/spiflash.h"
typedef struct pi_spiflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_spiflash_conf_init(conf)
#elif defined USE_MRAM
typedef struct pi_mram_conf flash_conf_t;
#define flash_conf_init(conf) pi_mram_conf_init(conf)
#else
typedef struct pi_default_flash_conf flash_conf_t;
#define flash_conf_init(conf) pi_default_flash_conf_init(conf)
#endif

#ifdef USE_HYPERRAM
#include "bsp/ram/hyperram.h"
typedef struct pi_hyperram_conf ram_conf_t;
#define ram_conf_init(conf) pi_hyperram_conf_init(conf)
#else
typedef struct pi_default_ram_conf ram_conf_t;
#define ram_conf_init(conf) pi_default_ram_conf_init(conf)
#endif

#define BUFFER_SIZE 2048 // 128
static uint8_t buffer[BUFFER_SIZE];

static struct pi_device flash;
static flash_conf_t flash_conf;

static struct pi_device fs;
static struct pi_readfs_conf fs_conf;

struct pi_device ram;
static ram_conf_t ram_conf;

void open_fs() {
  // SCHEREMO: Fix FS
  // Open filesystem on flash.
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) {
    printf("ERROR: Cannot mount filesystem! Exiting...\n");
    pmsis_exit(-2);
  }
}

void mem_init() {
  flash_conf_init(&flash_conf);
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) {
    printf("ERROR: Cannot open flash! Exiting...\n");
    pmsis_exit(-1);
  }

  ram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) {
    printf("ERROR: Cannot open ram! Exiting...\n");
    pmsis_exit(-3);
  }
}

struct pi_device *get_ram_ptr() { return &ram; }

void *ram_malloc(size_t size) {
  void *ptr = NULL;
  pi_ram_alloc(&ram, (uint32_t *)&ptr, size);
  return ptr;
}

void ram_free(void *ptr, size_t size) {
  pi_ram_free(&ram, (uint32_t)ptr, size);
}

void ram_read(void *dest, void *src, const size_t size) {
  pi_ram_read(&ram, (uint32_t)src, dest, size);
}

void ram_write(void *dest, void *src, const size_t size) {
  pi_ram_write(&ram, (uint32_t)dest, src, size);
}

void *cl_ram_malloc(size_t size) {
  uint32_t addr;
  pi_cl_ram_alloc_req_t req;
  pi_cl_ram_alloc(&ram, size, &req);
  pi_cl_ram_alloc_wait(&req, &addr);
  return (void *)addr;
}

void cl_ram_free(void *ptr, size_t size) {
  pi_cl_ram_free_req_t req;
  pi_cl_ram_free(&ram, (uint32_t)ptr, size, &req);
  pi_cl_ram_free_wait(&req);
}

void cl_ram_read(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_read(&ram, (uint32_t)src, dest, size, &req);
  pi_cl_ram_read_wait(&req);
}

void cl_ram_write(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  if (gap9_ram_copy_2d((uint32_t)dest, src, size, size, size, 0, &req))
    pi_cl_ram_write_wait(&req);
}

// GAP9 EVK PSRAM: a burst that crosses a page boundary with only a few bytes before it
// wraps -- the bytes after the boundary land at the START of the same page (measured: a
// 120 B L2->L3 row with 2 B before a 2 KB boundary wrote its last 118 B 2048 B lower; rows
// with >= 5 B before the boundary were correct). gvsoc does not model it. Rows at risk are
// rare (head < GAP9_RAM_MIN_HEAD at a GAP9_RAM_PAGE boundary), so the common case stays one
// asynchronous pi_cl_ram_copy_2d (returns 1: req is in flight); a transfer that contains a
// risky row is copied synchronously instead, with the head of each risky row split off as
// its own copy (returns 0: nothing in flight).
static inline int gap9_ram_row_risky(uint32_t ext, uint32_t len) {
  const uint32_t head = GAP9_RAM_PAGE - (ext & (GAP9_RAM_PAGE - 1));
  return head < GAP9_RAM_MIN_HEAD && len > head;
}

int gap9_ram_copy_2d(uint32_t ext, void *loc, uint32_t size, uint32_t stride, uint32_t len, int ext2loc,
                     pi_cl_ram_req_t *req) {
  if (len >= size || stride == len) { // contiguous: only the first boundary can have a short head
    stride = len = size;
  }
  uint32_t rows = size / len, r = 0;
  for (; r < rows; r++)
    if (gap9_ram_row_risky(ext + r * stride, len)) break;
  if (r == rows) {
    pi_cl_ram_copy_2d(&ram, ext, loc, size, stride, len, ext2loc, req);
    return 1;
  }
  uint8_t *l = (uint8_t *)loc;
  uint32_t r0 = 0; // first row not yet copied
  for (r = 0; r <= rows; r++) {
    const int risky = r < rows && gap9_ram_row_risky(ext + r * stride, len);
    if (r < rows && !risky) continue;
    if (r > r0) { // rows r0 .. r-1 in one 2-D copy
      pi_cl_ram_copy_2d(&ram, ext + r0 * stride, l + r0 * len, (r - r0) * len, stride, len, ext2loc, req);
      pi_cl_ram_copy_wait(req);
    }
    if (risky) { // head, then the rest from the page boundary on
      const uint32_t e = ext + r * stride, head = GAP9_RAM_PAGE - (e & (GAP9_RAM_PAGE - 1));
      pi_cl_ram_copy(&ram, e, l + r * len, head, ext2loc, req);
      pi_cl_ram_copy_wait(req);
      pi_cl_ram_copy(&ram, e + head, l + r * len + head, len - head, ext2loc, req);
      pi_cl_ram_copy_wait(req);
    }
    r0 = r + 1;
  }
  return 0;
}

size_t load_file_to_ram(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  size_t size = fd->size;
  size_t load_size = 0;
  size_t remaining_size = size;

  size_t offset = 0;
  do {

    remaining_size = size - offset;
    load_size = BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;

    pi_cl_fs_req_t req;
    pi_cl_fs_read(fd, buffer, load_size, &req);
    pi_cl_fs_wait(&req);
    cl_ram_write(dest + offset, buffer, load_size);
    offset += load_size;
  } while (offset < size);

  return offset;
}

size_t load_file_to_local(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  const size_t size = fd->size;
  size_t remaining_size = size;
  size_t offset = 0;
  pi_cl_fs_req_t req;

  while (offset < size) {
    remaining_size = size - offset;
    size_t load_size =
        BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;
    pi_cl_fs_read(fd, buffer, load_size, &req);
    pi_cl_fs_wait(&req);
    memcpy(dest + offset, buffer, load_size);
    offset += load_size;
  }

  return offset;
}
