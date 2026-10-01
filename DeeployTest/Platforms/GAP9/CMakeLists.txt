# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})



set(${SDKCONFIG_FILE} ${CMAKE_CURRENT_LIST_DIR}/sdk.config)

file(GLOB_RECURSE SOURCES
    src/CycleCounter.c
    src/deeploytest.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
# add_executable(${ProjectId} ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

# Network.c needs CycleCounter.h when --profileTiling is enabled
target_include_directories(network PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId} "gap9.evk")
add_board_deployment(${ProjectId} "gap9.evk")

if(POWER_MEASUREMENT)
  target_compile_definitions(${ProjectId} PRIVATE POWER_MEASUREMENT)
endif()

# L1-memory knobs. SLAVESTACKSIZE sizes the per-core slave stacks (defaults to
# 768 B in deeploytest.c; moving them into L1 TCDM at that size took the full
# FEMBA model from 288M to 180M cycles). CC_STACK_SIZE sizes the cluster
# controller / master stack, which also lives in L1.
# Example: cmake ... -DSLAVESTACKSIZE=768 -DCC_STACK_SIZE=4098
# Also honour the environment, because the test runners do not forward -D to CMake.
# SelectiveScan is happy at the 768 B default; SSDScan needs >= 1024 because it keeps
# dB_chunk[Chunk_size * N] and five more Chunk_size arrays on the stack (736 B at
# Chunk_size=8, and it grows linearly with Chunk_size). Below that it writes past the
# stack into the address space under L1 and gvsoc reports an invalid access at
# offset 0x0fffffxx.
if(NOT SLAVESTACKSIZE AND DEFINED ENV{SLAVESTACKSIZE})
  set(SLAVESTACKSIZE $ENV{SLAVESTACKSIZE})
endif()
if(SLAVESTACKSIZE)
  message(STATUS "[Deeploy GAP9] SLAVESTACKSIZE=${SLAVESTACKSIZE} B/core")
  target_compile_definitions(${ProjectId} PRIVATE SLAVESTACKSIZE=${SLAVESTACKSIZE})
endif()

if(CC_STACK_SIZE)
  target_compile_definitions(${ProjectId} PRIVATE CC_STACK_SIZE=${CC_STACK_SIZE})
endif()

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(network PRIVATE
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
    -O3
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
