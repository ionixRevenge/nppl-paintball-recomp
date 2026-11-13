cmake_minimum_required(VERSION 3.20)
project(nppl_pc C CXX)

# Generated C from recomp/recomp.py (derived from the user's own ISO; never committed)
set(RECOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/recomp" CACHE PATH "recompiler output directory")
file(GLOB RECOMP_SOURCES "${RECOMP_DIR}/recomp_*.c" "${RECOMP_DIR}/table.c")
if(NOT RECOMP_SOURCES)
    message(FATAL_ERROR "No recompiled sources in ${RECOMP_DIR}. Run: python recomp/recomp.py iso/SLUS_218.55 build/recomp")
endif()

set(RECOMP_OPT "/O2" CACHE STRING "optimisation flag for generated code (/O2: ~25% more fps than /O1 but ~35 min build; /Od builds much faster)")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

add_library(recomp STATIC ${RECOMP_SOURCES})
target_include_directories(recomp PUBLIC include "${RECOMP_DIR}")
if(MSVC)
    # generated code: big objects, labels/vars unused by design
    target_compile_options(recomp PRIVATE /bigobj /W0 ${RECOMP_OPT} /MP)
else()
    target_compile_options(recomp PRIVATE -w -O1 -fno-strict-aliasing)
endif()

add_executable(nppl_pc
    src/main.cpp src/cpu.cpp src/memory.cpp src/hw.cpp src/kernel.cpp src/devices.cpp src/sif.cpp src/iop_modules.cpp src/pad.cpp src/gamepad.cpp src/iso.cpp src/iop_cdvd.cpp src/iop_mc.cpp src/iop_audio.cpp src/audio_out.cpp src/hle_fileio.cpp src/hle_debug.cpp src/hle_setjmp.cpp src/vu.cpp src/gs.cpp src/dma.cpp src/display.cpp src/profiler.cpp)
target_include_directories(nppl_pc PRIVATE include src)
target_link_libraries(nppl_pc PRIVATE recomp)
if(MSVC)
    target_compile_options(nppl_pc PRIVATE /W3 /MP /O2 /Zi)
    target_compile_definitions(nppl_pc PRIVATE _CRT_SECURE_NO_WARNINGS)
    # recompiled code nests calls deeply; give the main thread a large stack
    target_link_options(nppl_pc PRIVATE /STACK:67108864 /DEBUG /OPT:REF /OPT:ICF)   # PDB for -prof and crash logs
    target_link_libraries(nppl_pc PRIVATE dbghelp)
endif()

# VU1 interpreter benchmark (replays vusnap.bin captured with NPPL_VUSNAP)
add_executable(vubench tools/vubench.cpp src/vu.cpp)
target_include_directories(vubench PRIVATE include src)
if(MSVC)
    target_compile_options(vubench PRIVATE /W3 /O2 /Zi)
    target_compile_definitions(vubench PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(vubench PRIVATE dbghelp winmm)
    target_link_options(vubench PRIVATE /DEBUG)
endif()
