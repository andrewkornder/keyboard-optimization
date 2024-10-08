cmake_minimum_required(VERSION 3.28)
project(keyboard-optimization CUDA)

set(CMAKE_CUDA_STANDARD 17)

function(CREATE_ name)
    add_executable(${name}
        src/api.cu

        src/lib/config.cu
        src/lib/config.cuh

        src/record.cu
        src/record.cuh

        src/common.cu
        src/common.cuh
        src/general.cuh
        src/test.cu
        src/test.cuh
        src/rng.cu
        src/rng.cuh
        src/carpalx.cu

        src/genetic/sort.cu
        src/genetic/genetic.cu
        src/genetic/genetic.cuh
        src/genetic/sort.cuh
        src/genetic/perf.cu
        src/genetic/perf.cuh
        src/genetic/kernel.cu
        src/genetic/kernel.cuh
        src/genetic/duplicates.cu
        src/genetic/duplicates.cuh

        src/lib/def.cuh
        src/lib/choices.cuh

        src/lib/text.cu
        src/lib/text.cuh
        src/lib/metric.cu
        src/lib/metric.cuh

        src/lib/md5.cu
        src/lib/md5.cuh

        src/lib/glob.cuh
        src/lib/glob.cu
    )
    list(SUBLIST ARGV 1 -1 define)
    target_include_directories(${name} PUBLIC
        src
        src/lib
        src/genetic
    )
    target_compile_definitions(${name} PUBLIC ${define})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    message("[${name}] defining: ${define}")
    add_dependencies(compile-all ${name})
endfunction()

function(CREATE_DD name metric)
    CREATE_(${name} ${metric})
endfunction()

add_custom_target(compile-all)
CREATE_DD(distance METRIC_DIST)
CREATE_DD(dickens METRIC_OKP)
CREATE_DD(carpalx METRIC_CARPALX)


set(CMAKE_CUDA_FLAGS "--use_fast_math --restrict -Xcompiler=\"/O2 /fp:fast /Qpar\"")
#set(CMAKE_CUDA_FLAGS -lineinfo)