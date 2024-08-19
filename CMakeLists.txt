cmake_minimum_required(VERSION 3.28)
project(keyboard CUDA)

set(CMAKE_CUDA_STANDARD 17)

function(CREATE_ name)
    add_executable(${name}
        src/api.cu
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
        src/lib/constants.cuh

        src/lib/text.cu
        src/lib/text.cuh
        src/lib/metric.cu
        src/lib/metric.cuh
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
endfunction()

function(CREATE_DD name metric)
    CREATE_(${name} ${metric})
    CREATE_("${name}-ddup" ${metric} REMOVE_DUPLICATES)
endfunction()

CREATE_DD(metric-dist METRIC_DIST)
CREATE_DD(metric-okp METRIC_OKP)
CREATE_DD(metriccarpalx METRIC_CARPALX)
