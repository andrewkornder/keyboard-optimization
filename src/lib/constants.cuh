#ifndef MACROS_H
#define MACROS_H


#define FUNCTOR(name, ret, args, body) \
struct name { \
__device__ __host__ ret operator()args const noexcept body \
}


constexpr uint64_t THREADS_PER_BLOCK = 1024;

#define LAUNCH_BOUNDS_DEFAULT __launch_bounds__(THREADS_PER_BLOCK)
// #define CEILDIV(a, b) ((a) / (b) + ((a) % (b) != 0))
#define SWAP(array, x_, y_) {const auto temp = array[x_]; array[x_] = array[y_]; array[y_] = temp;}

#define COPYMODE_CAST 1
#define COPYMODE_MEMCPY 2
#define COPYMODE_BYTE 3

#endif //MACROS_H
