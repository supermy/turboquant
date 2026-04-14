#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t id;
    float distance;
} VQQueryResult;

typedef struct {
    const float* query;
    int d;
    int k;
    int nprobe;
    int refine_factor;
    int use_sq8;
} VQIVFSearchParams;

typedef struct {
    const float* query;
    int d;
    int k;
    int index_type;
} VQFlatSearchParams;

void* vq_engine_open(const char* path);
void vq_engine_close(void* engine);

VQQueryResult* vq_ivf_search(void* engine, const VQIVFSearchParams* params, int* n_results);
VQQueryResult* vq_flat_search(void* engine, const VQFlatSearchParams* params, int* n_results);

VQQueryResult* vq_ivf_batch_search(
    void* engine,
    const float* queries, int n_queries,
    int d, int k, int nprobe, int refine_factor, int use_sq8,
    int** n_results_per_query);

void vq_results_free(VQQueryResult* results);
void vq_n_results_free(int* n_results);

#ifdef __cplusplus
}
#endif
