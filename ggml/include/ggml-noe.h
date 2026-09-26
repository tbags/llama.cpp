#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a shared buffer type for Cix NOE device allocations
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_noe_buffer_type(void);

// Initializes the Cix NOE device backend context
GGML_BACKEND_API ggml_backend_t ggml_backend_noe_init(void);

// Checks if the backend is an NOE backend
GGML_BACKEND_API bool ggml_backend_is_noe(ggml_backend_t backend);

// Returns the NOE backend registry entry
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_noe_reg(void);

#ifdef __cplusplus
}
#endif
