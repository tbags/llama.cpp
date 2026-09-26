if (DEFINED NOE_ROOT)
    message(STATUS "noe: using NOE_ROOT override: ${NOE_ROOT}")
    set(NOE_HINT "${NOE_ROOT}")
else()
    set(NOE_HINT "")
endif()

find_path(NOE_INCLUDE
            NAMES cix_noe_standard_api.h
            HINTS ${NOE_HINT} /usr/share/cix /usr /usr/local
            PATH_SUFFIXES include/npu include npu)
if (NOE_INCLUDE)
    message(STATUS "noe: found include: ${NOE_INCLUDE}")
else()
    message(FATAL_ERROR "noe: include directory not found")
endif()

find_library(NOE_LIB
                NAMES noe
                HINTS ${NOE_HINT} /usr/share/cix /usr /usr/local
                PATH_SUFFIXES lib lib64 share/cix/lib)
if (NOE_LIB)
    message(STATUS "noe: found library: ${NOE_LIB}")
else()
    message(FATAL_ERROR "noe: library not found")
endif()

file(GLOB GGML_SOURCES_NOE "*.c" "*.cpp")
file(GLOB GGML_HEADERS_NOE "*.h" "*.hpp")

ggml_add_backend_library(ggml-noe ${GGML_HEADERS_NOE} ${GGML_SOURCES_NOE})
target_link_libraries(ggml-noe PRIVATE ${NOE_LIB})
target_include_directories(ggml-noe PRIVATE ${NOE_INCLUDE})
target_compile_definitions(ggml-noe PRIVATE GGML_USE_NOE)
