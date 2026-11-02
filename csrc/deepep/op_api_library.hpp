#pragma once

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace deep_ep::op_api {

inline void *custom_library()
{
    static void *handle = []() -> void * {
        const char *path = std::getenv("SGLANG_DEEPEP_OP_API_LIBRARY");
        const bool pinned = path != nullptr && path[0] != '\0';
        if (pinned && path[0] != '/') {
            throw std::runtime_error("SGLANG_DEEPEP_OP_API_LIBRARY must be an absolute path");
        }
        void *library = dlopen(pinned ? path : "libcust_opapi.so", RTLD_LAZY | RTLD_LOCAL);
        if (library == nullptr && pinned) {
            const char *error = dlerror();
            throw std::runtime_error(std::string("Cannot load pinned DeepEP API library ") + path + ": " +
                                     (error != nullptr ? error : "unknown loader error"));
        }
        return library;
    }();
    return handle;
}

inline void *resolve(const char *name)
{
    // Use the exact vendor DSO. Another component may have already loaded a
    // different libcust_opapi.so with the same SONAME in this process.
    if (void *library = custom_library()) {
        if (void *address = dlsym(library, name)) {
            return address;
        }
    }
    // CANN built-in tensor/scalar helpers still come from the standard DSO.
    static void *standard = dlopen("libopapi.so", RTLD_LAZY | RTLD_LOCAL);
    return standard != nullptr ? dlsym(standard, name) : nullptr;
}

}  // namespace deep_ep::op_api
