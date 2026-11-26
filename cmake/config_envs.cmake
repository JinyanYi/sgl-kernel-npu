# find python binary
find_program(PYTHON_EXECUTABLE NAMES python3)

if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message(FATAL_ERROR "python3 is not found, install python firstly")
endif ()

# Discover package locations without loading the NPU backend on build hosts.
execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
        "${PYTHON_EXECUTABLE}" -c [=[
from importlib.util import find_spec
from pathlib import Path
import json

import torch


def package_dir(name):
    spec = find_spec(name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(name)
    return str(Path(spec.origin).resolve().parent)


print(json.dumps({
    "TORCH_DIR": package_dir("torch"),
    "TORCH_NPU_DIR": package_dir("torch_npu"),
    "PYBIND11_DIR": package_dir("pybind11"),
    "TORCH_ABI_ENABLED": torch.compiled_with_cxx11_abi(),
}))
]=]
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE ENV_DEFINES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to discover the Torch build environment: ${EXEC_RESULT}")
endif ()
foreach (key TORCH_DIR TORCH_NPU_DIR PYBIND11_DIR TORCH_ABI_ENABLED)
    string(JSON ${key} GET "${ENV_DEFINES}" ${key})
endforeach ()

message(STATUS "SOC_VERSION=${SOC_VERSION}")
message(STATUS "TORCH_DIR=${TORCH_DIR}")
message(STATUS "TORCH_NPU_DIR=${TORCH_NPU_DIR}")
message(STATUS "PYBIND11_DIR=${PYBIND11_DIR}")

# set _GLIBCXX_USE_CXX11_ABI
if (TORCH_ABI_ENABLED)
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=1")
else ()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=0")
endif ()
