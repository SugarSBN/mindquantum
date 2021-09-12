FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

function(find_python_package out_inc out_lib)
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python3")
  else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
  endif()

  execute_process(
          COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE inc)
  string(STRIP "${inc}" inc)
  set(${out_inc} ${inc} PARENT_SCOPE)

  execute_process(
          COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig as sysconfig; import os; \
                  print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE lib)
  string(STRIP "${lib}" lib)
  set(${out_lib} ${lib} PARENT_SCOPE)
endfunction()

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    if(WIN32)
        if(Python3_DIR)
            message("Python3_DIR set already: " ${Python3_DIR})
        else()
            string(LENGTH ${PYTHON_LIBRARIES} PYTHON_LIBRARIES_LEN)
            string(LENGTH "libpythonxx.a" Python3_NAME_LEN)
            math(EXPR Python3_DIR_LEN  ${PYTHON_LIBRARIES_LEN}-${Python3_NAME_LEN})
            string(SUBSTRING ${Python3_LIBRARIES} 0 ${Python3_DIR_LEN} Python3_DIR)
            message("Python3_DIR: " ${Python3_DIR})
        endif()
        link_directories(${Python3_DIR})
    endif()
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Patch)
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(DEP_DIR ${BUILD_DIR}/_deps)
set(PATCH_DIR ${CMAKE_SOURCE_DIR}/third_party/patch)
set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

include(${CMAKE_SOURCE_DIR}/cmake/pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/projectq.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/quest.cmake)