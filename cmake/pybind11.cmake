set(PKG_NAME pybind11)
set(PKG_ROOT ${DEP_DIR}/${PKG_NAME}-src)

if(PYTHON_VERSION MATCHES  "3.7")
    set(URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.4.3.tar.gz")
else()
    set(URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
endif()

FetchContent_Declare(${PKG_NAME} URL ${URL})
FetchContent_GetProperties(${PKG_NAME})
if(NOT ${PKG_NAME}_POPULATED)
    FetchContent_Populate(${PKG_NAME})
endif()

include_directories(${PKG_ROOT}/include)
add_subdirectory(${PKG_ROOT})