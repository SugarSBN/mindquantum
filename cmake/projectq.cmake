set(PKG_NAME projectq)
set(PKG_ROOT ${DEP_DIR}/${PKG_NAME}-src)

set(URL "https://gitee.com/mirrors/ProjectQ/repository/archive/v0.5.1.tar.gz")
set(PATCH_FILE ${PATCH_DIR}/projectq/projectq.patch001)

FetchContent_Declare(${PKG_NAME} URL ${URL})
FetchContent_Populate(${PKG_NAME})

set(${PKG_NAME}_PATCHED "0" CACHE STRING INTERNAL)
if(NOT ${PKG_NAME}_PATCHED EQUAL "1")
    message("Patching for ${PKG_NAME}")
    execute_process(COMMAND ${Patch_EXECUTABLE} -p1 INPUT_FILE
        ${PATCH_FILE} WORKING_DIRECTORY ${PKG_ROOT} RESULT_VARIABLE Result)
    set(${PKG_NAME}_PATCHED "1" CACHE STRING INTERNAL FORCE)
    if(NOT Result EQUAL "0")
        message(FATAL_ERROR "Failed patch: ${PATCH_FILE}")
    endif()
endif()

include_directories(${PKG_ROOT})