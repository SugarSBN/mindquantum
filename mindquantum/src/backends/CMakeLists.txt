set(BACKENDS
    projectq
    quest
)

foreach(_comp ${BACKENDS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${_comp})
    add_subdirectory(${_comp})
endforeach()