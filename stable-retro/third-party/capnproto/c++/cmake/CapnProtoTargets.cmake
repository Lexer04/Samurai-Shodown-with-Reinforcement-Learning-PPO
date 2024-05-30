# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.8)
   message(FATAL_ERROR "CMake >= 2.8.0 required")
endif()
if(CMAKE_VERSION VERSION_LESS "2.8.3")
   message(FATAL_ERROR "CMake >= 2.8.3 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.8.3...3.22)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS CapnProto::kj CapnProto::kj-test CapnProto::kj-async CapnProto::kj-http CapnProto::capnp CapnProto::capnp-rpc CapnProto::capnp-json CapnProto::capnpc CapnProto::capnp_tool CapnProto::capnpc_cpp CapnProto::capnpc_capnp)
  list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
  if(TARGET "${_cmake_expected_target}")
    list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
  else()
    list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
  endif()
endforeach()
unset(_cmake_expected_target)
if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
  unset(_cmake_targets_defined)
  unset(_cmake_targets_not_defined)
  unset(_cmake_expected_targets)
  unset(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT _cmake_targets_defined STREQUAL "")
  string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
  string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)


# Create imported target CapnProto::kj
add_library(CapnProto::kj STATIC IMPORTED)

set_target_properties(CapnProto::kj PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_constexpr"
  INTERFACE_INCLUDE_DIRECTORIES "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/kj/.."
  INTERFACE_LINK_LIBRARIES "pthread"
)

# Create imported target CapnProto::kj-test
add_library(CapnProto::kj-test STATIC IMPORTED)

set_target_properties(CapnProto::kj-test PROPERTIES
  INTERFACE_LINK_LIBRARIES "CapnProto::kj"
)

# Create imported target CapnProto::kj-async
add_library(CapnProto::kj-async STATIC IMPORTED)

set_target_properties(CapnProto::kj-async PROPERTIES
  INTERFACE_COMPILE_OPTIONS "-pthread"
  INTERFACE_LINK_LIBRARIES "CapnProto::kj"
)

# Create imported target CapnProto::kj-http
add_library(CapnProto::kj-http STATIC IMPORTED)

set_target_properties(CapnProto::kj-http PROPERTIES
  INTERFACE_LINK_LIBRARIES "CapnProto::kj-async;CapnProto::kj"
)

# Create imported target CapnProto::capnp
add_library(CapnProto::capnp STATIC IMPORTED)

set_target_properties(CapnProto::capnp PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/.."
  INTERFACE_LINK_LIBRARIES "CapnProto::kj"
)

# Create imported target CapnProto::capnp-rpc
add_library(CapnProto::capnp-rpc STATIC IMPORTED)

set_target_properties(CapnProto::capnp-rpc PROPERTIES
  INTERFACE_LINK_LIBRARIES "CapnProto::capnp;CapnProto::kj-async;CapnProto::kj"
)

# Create imported target CapnProto::capnp-json
add_library(CapnProto::capnp-json STATIC IMPORTED)

set_target_properties(CapnProto::capnp-json PROPERTIES
  INTERFACE_LINK_LIBRARIES "CapnProto::capnp;CapnProto::kj-async;CapnProto::kj"
)

# Create imported target CapnProto::capnpc
add_library(CapnProto::capnpc STATIC IMPORTED)

set_target_properties(CapnProto::capnpc PROPERTIES
  INTERFACE_LINK_LIBRARIES "CapnProto::capnp;CapnProto::kj"
)

# Create imported target CapnProto::capnp_tool
add_executable(CapnProto::capnp_tool IMPORTED)

# Create imported target CapnProto::capnpc_cpp
add_executable(CapnProto::capnpc_cpp IMPORTED)

# Create imported target CapnProto::capnpc_capnp
add_executable(CapnProto::capnpc_capnp IMPORTED)

# Import target "CapnProto::kj" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::kj APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::kj PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/kj/libkj.a"
  )

# Import target "CapnProto::kj-test" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::kj-test APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::kj-test PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/kj/libkj-test.a"
  )

# Import target "CapnProto::kj-async" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::kj-async APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::kj-async PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/kj/libkj-async.a"
  )

# Import target "CapnProto::kj-http" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::kj-http APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::kj-http PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/kj/libkj-http.a"
  )

# Import target "CapnProto::capnp" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/libcapnp.a"
  )

# Import target "CapnProto::capnp-rpc" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnp-rpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnp-rpc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/libcapnp-rpc.a"
  )

# Import target "CapnProto::capnp-json" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnp-json APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnp-json PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/libcapnp-json.a"
  )

# Import target "CapnProto::capnpc" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnpc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/libcapnpc.a"
  )

# Import target "CapnProto::capnp_tool" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnp_tool APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnp_tool PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/capnp"
  )

# Import target "CapnProto::capnpc_cpp" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnpc_cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnpc_cpp PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/capnpc-c++"
  )

# Import target "CapnProto::capnpc_capnp" for configuration "RelWithDebInfo"
set_property(TARGET CapnProto::capnpc_capnp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(CapnProto::capnpc_capnp PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/home/lab/rl_stable/stable-retro/third-party/capnproto/c++/src/capnp/capnpc-capnp"
  )

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
