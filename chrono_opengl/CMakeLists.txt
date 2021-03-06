#=============================================================================
# CMake configuration file for Chrono OpenGL
#
# Invoked from the main CMakeLists.txt using ADD_SUBDIRECTORY()
#=============================================================================

ADD_EXECUTABLE(embedfile embedfile.cpp)

#All shaders are compiled into strings by embedfile.cpp
SET(GLSL_SHADERS
  cloud_frag
  cloud_vert
  dot_frag
  dot_vert
  sphere_frag
  sphere_vert
  phong_frag
  phong_vert
  text_frag
  text_vert
  )
  

FOREACH(GLSL_SHADER ${GLSL_SHADERS})
  ADD_CUSTOM_COMMAND(
    OUTPUT  ${CMAKE_BINARY_DIR}/resources/${GLSL_SHADER}.h
    COMMAND embedfile 
            ${GLSL_SHADER}
            ${CMAKE_SOURCE_DIR}/chrono_opengl/resources/${GLSL_SHADER}.glsl 
            ${CMAKE_BINARY_DIR}/resources/${GLSL_SHADER}.h
    DEPENDS ${CMAKE_SOURCE_DIR}/chrono_opengl/resources/${GLSL_SHADER}.glsl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
ENDFOREACH()

SET(DEFAULT_OBJ_FILES
  box
  cone
  sphere
  cylinder
  )

FOREACH(DEFAULT_OBJ_FILE ${DEFAULT_OBJ_FILES})
  ADD_CUSTOM_COMMAND(
    OUTPUT  ${CMAKE_BINARY_DIR}/resources/${DEFAULT_OBJ_FILE}.h
    COMMAND embedfile 
            ${DEFAULT_OBJ_FILE}_mesh_data
            ${CMAKE_SOURCE_DIR}/chrono_opengl/resources/${DEFAULT_OBJ_FILE}.obj
            ${CMAKE_BINARY_DIR}/resources/${DEFAULT_OBJ_FILE}.h
    DEPENDS ${CMAKE_SOURCE_DIR}/chrono_opengl/resources/${DEFAULT_OBJ_FILE}.obj
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
ENDFOREACH()

# ------------------------------------------------------------------------------

# OBJ files are copied to the resources folder in the binary directory
SET(OBJ_FILES
  box.obj
  cone.obj
  cylinder.obj
  sphere.obj
  )

IF(MSVC)
  SET(OBJ_FILES_DEST "${CMAKE_BINARY_DIR}/bin/resources")
ELSEIF(XCODE_VERSION)
  SET(OBJ_FILES_DEST "${CMAKE_BINARY_DIR}/bin/resources")
ELSE()
  SET(OBJ_FILES_DEST "${CMAKE_BINARY_DIR}/resources")
ENDIF()

FOREACH(OBJ_FILE ${OBJ_FILES})
  FILE(COPY        "${CMAKE_SOURCE_DIR}/chrono_opengl/resources/${OBJ_FILE}"
       DESTINATION "${OBJ_FILES_DEST}")
ENDFOREACH()

# ------------------------------------------------------------------------------

SET(ChronoEngine_OpenGL_SOURCES 
	core/ChOpenGLShader.cpp
	core/ChOpenGLObject.cpp
	core/ChOpenGLVertexAttributes.cpp
	shapes/ChOpenGLMesh.cpp
	shapes/ChOpenGLCloud.cpp
	shapes/ChOpenGLWires.cpp
	shapes/ChOpenGLText.cpp
	shapes/ChOpenGLOBJ.cpp
	shapes/obj/ChOpenGLOBJLoader.cpp
	shapes/obj/tiny_obj_loader.cc
	UI/ChOpenGLContacts.cpp
	UI/ChOpenGLHUD.cpp
	UI/ChOpenGLGraphs.cpp
	ChOpenGLViewer.cpp 
	ChOpenGLCamera.cpp
	ChOpenGLWindow.cpp
	)

SET(ChronoEngine_OpenGL_HEADERS
	core/ChOpenGLBase.h
	core/ChApiOpenGL.h
	core/ChOpenGLShader.h
	core/ChOpenGLObject.h
	core/ChOpenGLVertexAttributes.h
	shapes/ChOpenGLCloud.h
	shapes/ChOpenGLWires.h
	shapes/ChOpenGLText.h
	shapes/ChOpenGLOBJ.h
	shapes/obj/ChOpenGLOBJLoader.h
	shapes/obj/tiny_obj_loader.h
	UI/ChOpenGLContacts.h
	UI/ChOpenGLHUD.h
	UI/ChOpenGLGraphs.h
	ChOpenGLViewer.h 
	ChOpenGLCamera.h
	ChOpenGLWindow.h
	ChOpenGLMaterials.h
	${CMAKE_BINARY_DIR}/resources/text_frag.h
	${CMAKE_BINARY_DIR}/resources/text_vert.h
	${CMAKE_BINARY_DIR}/resources/phong_frag.h
	${CMAKE_BINARY_DIR}/resources/phong_vert.h
	${CMAKE_BINARY_DIR}/resources/cloud_frag.h
	${CMAKE_BINARY_DIR}/resources/cloud_vert.h
	${CMAKE_BINARY_DIR}/resources/dot_frag.h
  ${CMAKE_BINARY_DIR}/resources/dot_vert.h
  ${CMAKE_BINARY_DIR}/resources/sphere_frag.h
  ${CMAKE_BINARY_DIR}/resources/sphere_vert.h
  ${CMAKE_BINARY_DIR}/resources/box.h
  ${CMAKE_BINARY_DIR}/resources/cone.h
  ${CMAKE_BINARY_DIR}/resources/sphere.h
  ${CMAKE_BINARY_DIR}/resources/cylinder.h
	)

IF (MSVC)
	# On Windows, include necessary Bullet source files.
	SET(ChronoEngine_OpenGL_SOURCES ${ChronoEngine_OpenGL_SOURCES}
		${CH_CHRONO_SDKDIR}/collision/bullet/BulletCollision/NarrowPhaseCollision/btPersistentManifold.cpp
		)
ENDIF()

SET(INCLUDE_DIR
	${CHRONO_INC}
	${CMAKE_SOURCE_DIR}
	${CMAKE_BINARY_DIR}
	${GLM_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIR}
	${GLFW_INCLUDE_DIR}
	)

INCLUDE_DIRECTORIES(${INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE})

# The OPENGL library is added to the project,
# and some custom properties of this target are set.

ADD_LIBRARY(ChronoEngine_OpenGL SHARED 
	${ChronoEngine_OpenGL_SOURCES}
	${ChronoEngine_OpenGL_HEADERS})

SET_TARGET_PROPERTIES(ChronoEngine_OpenGL PROPERTIES 
	COMPILE_FLAGS "${CH_BUILDFLAGS}"
LINK_FLAGS "${CH_LINKERFLAG_GPU}"
	COMPILE_DEFINITIONS "CH_API_COMPILE_OPENGL")

TARGET_LINK_LIBRARIES(ChronoEngine_OpenGL
	${GLEW_LIBRARY}
	${GLFW_LIBRARY}
	${OPENGL_LIBRARIES}
	${CHRONO_LIB}
	ChronoEngine_Parallel)

INSTALL(TARGETS ChronoEngine_OpenGL
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
