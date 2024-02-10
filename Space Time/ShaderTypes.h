//
//  ShaderTypes.h
//
//  This header file defines shared types and enumeration constants for Metal shaders and Swift/ObjC sources.
//  It facilitates interoperability between Metal and high-level languages by defining common data structures.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

// Conditionally define NS_ENUM for Metal or Objective-C/Swift environments.
#ifdef __METAL_VERSION__
    // Metal shader environment
    #define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
    typedef metal::int32_t EnumBackingType; // Define the type for enums in Metal.
#else
    // Objective-C/Swift environment
    #import <Foundation/Foundation.h>
    typedef NSInteger EnumBackingType; // Define the type for enums in Objective-C/Swift.
#endif

#include <simd/simd.h> // Import SIMD library for matrix and vector types.

// Enumeration for buffer indices to identify buffers in the shader.
typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions = 0, // Position data for mesh vertices.
    BufferIndexMeshGenerics  = 1, // Generic attribute data for mesh vertices.
    BufferIndexUniforms      = 2  // Uniform buffer for transformation matrices.
};

// Enumeration for vertex attributes to identify input vertex data in the shader.
typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition = 0, // Vertex position attribute.
    VertexAttributeTexcoord = 1, // Vertex texture coordinate attribute.
};

// Enumeration for texture indices to identify textures in the shader.
typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor = 0, // Color texture.
};

// Structure for transformation matrices used in vertex shading.
typedef struct
{
    matrix_float4x4 projectionMatrix; // Projection matrix for 3D projection.
    matrix_float4x4 modelViewMatrix;  // Model-view matrix for model transformation and camera view.
} Uniforms;

// Array of Uniforms structures to support multiple transformation matrices.
typedef struct
{
    Uniforms uniforms[2]; // Array of Uniforms, allowing for multiple sets of matrices.
} UniformsArray;

#endif /* ShaderTypes_h */
