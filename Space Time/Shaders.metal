//
//  Shaders.metal
//
// Oluwasanmi Adenaiye MD MS 01.24.2024
// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>


// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

#define AA 2
#define AB 2
#define sphereDistance 50

//struct Vertex{
//    float3 position [[attribute(VertexAttributePosition)]];
// //   float2 texCoord [[attribute(VertexAttributeTexcoord)]];
//};
//
//
//struct ColorInOut{
//    float4 position [[position]];
// //   float2 texCoord;
//};



float4 returnSPH(){
    float4 sph1 = float4(0.0,0.0,0.0,1.0);
    return sph1;
}

float4 returnSPH12(float3 cameraPosition, float3 cameraDirection, float distanceToFront) {
    float3 sphereCenter = cameraPosition + normalize(cameraDirection) * distanceToFront;
    return float4(sphereCenter, 1.0); // Assuming the sphere's radius is 1.0
}

//const float4 sph1 = float4(0.0,0.0,0.0,1.0);

float shpIntersect(float3 ro, float3 rd, float4 sph )
{
    float3 oc = ro - sph.xyz;
    float b = dot( rd, oc );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h>0.0 ) h = -b - sqrt( h );
    return h;
}

float sphDistance(  float3 ro,  float3 rd,  float4 sph )
{
    float3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float h = dot( oc, oc ) - b*b;
    return sqrt( max(0.0,h)) - sph.w;
}

float sphSoftShadow(  float3 ro,  float3 rd,  float4 sph,  float k )
{
    float3 oc = sph.xyz - ro;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    return (b<0.0) ? 1.0 : 1.0 - smoothstep( 0.0, 1.0, k*h/b );
}

float3 sphNormal(  float3 pos,  float4 sph )
{
    return (pos - sph.xyz)/sph.w;
}

float3 fancyCube2(texture2d<float> texture, sampler sam,  float3 direction,  float s,  float b )
{
    
        
    //calculate the sampling coordinates based on direction
    float2 coordX = 0.5 + s * direction.yz/direction.x;
    float2 coordY = 0.5 + s * direction.zx/direction.y;
    float2 coordZ = 0.5 + s * direction.xy/direction.z;
    
    
    //sample texture at calculated coordinates
    float3 colx = texture.sample( sam, coordX, level(b) ).xyz;
    float3 coly = texture.sample( sam, coordY, level(b) ).xyz;
    float3 colz = texture.sample( sam, coordZ, level(b) ).xyz;
    
    //calculate weighted color components
    float3 n = direction * direction;
    float3 resultColor = (colx * n.x + coly * n.y + colz * n.z)/ (n.x +n.y + n.z);
    return resultColor;
}

float3 fancyCube(texture2d<float> texture, sampler sam, float3 direction, float s, float b) {
    // Calculate the sampling coordinates based on direction and ensure they tile by wrapping with fract
    float2 coordX = fract(0.5 + s * direction.yz / direction.x);
    float2 coordY = fract(0.5 + s * direction.zx / direction.y);
    float2 coordZ = fract(0.5 + s * direction.xy / direction.z);
    
    // Sample texture at calculated coordinates
    float3 colx = texture.sample(sam, coordX, level(b)).xyz;
    float3 coly = texture.sample(sam, coordY, level(b)).xyz;
    float3 colz = texture.sample(sam, coordZ, level(b)).xyz;
    
    // Calculate weighted color components
    float3 n = direction * direction;
    float3 resultColor = (colx * n.x + coly * n.y + colz * n.z) / (n.x + n.y + n.z);
    return resultColor;
}


float2 hash( float2 p ) { p=float2(dot(p,float2(127.1,311.7)),dot(p,float2(269.5,183.3))); return fract(sin(p)*43758.5453); }

float2 voronoi(  float2 x )
{
    float2 n = floor( x );
    float2 f = fract( x );

    float3 m = float3( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        float2  g = float2( float(i), float(j) );
        float2  o = hash( n + g );
        float2  r = g - f + o;
        float d = dot( r, r );
        if( d<m.x ) m = float3( d, o );
    }

    return float2( sqrt(m.x), m.y+m.z );
}

//=======================================================

float3 background(  float3 direction,  float3 l ,
                  texture2d<float> iChannel1,
                  sampler sam)
{
    float3 col = float3(0.0);
    
         col += 0.5*pow( fancyCube( iChannel1,sam,direction, 0.05, 5.0 ).zyx, float3(2.0) );
         col += 0.2*pow( fancyCube( iChannel1,sam, direction, 0.10, 3.0 ).zyx, float3(1.5) );
         col += 0.8*float3(0.80,0.5,0.6)*pow( fancyCube( iChannel1,sam, direction, 0.1, 0.0 ).xxx, float3(6.0) );
    
    float stars = smoothstep( 0.3, 0.7, fancyCube( iChannel1,sam,direction, 0.91, 0.0 ).x );

    float3 n = abs(direction);
    n = n * n * n;
    //(50.0 * direction.xy, 1.0);
    float2 vxy = voronoi( 50.0 * direction.xy );
    float2 vyz = voronoi( 50.0 * direction.yz );
    float2 vzx = voronoi( 50.0 * direction.zx );
    float2 r = (vyz*n.x + vzx*n.y + vxy*n.z) / (n.x+n.y+n.z);
    col += 0.5 * stars * clamp(1.0-(3.0+r.y*5.0)*r.x,0.0,1.0);

    col = 1.5 * col - 0.2;
    col += float3(-0.05,0.1,0.0);

    float s = clamp( dot(direction,l), 0.0, 1.0 );
    col += 0.4*pow(s,5.0) * float3(1.0,0.7,0.6) * 2.0;
    col += 0.4*pow(s,64.0) * float3(1.0,0.9,0.8) * 2.0;
    
    return col;
}

//--------------------------------------------------------------------


float rayTrace(  float3 ro,  float3 rd )
{
    //return shpIntersect( ro, rd, returnSPH1(ro, rd, sphereDistance) );
    return shpIntersect( ro, rd, returnSPH() );

}

float map(  float3 pos ,float3 ro,  float3 rd )
{
    //float2 r = pos.xz - returnSPH1(ro, rd, sphereDistance).xz;
    float2 r = pos.xz - returnSPH().xz;

    float h = 1.0-2.0/(1.0+0.3*dot(r,r));
    return pos.y - h;
}

float rayMarch(  float3 ro,  float3 rd, float tmax )
{
    float t = 0.0;
    
    // bounding plane
    float h = (1.0-ro.y)/rd.y;
    if( h>0.0 ) t=h;

    // raymarch
    for( int i=0; i<20; i++ )
    {
        float3 pos = ro + t*rd;
        float h = map( pos ,ro,rd);
        if( h<0.001 || t>tmax ) break;
        t += h;
    }
    return t;
}

// Renders the scene, calculating lighting, reflections, and environmental effects.
float3 render( float3 rayOrigin, float3 rayDirection, texture2d<float> environmentTexture, texture2d<float> cloudTexture, sampler textureSampler, float time) {
    
    // Light direction for the scene
    float3 lightDirection = normalize(float3(1.0, 0.2, 1.0));
    
    // Initial color calculated based on the background function
    float3 color = background(rayDirection, lightDirection, cloudTexture, textureSampler);
    
    // Perform ray tracing to detect sphere intersection
    float intersectionDistance = rayTrace(rayOrigin, rayDirection);

    // If there is an intersection, calculate the color at the intersection point
    if (intersectionDistance > 0.0) {
        // Base material color
        float3 materialColor = float3(0.18);
        // Calculate the intersection position
        float3 intersectionPosition = rayOrigin + intersectionDistance * rayDirection;
        // Calculate normal at the intersection point
        //float3 normal = sphNormal(intersectionPosition, returnSPH1(rayOrigin, rayDirection, sphereDistance));
        float3 normal = sphNormal(intersectionPosition, returnSPH());

        // Animation modifiers based on time
        float animationModifier = 0.1 * time;
        float2 phaseRotation = float2(cos(animationModifier), sin(animationModifier));
        // Transformed normal for animation effect
        float3 transformedNormal = normal;
        transformedNormal.xz = float2x2(phaseRotation.x, -phaseRotation.y, phaseRotation.y, phaseRotation.x) * transformedNormal.xz;

        // Additional animation effect
        float animationModifier2 = 0.08 * time - 1.0 * (1.0 - normal.y * normal.y);
        phaseRotation = float2(cos(animationModifier2), sin(animationModifier2));
        float3 transformedNormal2 = normal;
        transformedNormal2.xz = float2x2(phaseRotation.x, -phaseRotation.y, phaseRotation.y, phaseRotation.x) * transformedNormal2.xz;

        // Reflection calculation for reflective materials
        float3 reflection = reflect(rayDirection, normal);
        float fresnelEffect = clamp(1.0 + dot(normal, rayDirection), 0.0, 1.0);

        // Lighting effects based on the environment texture
        float lightingEffect = fancyCube(environmentTexture, textureSampler, transformedNormal, 0.03, 0.0).x;
        lightingEffect += -0.1 + 0.3 * fancyCube(environmentTexture, textureSampler, transformedNormal, 8.0, 0.0).x;

        // Calculate sea and land colors based on lighting and Fresnel effects
        float3 seaColor = mix(float3(0.0, 0.07, 0.2), float3(0.0, 0.01, 0.3), fresnelEffect) * 0.15;
        float3 landColor = mix(float3(0.02, 0.04, 0.0), float3(0.05, 0.1, 0.0), smoothstep(0.4, 1.0, fancyCube(environmentTexture, textureSampler, transformedNormal, 0.1, 0.0).x)) * fancyCube(environmentTexture, textureSampler, transformedNormal, 0.3, 0.0).xyz * 0.5;

        // Determine the material color based on the calculated sea and land colors
        float landOrSea = smoothstep(0.45, 0.46, lightingEffect);
        materialColor = mix(seaColor, landColor, landOrSea);

        // Cloud effects
        float3 cloudWrapEffect = -1.0 + 2.0 * fancyCube(cloudTexture, textureSampler, transformedNormal2.xzy, 0.025, 0.0).xyz;
        float cloudDensity = fancyCube(cloudTexture, textureSampler, transformedNormal2 + 0.2 * cloudWrapEffect, 0.05, 0.0).y;
        float clouds = smoothstep(0.3, 0.6, cloudDensity);
        
        // Combine material and cloud effects
        materialColor = mix(materialColor, float3(0.93 * 0.15), clouds);

        // Diffuse lighting calculation
        float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
        materialColor *= 0.8;
        float3 linearColor = float3(3.0, 2.5, 2.0) * diffuse;
        linearColor += 0.01;
        color = materialColor * linearColor;
        color = pow(color, float3(0.4545)); // Gamma correction
        color += 0.6 * fresnelEffect * fresnelEffect * float3(0.9, 0.9, 1.0) * (0.3 + 0.7 * diffuse);

        // Specular highlights
        float specular = clamp(dot(reflection, lightDirection), 0.0, 1.0);
        float totalSpecular = pow(specular, 3.0) + 0.5 * pow(specular, 16.0);
        color += (1.0 - 0.5 * landOrSea) * clamp(1.0 - 2.0 * clouds, 0.0, 1.0) * 0.3 * float3(0.5, 0.4, 0.3) * totalSpecular * diffuse;
    }
    
    // Raymarch additional elements if needed
    float maximumDistance = 20.0;
    if (intersectionDistance > 0.0) maximumDistance = intersectionDistance;
    intersectionDistance = rayMarch(rayOrigin, rayDirection, maximumDistance);
    if (intersectionDistance < maximumDistance) {
        // Calculate position and wireframe effects if the raymarch finds an intersection
        float3 position = rayOrigin + intersectionDistance * rayDirection;
        float2 sinPattern = sin(2.0 * 6.2831 * position.xz);
        float3 wireframeColor = float3(0.0);
        wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.x));
        wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.y));
        wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.x));
        wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.y));
        //wireframeColor *= 0.2 + 1.0 * sphSoftShadow(position, lightDirection, returnSPH1(rayOrigin, rayDirection, sphereDistance), 4.0);
        wireframeColor *= 0.2 + 1.0 * sphSoftShadow(position, lightDirection, returnSPH(), 4.0);

        color += wireframeColor * 0.5 * exp(-0.05 * intersectionDistance * intersectionDistance);
    }

    // Outer glow effect for objects
//    if (dot(rayDirection, returnSPH1(rayOrigin, rayDirection, sphereDistance).xyz - rayOrigin) > 0.0) {
//        float distance = sphDistance(rayOrigin, rayDirection, returnSPH1(rayOrigin, rayDirection, sphereDistance));
    if (dot(rayDirection, returnSPH().xyz - rayOrigin) > 0.0) {
        float distance = sphDistance(rayOrigin, rayDirection, returnSPH());
        float3 glow = float3(0.0);
        glow += float3(0.6, 0.7, 1.0) * 0.3 * exp(-2.0 * abs(distance)) * step(0.0, distance);
        glow += 0.6 * float3(0.6, 0.7, 1.0) * 0.3 * exp(-8.0 * abs(distance));
        glow += 0.6 * float3(0.8, 0.9, 1.0) * 0.4 * exp(-100.0 * abs(distance));
        color += glow * 1.5;
    }
    // Apply time-based fading to the color
    color *= smoothstep(0.0, 6.0, time);

    return color;
}

// Sets up the camera matrix based on the camera's position, target, and roll angle.
float3x3 setCamera(float3 cameraPosition, float3 cameraTarget, float cameraRoll) {
    float3 forwardVector = normalize(cameraTarget - cameraPosition);
    float3 rightVector = float3(sin(cameraRoll), cos(cameraRoll), 0.0);
    float3 upVector = normalize(cross(forwardVector, rightVector));
    float3 correctedRightVector = normalize(cross(upVector, forwardVector));
    return float3x3(correctedRightVector, upVector, -forwardVector);
}

//struct PoseConstants {
//    float4x4 projectionMatrix;
//    float4x4 viewMatrix;
//    float3 cameraPosition;
//};
//
//struct InstanceConstants {
//    float4x4 modelMatrix;
//};

struct VertexIn {
    float3 position  [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float  time;
    float3 modelNormal;
    float3 RayOri;
    float3 RayDir;
};



vertex VertexOut vertexShader(const VertexIn in [[stage_in]],
                              constant UniformsArray &uniformsArray [[buffer(1)]]){

    VertexOut out;
    
    
//    // Hardcoded positions for debugging
//    float3 positions[6] = { 
//        float3(-1.0, 1.0, 0.0),
//        float3( -1.0, -1.0, 0.0),
//        float3(1.0, -1.0, 0.0),
//        float3(-1.0,  1.0, 0.0),
//        float3(1.0, -1.0, 0.0),
//        float3( 1.0,  1.0, 0.0)};
//    // Define all corners
//        out.position = float4(positions[in.vertexID], 1.0);
    
    
    
    // access the first set of uniforms
    constant Uniforms& uniforms = uniformsArray.uniforms[0];
    
    //float4 clipPositions = float4(in.position.xy, 0.9f, 1.0f);
    
    // Transform vertex positions to clip space
    //out.position = uniforms.projectionMatrix * clipPositions;// uniforms.modelViewMatrix * float4(in.position, 1.0f);
    
    out.position = uniforms.projectionMatrix  * uniforms.viewMatrix * uniforms.modelMatrix * float4(in.position, 1.0f);
    //* uniforms.viewMatrix * uniforms.modelMatrix
    //out.position = float4(in.position, 1.0f);
    //out.position = float4(in.position, 1.0f);
    // The ray origin is the camera position in world space, already provided in uniforms.
    out.RayOri = uniforms.cameraPosition;
    
    // Calculate the world position of the vertex by applying the model-view matrix to the vertex position.
    // This assumes that the model-view matrix transforms vertices from model space to view space.
    // To get the position in world space, you would typically use the model matrix alone, but
    // since we're directly using the modelViewMatrix, it implies the cameraPosition is considering the view transformation.

    float4 worldPosition = (uniforms.modelMatrix) * float4(in.position, 1.0f);
    
    // The ray direction is from the camera to the vertex in world space.
    // However, since worldPosition is in view space after applying modelViewMatrix,
    // we don't perform a subtraction between worldPosition and cameraPosition here.
    // If you need the direction vector from the camera to the vertex in world space, you will have to adjust this calculation
    // based on how you've setup your transformations and what space you're working in.
    
    // For a typical scenario where you might want to calculate a view direction vector in view space,
    // you could simply use the normalized vertex position in view space (ignoring translation).
    
    out.RayDir = normalize(worldPosition.xyz - out.RayOri);
    out.time = uniforms.time;
    return out;
}


//fragment float4 fragmentShader(VertexOut in [[stage_in]],
//                               constant UniformsArray &uniformsArray [[buffer(0)]],
//                              texture2d<float> iChannel1 [[texture(2)]],
//                              texture2d<float> iChannel0 [[texture(1)]]) {
//    
//    // access the first set of uniforms
//    constant Uniforms& uniforms = uniformsArray.uniforms[0];
//    
//    float4 finalColor = float4(1,0,0,1);
//    
//    float cameraPositionX = uniforms.viewMatrix.columns[3][0];
//    float zo = 1.0 + smoothstep( 5.0, 15.0, abs(in.time-48.0) );
//
//
//    float an = 3.0 + 0.05 * in.time + 6.0 * cameraX/2732;
//    
////  float an = 3.0 + 0.05 * in.time + 6.0 * cameraX/iResolution.x;
////  2732 X 2048
//    
//    float3 ro = zo * float3( 2.0 * cos(an), 1.0, 2.0 * sin(an) );
//    float3 rt = float3( 1.0, 0.0, 0.0 );
//    float3x3 cam = setCamera( ro, rt, 0.35 );
//    sampler sam;
//    //finalColor.xyz = render(in.RayOri, in.RayDir, iChannel0, iChannel1, sam, in.time);
//    finalColor.xyz = render(ro+cam * in.RayOri, cam * in.RayDir, iChannel0, iChannel1, sam, in.time);
//    return finalColor;
//    //return float4(1,0,0,1);
//}
fragment float4 fragmentShader(VertexOut vertexOutput [[stage_in]],
                               constant UniformsArray &shaderUniformsArray [[buffer(0)]],
                               texture2d<float> stars [[texture(2)]],
                               texture2d<float> map [[texture(1)]]) {
    
    // Access the first set of uniforms
    constant Uniforms& uniforms = shaderUniformsArray.uniforms[0];
    float4 outputColor = float4(1,0,0,1);
    float cameraPositionX = uniforms.viewMatrix.columns[0][0];
    float zoomFactor = 1.0 + smoothstep(5.0, 15.0, abs(vertexOutput.time - 48.0));
    float angleOffset = 3.0 + 0.05 * vertexOutput.time + 6.0 * cameraPositionX / 2732;
    float3 rayOrigin = zoomFactor * float3(2.0 * cos(angleOffset), 1.0, 2.0 * sin(angleOffset));
    float3 rayTarget = vertexOutput.RayDir;//float3(1.0, 0.0, 0.0);
    float3x3 cameraMatrix = setCamera(rayOrigin, rayTarget, 0.35);
    sampler samplerState;
    
    // Calculate ray direction for this fragment
    float3 rayDir = normalize(cameraMatrix * vertexOutput.RayDir);
    
    
    //outputColor.xyz = render(vertexOutput.RayOri, vertexOutput.RayDir, stars, map, samplerState, vertexOutput.time);
    outputColor.xyz = render(rayOrigin , rayDir, map, stars, samplerState, vertexOutput.time);
    
    // Apply a vignette effect based on fragment position
    //float2 q = vertexOutput.position.xy / 2732;
    //outputColor.xyz *= 0.2 + 0.8 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.1);
    
    return outputColor;
}



//vertex ColorInOut vertexShader(Vertex in [[stage_in]],
//                               ushort amp_id [[amplification_id]],
//                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]])
//{
//    ColorInOut out;
//
//    Uniforms uniforms = uniformsArray.uniforms[amp_id];
//
//    float4 position = float4(in.position, 1.0);
//    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
//
//
//    out.texCoord = in.texCoord;
//
//    return out;
//}
//
//fragment float4 fragmentShader(ColorInOut input [[stage_in]],
//               texture2d<float> iChannel0 [[texture(0)]],
//               texture2d<float> iChannel1 [[texture(1)]],
//               sampler sam [[sampler(0)]],
//                constant float &iTime [[buffer(0)]], //cuurent time in seconds
//                constant float2 &iResolution [[buffer(1)]], //it seems this should be float3
//               constant int &iFrame [[buffer(2)]])
//{
//    float3 accumulatedColor = float3(0.0);
//
//
//    // number of samples per axis for antialiasing, adjust as needed
//    int ZERO = min(iFrame,0); //use whichever is lower between 0 and iFrame
//
//    for( int sampleX = ZERO; sampleX < AA; ++sampleX){
//        for( int sampleY = ZERO; sampleY < AA; ++sampleY){
//
//            //calculate the offset for the current sub-pixel
//            //float2 subPixelOffset = float2(float(sampleX),float(sampleY)) / float(AA) - 0.5;
//
//            // Calculate sub-pixel offset for anti-aliasing
//            float2 subPixelOffset = (float2(sampleX, sampleY) + 0.5) / float(AA) - 0.5;
//
//            //adjust coords with sub
////            float2 adjustedCoordinates = (2.0 * (input.texCoord + subPixelOffset) - iResolution) / iResolution.y;
//
//
//            // Adjust texture coordinates with sub-pixel offset and normalize
//                        float2 adjustedCoordinates = (2.0 * (input.texCoord + subPixelOffset) - 1.0) * float2(iResolution.x / iResolution.y, 1.0);
//
//
////            float3 rayDirection = normalize(float3(adjustedCoordinates, -2.0));
//
//            // Calculate ray direction for current sub-pixel
//            float3 rayDirection = normalize(float3(adjustedCoordinates, -1.0));
//
//            //float3 rayOrigin;
//
//            // Define or calculate the ray origin based on your scene setup
//            // This example assumes a fixed origin for simplicity
//            float3 rayOrigin = float3(0.0, 0.0, 5.0); // Example origin
//
//            //accumulatedColor += render(rayOrigin,rayDirection,iChannel0,iChannel1,sam,iTime);
//            // Accumulate color from rendering function
//            accumulatedColor += render(rayOrigin, rayDirection, iChannel0, iChannel1, sam, iTime);
//        }
//    }
//
//    // Average the accumulated color by the number of samples
//    accumulatedColor /= float(AA * AA);
//
//    // Apply a simple vignette effect based on texture coordinates to reduce brightness at the edges
//    float2 normalizedCoordinates = input.texCoord/iResolution;
////    accumulatedColor * = 0.2 + 0.8 * pow(16.0, normalizedCoordinates.x *normalizedCoordinates.y * (1.0 - normalizedCoordinates.x) * 91.0 - normalizedCoordinates.y),0.1);
//
//    float vignette = 0.2 + 0.8 * pow(16.0 * normalizedCoordinates.x * normalizedCoordinates.y * (1.0 - normalizedCoordinates.x) * (1.0 - normalizedCoordinates.y), 0.1);
//    accumulatedColor *= vignette;
//
//
//    // Convert the accumulated linear color to sRGB space before output
//    float4 outputColor = float4(pow(accumulatedColor, 1.0/2.2), 1.0);//this should be skipped later
//    return outputColor;// so should this
//
//    {
//        // pixel coordinates
//        float2 o = float2(float(m),float(n)) / float(AA) - 0.5;
//        float2 p = ((2 * input.texCoord)-1); // (2.0*(fragCoord+o)-iResolution.xy)/iResolution.y;
//        float zo = 1.0 + smoothstep( 5.0, 15.0, abs(iTime - 48.0) );
//        float an = 3.0 + 0.05* iTime + 6.0 * cameraPosition;//iMouse.x/iResolution.x;
//        float3 ro = zo * float3(2.0 * cos(an), 1.0, 2.0 * sin(an));
//        float3 rt = float3( 1.0, 0.0, 0.0 );
//        mat3 cam = setCamera( ro, rt, 0.35 );
//        float3 rd = normalize( cam * float3( p, -2.0) );
//        col += render( ro, rd ); //ray origin and ray direction returns color based on light, shadow, relections e.t.c.
//    }
//
//    col /= float(AA*AA);
//    float2 q = fragCoord / iResolution.xy;
//    col *= 0.2 + 0.8*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1 );
//    fragColor = float4( col, 1.0 );
//
//}
