
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
#define planetNumber 8

//float4 returnSPH(){
 //   float4 sph1 = float4(0.0,0.0,0.0,0.80);
//    return sph1;
//}

// Signed Distance Function for a sphere representing a planet
float sdSphere(float3 position, float3 center, float radius) {
    return length(position - center) - radius;
}

float sdSaturnSphere(float3 position, float3 center, float3 radius) {
    
    float ko = length(position-center/radius);
    float k1 = length(position-center/(radius*radius));
    return ko*(ko-1.0)/k1;
}

float sdSphereBig(float3 position, float radius) {
    return length(position) - radius;
}

float sdSphereSmall(float3 position, float radius) {
    return  radius - length(position);
}

float sdf_flatDisk(float3 position, float3 center, float bigR, float smallR){
    float outerDisk = sdSphereBig(position, bigR);
    float innerDisk = sdSphereSmall(position, smallR);
    return max(outerDisk,-innerDisk);
}


float sdSaturn(float3 position,float3 center, float3 planetRadius, float2 ringRadius){
    float distancePlanet = sdSaturnSphere(position, center, planetRadius);
    float3 flatDiskPosition = position;
    float distanceRings = sdf_flatDisk(position, center, ringRadius.x, ringRadius.y);
    float combinedDistance = min(distancePlanet,distanceRings);
    return combinedDistance;
    
}

typedef struct {
    float distance;
    int planetIndex;
} IntersectionResult;


//// Planet properties;
//constant float3 planetPositions[] = {
//    {2.0, 3.0, 0.0},
//    {1.0, 2.0, 0.3},
//    {3.7, 1.0, 0.0}
//
//}; // Positions of two planets
//
//constant float planetRadii[] = {
//    0.3,
//    1.0,
//    0.5,
//}; // Radii of two planets
// Planet properties;
constant float3 planetPositions[] = {
    {-8.0, 0.5, 1.0},  // Mercury
    {-6.0, -0.5, -1.5}, // Venus
    {-4.0, 0.0, 2.0},   // Earth
    {-2.0, 1.0, -2.5},  // Mars
    {5.0, -1.5, 3.0},   // Jupiter
    {10.0, 2.0, -4.0},  // Saturn
    {15.0, -2.0, 5.0},  // Uranus
    {20.0, 1.5, -6.0}   // Neptune
}; // Positions of eight planets in a more varied layout

constant float planetRadii[] = {
    0.05*3, // Mercury
    0.12*3, // Venus
    0.13*3, // Earth
    0.07*3, // Mars
    0.45*3, // Jupiter
    0.4*3,  // Saturn
    0.3*3,  // Uranus
    0.29*3  // Neptune
}; // Radii of eight planets, scaled for visualization

float sphereIntersect(float3 rayOrigin, float3 rayDirection, float4 sphere) {
    // Offset from ray origin to sphere center
    float3 originToCenter = rayOrigin - sphere.xyz;
    
    // 'b' is part of the quadratic equation components (dot product of ray direction and the origin-to-center vector)
    float bComponent = dot(rayDirection, originToCenter);
    
    // 'c' is another part of the quadratic equation, representing the square of the distance from the origin to the sphere center minus the sphere's radius squared
    float cComponent = dot(originToCenter, originToCenter) - sphere.w * sphere.w;
    
    // Discriminant of the quadratic equation, used to determine if there is an intersection
    float discriminant = bComponent * bComponent - cComponent;
    
    // If the discriminant is positive, there's an intersection
    if (discriminant > 0.0) {
        // The distance to the intersection point along the ray direction
        float distanceToIntersection = -bComponent - sqrt(discriminant);
        return distanceToIntersection;
    }
    
    return -1.0; // Return -1 to indicate no intersection
}


//Manually enter surface distances
//EARTH
float sphereSurfaceDistance(float3 rayOrigin, float3 rayDirection, float4 sphere) {
    float3 originToCenter = rayOrigin - sphere.xyz; // Vector from ray origin to sphere center
    float projectionLength = dot(originToCenter, rayDirection); // Length of the projection of originToCenter onto rayDirection
    float closestApproach = dot(originToCenter, originToCenter) - projectionLength * projectionLength; // Closest approach squared
    return sqrt(max(0.0, closestApproach)) - sphere.w; // Subtract sphere radius to get distance to surface
}



// END OF SURFACE DISTANCES


float sphereSoftShadow(float3 rayOrigin, float3 rayDirection, float4 sphere, float penumbraFactor) {
    float3 centerToOrigin = sphere.xyz - rayOrigin; // Vector from sphere center to ray origin
    float projectionLength = dot(centerToOrigin, rayDirection); // Length of the projection onto rayDirection
    float occlusionCheck = dot(centerToOrigin, centerToOrigin) - sphere.w * sphere.w; // Sphere occlusion check squared
    float discriminant = projectionLength * projectionLength - occlusionCheck; // Discriminant for shadow calculation
    return (projectionLength < 0.0) ? 1.0 : 1.0 - smoothstep(0.0, 1.0, penumbraFactor * discriminant / projectionLength); // Shadow intensity
}

float3 sphNormal(  float3 pos,  float4 sph )
{
    return (pos - sph.xyz)/sph.w;
}

float3 DirectionalTextureSampling(texture2d<float> texture, sampler sam,  float3 direction,  float s,  float b )
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


float2 hashPseudoRandom(float2 position) {
    position = float2(dot(position, float2(127.1, 311.7)), dot(position, float2(269.5, 183.3)));
    return fract(sin(position) * 43758.5453);
}

float2 voronoiPattern(float2 point) {
    float2 cellIndex = floor(point); // The cell index in the Voronoi diagram
    float2 cellFraction = fract(point); // The fractional part within the Voronoi cell

    float3 minimumDistance = float3(8.0); // Initialize with a large distance

    for (int yOffset = -1; yOffset <= 1; yOffset++) {
        for (int xOffset = -1; xOffset <= 1; xOffset++) {
            float2 neighborCell = float2(float(xOffset), float(yOffset));
            float2 randomOffset = hashPseudoRandom(cellIndex + neighborCell);
            float2 pointToCellVector = neighborCell - cellFraction + randomOffset; // Vector from the point to this cell
            float distanceSquared = dot(pointToCellVector, pointToCellVector);

            if (distanceSquared < minimumDistance.x) {
                minimumDistance = float3(distanceSquared, randomOffset);
            }
        }
    }

    return float2(sqrt(minimumDistance.x), minimumDistance.y + minimumDistance.z);
}


//=======================================================

float3 background( float3 viewDirection, 
                  float3 lightDirection,
                            texture2d<float> environmentTexture,
                            sampler textureSampler)
{
    float3 backgroundColor = float3(0.0);

    // Apply variations to the background color using the fancyCube function to sample the environment texture
    backgroundColor += 0.5 * pow(fancyCube(environmentTexture, textureSampler, viewDirection, 0.05, 5.0).zyx, float3(2.0));
    backgroundColor += 0.2 * pow(fancyCube(environmentTexture, textureSampler, viewDirection, 0.10, 3.0).zyx, float3(1.5));
    backgroundColor += 0.8 * float3(0.80, 0.5, 0.6) * pow(fancyCube(environmentTexture, textureSampler, viewDirection, 0.1, 0.0).xxx, float3(6.0));

    // Calculate star intensity based on the sampled value from fancyCube
    float starIntensity = smoothstep(0.3, 0.7, fancyCube(environmentTexture, textureSampler, viewDirection, 0.91, 0.0).x);

    // Normalize direction for volumetric effect calculation
    float3 normalizedDirection = abs(viewDirection);
    normalizedDirection = normalizedDirection * normalizedDirection * normalizedDirection;

    // Generate procedural patterns using Voronoi function for additional texture in the background
    float2 patternXY = voronoiPattern(50.0 * viewDirection.xy);
    float2 patternYZ = voronoiPattern(50.0 * viewDirection.yz);
    float2 patternZX = voronoiPattern(50.0 * viewDirection.zx);
    float2 combinedPattern = (patternYZ * normalizedDirection.x + patternZX * normalizedDirection.y + patternXY * normalizedDirection.z) / (normalizedDirection.x + normalizedDirection.y + normalizedDirection.z);
    backgroundColor += 0.5 * starIntensity * clamp(1.0 - (3.0 + combinedPattern.y * 5.0) * combinedPattern.x, 0.0, 1.0);

    // Adjust the final color with a base color and highlight based on view direction and light direction
    backgroundColor = 1.5 * backgroundColor - 0.2;
    backgroundColor += float3(-0.05, 0.1, 0.0);

    // Calculate the amount of light hitting the surface
    float lightAmount = clamp(dot(viewDirection, lightDirection), 0.0, 1.0);
    backgroundColor += 0.4 * pow(lightAmount, 5.0) * float3(1.0, 0.7, 0.6) * 2.0;
    backgroundColor += 0.4 * pow(lightAmount, 64.0) * float3(1.0, 0.9, 0.8) * 2.0;
    
    return backgroundColor;
}


//--------------------------------------------------------------------


// Helper function to find the nearest intersection with any planet
// Function to find the nearest intersection with any planet
IntersectionResult findNearestIntersection(float3 rayOrigin, float3 rayDirection) {
    IntersectionResult result;
    result.distance = INFINITY;
    result.planetIndex = -1; // -1 indicates no intersection
    
    for (int i = 0; i < planetNumber; ++i) {
        float4 sphere = float4(planetPositions[i], planetRadii[i]);
        float distance = sphereIntersect(rayOrigin, rayDirection, sphere);
        
        if (distance > 0.0 && distance < result.distance) {
            result.distance = distance;
            result.planetIndex = i;
        }
    }
    
    return result;
}

float calculateMinimumDistanceToPlanets(float3 position) {
    float minDistance = INFINITY; // Start with a large number to find the minimum distance
    for (unsigned int i = 0; i < planetNumber; ++i) { // Iterate over planets
        float distance = sdSphere(position, planetPositions[i], planetRadii[i]);
        minDistance = min(minDistance, distance); // Find the closest distance
    }
    return minDistance;
}


float marchRayThroughSpace2(float3 rayOrigin, float3 rayDirection, float maxDistance) {
    float traveledDistance = 0.0;

    // Check for intersection with a horizontal plane at y = 1.0
    float initialHeightIntersection = (1.0 - rayOrigin.y) / rayDirection.y;
    if (initialHeightIntersection > 0.0) traveledDistance = initialHeightIntersection;

    // March the ray
    for (int step = 0; step < 20; step++) {
        float3 currentPosition = rayOrigin + traveledDistance * rayDirection;
        float heightAdjustment = calculateMinimumDistanceToPlanets(currentPosition);
        if (heightAdjustment < 0.001 || traveledDistance > maxDistance) break; // Stop if close to surface or max distance reached
        traveledDistance += heightAdjustment;
    }
    return traveledDistance;
}

float marchRayThroughSpace(float3 rayOrigin, float3 rayDirection, float maxDistance) {
    float traveledDistance = 0.0;

    // Check for intersection with a horizontal plane at y = 1.0
    float initialHeightIntersection = (1.0 - rayOrigin.y) / rayDirection.y;
    if (initialHeightIntersection > 0.0 && initialHeightIntersection < maxDistance) {
        traveledDistance = initialHeightIntersection;
    }

    // March the ray
    for (int step = 0; step < 20; step++) {
        float3 currentPosition = rayOrigin + traveledDistance * rayDirection;
        float heightAdjustment = calculateMinimumDistanceToPlanets(currentPosition);
        if (heightAdjustment < 0.001 || traveledDistance >= maxDistance) {
            // Stop if close to surface or max distance reached
            break;
        }
        traveledDistance += heightAdjustment;
    }

    // Ensure traveledDistance does not exceed maxDistance
    traveledDistance = min(traveledDistance, maxDistance);

    return traveledDistance;
}

// Renders the scene, calculating lighting, reflections, and environmental effects.
float3 render2( float3 rayOrigin, float3 rayDirection, texture2d<float> environmentTexture, texture2d<float> cloudTexture, sampler textureSampler, float time) {

    // Light direction for the scene
    float3 lightDirection = normalize(float3(1.0, 0.2, 1.0));

    // Initial color calculated based on the background function
    float3 color = background(rayDirection, lightDirection, cloudTexture, textureSampler);
    //float3 color = float3 (1,0,0);

    // Perform ray tracing to detect sphere intersection
    IntersectionResult intersection = findNearestIntersection(rayOrigin, rayDirection);
    float4 sph = float4(planetPositions[intersection.planetIndex], planetRadii[intersection.planetIndex]);
    
    // If there is an intersection, calculate the color at the intersection point
    if (intersection.distance > 0.0 && intersection.planetIndex >= 0) {

        // Base material color
        float3 materialColor = float3(0.18);
        // Calculate the intersection position
        float3 intersectionPosition = rayOrigin + intersection.distance * rayDirection;
        
        // Calculate normal at the intersection point
        float3 normal = sphNormal(intersectionPosition, sph);

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
        
        
//        if (intersection.planetIndex == 6) {
//            float distance = sphereSurfaceDistance(rayOrigin, rayDirection, sph);
//            // Calculate normalized direction from the ray origin to the sphere center
//            float3 dirToCenter = normalize(sph.xyz - rayOrigin);
//            // Assuming the equatorial plane is perpendicular to the Y-axis
//            float equatorialPlaneDistance = length(dirToCenter.xy); // Distance in the equatorial plane
//            float absDistanceFromEquator = abs(distance) - sph.w; // Absolute distance from the sphere surface
//            float ringWidth = 5.5 * sph.w; // Adjust ring width as needed
//
//            // Calculate ring intensity based on distance from the equator and within a certain width
//            float ringIntensity = smoothstep(ringWidth, 0.0, abs(absDistanceFromEquator - sph.w * 3.1));
//
//            // Define base glow color
//            float3 baseGlowColor = float3(0.6, 0.7, 1.0);
//
//            // Apply ring intensity to the glow
//            float3 glow = baseGlowColor * ringIntensity;
//
//            // Final color adjustment
//            color += glow * 1.5;
//        }
        if (intersection.planetIndex == 6) {
            // Compute the direction to the intersection point from the planet center
            float3 dirToIntersection = normalize(intersectionPosition - sph.xyz);
            // Assuming the ring plane is perpendicular to the Y-axis, calculate the height above/below this plane
            float heightAboveRingPlane = dot(dirToIntersection, float3(0, 1, 0)) * length(intersectionPosition - sph.xyz);
            // Determine if the intersection point is within the ring's plane (with some thickness)
            float ringPlaneThickness = 0.95 * sph.w; // Adjust for the thickness of the rings
            bool withinRingPlane = abs(heightAboveRingPlane) < ringPlaneThickness;
            
            // Additionally, check if we're within the radial bounds of the rings
            float radialDistanceFromPlanetCenter = length(intersectionPosition.xy - sph.xy);
            bool withinRadialBounds = radialDistanceFromPlanetCenter > sph.w && radialDistanceFromPlanetCenter < sph.w * 1.5; // Example bounds
            
            if (withinRingPlane && withinRadialBounds) {
                // Define the ring's glow effect
                float3 ringGlowColor = float3(0.6, 0.7, 1.0);
                float ringGlowIntensity = 1.0 - smoothstep(sph.w, sph.w * 1.5, radialDistanceFromPlanetCenter); // Falloff based on distance
                color += ringGlowColor * ringGlowIntensity * 10.5; // Adjust intensity as needed
            }
        }

    }

    float maximumDistance = 20.0;

    // Perform ray marching to find the distance traveled or detect any obstacle
    float traveledDistance = marchRayThroughSpace(rayOrigin, rayDirection, maximumDistance);

    // Determine if the ray has hit the background by checking if it has traveled close to the maximum distance
    // without significant obstruction and ensuring no closer planet intersection was detected
    bool isBackground = traveledDistance >= (maximumDistance - 0.01) && intersection.planetIndex < 0;

    // Apply the wireframe effect only if the ray is considered to have hit the background
    if (isBackground) {
        // Normalize traveledDistance within a visible range for the wireframe effect
        // Adjusting traveledDistance directly for effect application
        float effectiveDistance = traveledDistance / maximumDistance; // Normalize to [0,1]
        //effectiveDistance *= 5; // Apply scaling to enhance visibility if necessary

        // Calculate position for wireframe effect using the original traveledDistance
        float3 position = rayOrigin + (rayDirection * maximumDistance); // Use maximumDistance to ensure it's the background position
        
        // Calculate distance from the position to the planet's surface
        float distanceToPlanetSurface = max(length(position - sph.xyz) - sph.x, 0.0);
        float influenceFactor = (length(rayOrigin - sph.xyz));
        float2 patternOffset = float2(influenceFactor * 0.1);
        
        
        // Apply distortion based on distance to planet surface
        float distortionFactor = 1.0 / (1.0 + distanceToPlanetSurface * 0.2); // Adjust 0.1 to control the distortion strength
        
        // Distort the wireframe pattern using the distortion factor
        float2 sinPattern = sin(distortionFactor * 2.0 * 16.2831 * (position.xz + patternOffset));

        //float2 sinPattern = sin(2.0 * 6.2831 * sph.xz);
        
        float3 wireframeColor = float3(0.0);
        wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.x));
        wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.y));
        wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.x));
        wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.y));
        wireframeColor *= 0.2 + 1.0 * sphereSoftShadow(position, lightDirection, sph, 4.0);
        // The visibility effect is adjusted by effectiveDistance, not traveledDistance
        //color += wireframeColor * 0.5 * exp(-0.05 * effectiveDistance * effectiveDistance)*.81;
        
 /* testing*/
        // Distort the wireframe pattern using the distortion factor
        float2 sinPattern2 = sin(distortionFactor * 2.0 * 16.2831 * (position.xy + patternOffset));
        float3 wireframeColor2 = float3(0.0);
        wireframeColor2 += 1.0 * exp(-12.0 * abs(sinPattern2.x));
        //wireframeColor2 += 1.0 * exp(-12.0 * abs(sinPattern.y));
        //wireframeColor2 += 0.5 * exp(-4.0 * abs(sinPattern.x));
        //wireframeColor2 += 0.5 * exp(-4.0 * abs(sinPattern.y));
        //wireframeColor2 *= 0.2 + 1.0 * sphereSoftShadow(position, lightDirection, sph, 4.0);
        // The visibility effect is adjusted by effectiveDistance, not traveledDistance
        color += wireframeColor2 * 0.5 * exp(-0.05 * effectiveDistance * effectiveDistance)*.81;
        
/* testing*/

        
        
    }
    
    // Outer glow effect for objects
//    if (dot(rayDirection, sph.xyz - rayOrigin) > 0.0) {
//        float distance = sphereSurfaceDistance(rayOrigin, rayDirection, sph);
//        float3 glow = float3(0.0);
//        glow += float3(0.6, 0.7, 1.0) * 0.3 * exp(-2.0 * abs(distance)) * step(0.0, distance);
//        glow += 0.6 * float3(0.6, 0.7, 1.0) * 0.3 * exp(-8.0 * abs(distance));
//        glow += 0.6 * float3(0.8, 0.9, 1.0) * 0.4 * exp(-100.0 * abs(distance));
//        color += glow * 1.5;
//    }

    
    // Apply time-based fading to the color
    color *= smoothstep(0.0, 6.0, time);

    return color;
    
//    if (intersection.distance > 0.0 && intersection.planetIndex == 1){
//        return float3(1,0,0);
//    }
}



//// Helper function to calculate glow effect for planets
//float3 calculateGlowForPlanet(float3 rayOrigin, float3 rayDirection, float4 sph) {
//    if (dot(rayDirection, sph.xyz - rayOrigin) > 0.0) {
//        float distance = sphereSurfaceDistance(rayOrigin, rayDirection, sph);
//        float3 glow = float3(0.0);
//        glow += float3(0.6, 0.7, 1.0) * 0.3 * exp(-2.0 * abs(distance)) * step(0.0, distance);
//        glow += 0.6 * float3(0.6, 0.7, 1.0) * 0.3 * exp(-8.0 * abs(distance));
//        glow += 0.6 * float3(0.8, 0.9, 1.0) * 0.4 * exp(-100.0 * abs(distance));
//        return glow * 1.5;
//    }
//    return float3(0.0);
//}
//
//// Helper function to calculate wireframe effect for the background
//float3 calculateWireframeEffect(float3 position, float3 lightDirection, float4 sph, float intersectionDistance) {
//    float2 sinPattern = sin(2.0 * 6.2831 * position.xz);
//    float3 wireframeColor = float3(0.0);
//    wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.x));
//    wireframeColor += 1.0 * exp(-12.0 * abs(sinPattern.y));
//    wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.x));
//    wireframeColor += 0.5 * exp(-4.0 * abs(sinPattern.y));
//    wireframeColor *= 0.2 + 1.0 * sphereSoftShadow(position, lightDirection, sph, 4.0);
//    return wireframeColor * 0.5 * exp(-0.05 * intersectionDistance * intersectionDistance);
//}
//
//// Renders the scene, calculating lighting, reflections, and environmental effects.
//float3 render(float3 rayOrigin, float3 rayDirection, texture2d<float> environmentTexture, texture2d<float> cloudTexture, sampler textureSampler, float time) {
//
//    // Light direction for the scene
//    float3 lightDirection = normalize(float3(1.0, 0.2, 1.0));
//
//    // Initial color calculated based on the background function
//    float3 color = background(rayDirection, lightDirection, cloudTexture, textureSampler);
//
//    // Perform ray tracing to detect sphere intersection
//    float intersectionDistance;
//    IntersectionResult intersection = findNearestIntersection(rayOrigin, rayDirection);
//    if (intersection.distance > 0.0 && intersection.planetIndex >= 0) {
//        // Only process glow effect for planets here
//        float4 sph = float4(planetPositions[intersection.planetIndex], planetRadii[intersection.planetIndex]);
//
//        // Calculate the color at the intersection point, including the glow effect for planets
//        // (Planet rendering logic here, including glow effect)
//        // Note: Detailed planet rendering and glow effect logic goes here.
//        // For brevity, it's not fully repeated. Focus is on the structure for glow effect application.
//        float3 glow = calculateGlowForPlanet(rayOrigin, rayDirection, sph);
//        color += glow;
//    }
//
//    // Raymarch additional elements for the background
//    float maximumDistance = 20.0;
//    if (intersection.distance > 0.0) maximumDistance = intersection.distance;
//    intersectionDistance = marchRayThroughSpace(rayOrigin, rayDirection, maximumDistance);
//    if (intersectionDistance < maximumDistance) {
//        // Calculate position and apply wireframe effects for the background
//        float3 position = rayOrigin + intersectionDistance * rayDirection;
//        float3 wireframeColor = calculateWireframeEffect(position, lightDirection,float4(planetPositions[intersection.planetIndex], planetRadii[intersection.planetIndex]),intersectionDistance);
//        color += wireframeColor;
//    }
//
//    // Apply time-based fading to the color
//    color *= smoothstep(0.0, 6.0, time);
//
//    return color;
//}
//
//
///////
//
//
//// Function to calculate the base color of the planet
//float3 calculatePlanetColor3(float3 intersectionPosition, float3 normal, float3 lightDirection, texture2d<float> environmentTexture, sampler textureSampler, float time) {
//    // Diffuse lighting factor
//    //float diffuse = max(dot(normal, lightDirection), 0.0);
//    // Simple base color for demonstration, can be replaced with texture sampling or more complex logic
//    //float3 baseColor = float3(0.4, 0.5, 0.6);
//    //return baseColor * diffuse;
//    
//    
//    // Base material color
//    float3 materialColor = float3(0.18);
//    // Calculate the intersection position
//    //intersectionPosition = rayOrigin + intersection.distance * rayDirection;
//    
//    // Calculate normal at the intersection point
//    //float3 normal = sphNormal(intersectionPosition, sph);
//
//    // Animation modifiers based on time
//    float animationModifier = 0.1 * time;
//    float2 phaseRotation = float2(cos(animationModifier), sin(animationModifier));
//    // Transformed normal for animation effect
//    float3 transformedNormal = normal;
//    transformedNormal.xz = float2x2(phaseRotation.x, -phaseRotation.y, phaseRotation.y, phaseRotation.x) * transformedNormal.xz;
//
//    // Additional animation effect
//    float animationModifier2 = 0.08 * time - 1.0 * (1.0 - normal.y * normal.y);
//    phaseRotation = float2(cos(animationModifier2), sin(animationModifier2));
//    float3 transformedNormal2 = normal;
//    transformedNormal2.xz = float2x2(phaseRotation.x, -phaseRotation.y, phaseRotation.y, phaseRotation.x) * transformedNormal2.xz;
//
//    // Reflection calculation for reflective materials
//    float3 reflection = reflect(rayDirection, normal);
//    float fresnelEffect = clamp(1.0 + dot(normal, rayDirection), 0.0, 1.0);
//
//    // Lighting effects based on the environment texture
//    float lightingEffect = fancyCube(environmentTexture, textureSampler, transformedNormal, 0.03, 0.0).x;
//    lightingEffect += -0.1 + 0.3 * fancyCube(environmentTexture, textureSampler, transformedNormal, 8.0, 0.0).x;
//
//    // Calculate sea and land colors based on lighting and Fresnel effects
//    float3 seaColor = mix(float3(0.0, 0.07, 0.2), float3(0.0, 0.01, 0.3), fresnelEffect) * 0.15;
//    float3 landColor = mix(float3(0.02, 0.04, 0.0), float3(0.05, 0.1, 0.0), smoothstep(0.4, 1.0, fancyCube(environmentTexture, textureSampler, transformedNormal, 0.1, 0.0).x)) * fancyCube(environmentTexture, textureSampler, transformedNormal, 0.3, 0.0).xyz * 0.5;
//
//    // Determine the material color based on the calculated sea and land colors
//    float landOrSea = smoothstep(0.45, 0.46, lightingEffect);
//    materialColor = mix(seaColor, landColor, landOrSea);
//
//    // Cloud effects
//    float3 cloudWrapEffect = -1.0 + 2.0 * fancyCube(cloudTexture, textureSampler, transformedNormal2.xzy, 0.025, 0.0).xyz;
//    float cloudDensity = fancyCube(cloudTexture, textureSampler, transformedNormal2 + 0.2 * cloudWrapEffect, 0.05, 0.0).y;
//    float clouds = smoothstep(0.3, 0.6, cloudDensity);
//    
//    // Combine material and cloud effects
//    materialColor = mix(materialColor, float3(0.93 * 0.15), clouds);
//
//    // Diffuse lighting calculation
//    float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
//    materialColor *= 0.8;
//    float3 linearColor = float3(3.0, 2.5, 2.0) * diffuse;
//    linearColor += 0.01;
//    color = materialColor * linearColor;
//    color = pow(color, float3(0.4545)); // Gamma correction
//    color += 0.6 * fresnelEffect * fresnelEffect * float3(0.9, 0.9, 1.0) * (0.3 + 0.7 * diffuse);
//
//    // Specular highlights
//    float specular = clamp(dot(reflection, lightDirection), 0.0, 1.0);
//    float totalSpecular = pow(specular, 3.0) + 0.5 * pow(specular, 16.0);
//    color += (1.0 - 0.5 * landOrSea) * clamp(1.0 - 2.0 * clouds, 0.0, 1.0) * 0.3 * float3(0.5, 0.4, 0.3) * totalSpecular * diffuse;
//}
//
//// Function to calculate the glow effect for planets
//float3 calculateGlowEffect3(float distanceToSurface, float3 normal, float3 lightDirection) {
//    // Glow intensity decreases with distance to surface
//    float glowIntensity = exp(-distanceToSurface * 10.0);
//    // Glow color - can be adjusted
//    float3 glowColor = float3(0.65, 0.75, 1.0);
//    return glowColor * glowIntensity;
//}
//
//// Function to calculate the wireframe effect for the background
//float3 calculateWireframeEffect3(float3 position, float3 lightDirection) {
//    // Create a grid pattern based on the position
//    float2 gridPattern = sin(position.xz * 2.0 * 3.14159) * sin(position.yz * 2.0 * 3.14159);
//    float wireIntensity = smoothstep(0.95, 1.0, abs(gridPattern.x)) * smoothstep(0.95, 1.0, abs(gridPattern.y));
//    // Wireframe color - can be adjusted
//    float3 wireColor = float3(0.1, 0.2, 0.3);
//    return wireColor * wireIntensity;
//}
//
//// Renders the scene, calculating lighting, reflections, and environmental effects.
//float3 render3(float3 rayOrigin, float3 rayDirection, texture2d<float> environmentTexture, texture2d<float> cloudTexture, sampler textureSampler, float time) {
//    // Light direction for the scene
//    float3 lightDirection = normalize(float3(1.0, 0.2, 1.0));
//
//    // Initial color calculated based on the background function
//    // This function should calculate the background color, possibly using the environmentTexture
//    float3 color = background(rayDirection, lightDirection, cloudTexture, textureSampler);
//
//    // Perform ray tracing to detect sphere intersection
//    IntersectionResult intersection = findNearestIntersection(rayOrigin, rayDirection);
//
//    // If the ray intersects a planet, calculate its color, including glow effect
//    if (intersection.distance > 0.0 && intersection.planetIndex >= 0) {
//        // Retrieve the sphere data for the intersected planet
//        float4 sph = float4(planetPositions[intersection.planetIndex], planetRadii[intersection.planetIndex]);
//
//        // Calculate the base color of the planet at the intersection point
//        float3 intersectionPosition = rayOrigin + intersection.distance * rayDirection;
//        float3 normal = normalize(intersectionPosition - sph.xyz);
//        float3 planetColor = calculatePlanetColor(intersectionPosition, normal, lightDirection, environmentTexture, textureSampler);
//
//        // Apply glow effect around the planet
//        float distanceToSurface = length(intersectionPosition - sph.xyz) - sph.w;
//        float3 glowEffect = calculateGlowEffect(distanceToSurface, normal, lightDirection);
//        color += planetColor + glowEffect;
//    } else {
//        // For the background, apply the wireframe effect
//        // This effect simulates a grid or wireframe appearance for non-planet objects or empty space
//        float3 backgroundPosition = rayOrigin + 20.0 * rayDirection; // Arbitrary distance for background effect
//        color += calculateWireframeEffect(backgroundPosition, lightDirection);
//    }
//
//    // Apply time-based fading to the color to simulate environmental changes or effects over time
//    color *= smoothstep(0.0, 6.0, time);
//
//    return color;
//}





// Sets up the camera matrix based on the camera's position, target, and roll angle.
float3x3 setCamera(float3 cameraPosition, float3 cameraTarget, float cameraRoll) {
    float3 forwardVector = normalize(cameraTarget - cameraPosition);
    float3 rightVector = float3(sin(cameraRoll), cos(cameraRoll), 0.0);
    float3 upVector = normalize(cross(forwardVector, rightVector));
    float3 correctedRightVector = normalize(cross(upVector, forwardVector));
    return float3x3(correctedRightVector, upVector, -forwardVector);
}

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



vertex VertexOut vertexShaderAll(const VertexIn in [[stage_in]],
                                 uint vertexID [[vertex_id]],
                              constant UniformsArray &uniformsArray [[buffer(1)]]){

    VertexOut out;


    // access the first set of uniforms
    constant Uniforms& uniforms = uniformsArray.uniforms[0];



    out.position = uniforms.projectionMatrix  * uniforms.viewMatrix * uniforms.modelMatrix * float4(in.position, 1.0f);
    
    
    // Define fullscreen quad vertices
    float2 positions[6] = {

        float2(-1.0, 1.0),
        float2(-1.0, -1.0),
        float2(1.0, -1.0),
        
        float2(-1.0, 1.0),
        float2(1.0, -1.0),
        float2(1.0, 1.0),
    };

    // Output clip-space position
    out.position = float4(positions[vertexID], 0.1, 1.0);
    
    
    
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



fragment float4 fragmentShaderAll(VertexOut vertexOutput [[stage_in]],
                               constant UniformsArray &shaderUniformsArray [[buffer(0)]],
                               texture2d<float> stars [[texture(2)]],
                               texture2d<float> map [[texture(1)]]) {

    // Access the first set of uniforms
    constant Uniforms& uniforms = shaderUniformsArray.uniforms[0];
    float4 outputColor = float4(1,0,0,1);
    float cameraPositionX = uniforms.viewMatrix.columns[0][0];
    float zoomFactor = 1.0 + smoothstep(5.0, 15.0, abs(vertexOutput.time - 48.0));
    float angleOffset = 3.0 + 0.05 * vertexOutput.time + 6.0 * cameraPositionX / uniforms.viewportWidth;
    //float3 rayOrigin = zoomFactor * float3(2.0 * cos(angleOffset), 1.0, 2.0 * sin(angleOffset));
    
    // Modify rayOrigin to move the camera outwardly; increase the multiplier for cos and sin to push the camera back
    float3 rayOrigin = zoomFactor * float3(6.0 * cos(angleOffset), 2.5, 3.0 * sin(angleOffset)); // Increased distance

    float3 rayTarget = vertexOutput.RayDir;//float3(1.0, 0.0, 0.0);
    float3x3 cameraMatrix = setCamera(rayOrigin, rayTarget, 0.55);
    sampler samplerState;

    // Calculate ray direction for this fragment
    float3 rayDir = normalize(cameraMatrix * vertexOutput.RayDir);


    //outputColor.xyz = render(vertexOutput.RayOri, vertexOutput.RayDir, stars, map, samplerState, vertexOutput.time);
    outputColor.xyz = render2(rayOrigin , rayDir,map,stars,samplerState, vertexOutput.time);

    // Apply a vignette effect based on fragment position
    float2 q = vertexOutput.position.xy / 2732;
    outputColor.xyz *= 0.2 + 0.8 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.1);

    return outputColor;
    //return float4(1,0,0,1);
}



