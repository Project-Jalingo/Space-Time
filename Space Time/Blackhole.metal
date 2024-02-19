////
////  Blackhole.metal
////  Space Time
////
////  Created by User1 on 2/18/24.
////
//
//#include <metal_stdlib>
//using namespace metal;
//
////BUFFER A
//const vec2 camAngle = vec2(-5.0, 0.0);
//const float FOV = 60.0 * 0.0174533; //Radians Conversion
//void mainImage( out vec4 pixVector, in vec2 fragCoord )
//{
//    //Calculate ray vector
//    vec2 newCam = camAngle;
//    if(iMouse.xy == vec2(0.0)) {newCam = vec2(-1.6, 0.55);}
//    newCam.y = newCam.y + (-0.15 * sin(((iTime + 15.0) / 15.0)) - 0.2);
//    vec2 pixAxis =  newCam + vec2(8.0, 3.5) * (iMouse.xy) / iResolution.xy - 0.5 + FOV * (0.5 + fragCoord) / iResolution.x;
//    
//    
//    //Send ray vectors to next buffer
//    pixVector = vec4(pixAxis, 0.0, 1.0);
//}
//
//
////BUFFER B
//#define hashi(x)   lowbias32(x)
//#define hash(x)  ( float( hashi(x) ) / float( 0xffffffffU ) )
//
//#define steps  90
//const int starAA = 16; //Multisamples the stars
//vec3 camPos = vec3(0.0, 220.0, 0.0);
//const vec3 holePos = vec3(0.0, 0.0, -0.0);
//const float holeRadius = 10.0; //Issues below three with star projection
//const float detail = 3.0;//noise octaves, large performance impact
//const float density = 3.0; //Largest noise value
//const float stepVa = 20.0; //Maximum length of fine raymarching steps
//const float bounds = 500.0; //Boundary distance for black hole
//const float vol = 8.0;    //brightness of volume
//const float volDen = 0.6; //opacity of volume, between 0 and 1
//
//
////bias: 0.17353355999581582 ( very probably the best of its kind )
//uint lowbias32(uint x)
//{
//    x ^= x >> 16;
//    x *= 0x7feb352dU;
//    x ^= x >> 15;
//    x *= 0x846ca68bU;
//    x ^= x >> 16;
//    return x;
//}
//
////Random function and hash from https://www.shadertoy.com/view/WttXWX
//float rand(vec3 position)
//{
//    uvec3 V = uvec3(position);
//    float h = hash( V.x + (V.y<<16) + (V.z<<8));  // Converted 3D hash (should be ok too )
//    return h;
//
//}
//
//
////bilinear interpolation function
//float interpolate(vec3 position)
//{
//    vec3 quantPos = round((position + 0.5));
//    vec3 divPos = fract(1.0 * position);
//    
//
//    //Finds noise values for the corners, treats Z axis as a separate rectangle for ease of lerping
//    vec4 lerpXY = vec4(
//        rand(quantPos + vec3(0.0, 0.0, 0.0)),
//        rand(quantPos + vec3(1.0, 0.0, 0.0)),
//        rand(quantPos + vec3(1.0, 1.0, 0.0)),
//        rand(quantPos + vec3(0.0, 1.0, 0.0)));
//    
//    vec4 lerpXYZ = vec4(
//        rand(quantPos + vec3(0.0, 0.0, 1.0)),
//        rand(quantPos + vec3(1.0, 0.0, 1.0)),
//        rand(quantPos + vec3(1.0, 1.0, 1.0)),
//        rand(quantPos + vec3(0.0, 1.0, 1.0)));
//    
//    //Calculates the area of rectangles
//    vec4 weights = vec4(
//    abs((1.0 - divPos.x) * (1.0 - divPos.y)),
//    abs((0.0 - divPos.x) * (1.0 - divPos.y)),
//    abs((0.0 - divPos.x) * (0.0 - divPos.y)),
//    abs((1.0 - divPos.x) * (0.0 - divPos.y)));
//    
//    //linear interpolation between values
//    vec4 lerpFinal = mix(lerpXY, lerpXYZ, divPos.z);
//   
//    return weights.r * lerpFinal.r +
//           weights.g * lerpFinal.g +
//           weights.b * lerpFinal.b +
//           weights.a * lerpFinal.a;
//    
//}
//
////Octaves of noise, sligtly less than a perfect octave to hide bilinear filtering artifatcs
//float octave(vec3 coord, float octaves, float div)
//{
//    
//    float col;
//    float it = 1.0;
//    float cnt = 1.0;
//    for(float i = 1.0; i <= octaves; i++)
//    {
//        col += interpolate((it * coord / (div))) / it;
//        it = it * 1.9;
//        cnt = cnt + 1.0 / it;
//       
//    }
//    return pow(col / cnt, 1.0);
//}
//
////Procedural starmap
//float starField(vec3 vector)
//{
//    float b;
//    float sizeDiv = 500.0;
//    vector = sizeDiv * (1.0 + normalize(vector));
//    if(starAA > 0)
//    {
//        for(int i = 0; i < starAA; i++)
//        {
//           vector += 50.0 * rand(vec3(i)) / sizeDiv;
//           float a = 1.0 - (4.0 * rand((vector)));
//           if(a < 0.9) {a = 0.0;}
//           a = 6.0 * pow(a, 20.0);
//           b += a;
//        }
//    }
//    
//    return b / float(starAA);
//}
//
////Distance field for accretion disk
//vec2 distField(vec3 position, vec3 origin)
//{
//    //Distance Field inputs
//    float radius = 45.0;
//    float dist = distance(origin, position);
//    float distXY = max(distance(origin.xy, position.xy) - holeRadius, 0.0);
//    float fieldZ = max(0.0, pow(distance((origin.z), position.z), 2.5));
//    
//    
//    //calculates angle to transform a 2d function to a radial one
//    float angle = atan((position.x - origin.x) / (position.y - origin.y));
//    if(position.y <= origin.y) {angle = angle + 3.1415;}
//    angle = angle + 0.2 * iTime;
//   
//    //Distance field components
//    float cloud = pow(clamp(radius / (dist - holeRadius), 0.0, 1.0), 2.5);
//    float i;
//    float spiral;
//    float occ;
//    spiral = octave(vec3(dist, 50.0 * (1.0 + sin(angle))
//    , 2.0 * distance(origin.z, position.z)), detail, density);//3d noise function
//
//    //Merge components
//    float finalDF = cloud * clamp(spiral / (fieldZ), 0.0, 1.0);
//    if(finalDF < volDen){occ =(volDen - spiral);}
//    return vec2(finalDF, max(occ / (dist * distance(position.z, origin.z) / 800.0), 0.0));
//}
//
//
////Function for moving the ray
//vec3 rayCast(vec2 rayAxis)
//{
//    
//    float stepSize;
//    float gravDis = 2.0 * 1.666667 * 2392.3 * pow(1.0973, holeRadius);
//    
//    //Variables to determine position changes and ray vectors
//    float yTravel = camPos.y - holeRadius;
//    float timeOff = (iTime + 12.0) / 15.0;
//    vec3 newCam = vec3(camPos.x, camPos.y + cos(timeOff) * (yTravel), sin(timeOff) * 30.0);
//    vec3 rayPos = newCam;
//    vec3 rayVel = vec3(cos(rayAxis.x), sin(rayAxis.x), sin(rayAxis.y));
//    float rayDist = distance(rayPos, holePos);
//    float rayVol;
//    vec2 dField;
//    float colShift;
//    float occ = 1.0;
//    //Jump the ray forward to allow it to render the black hole at large distances
//    rayPos += rayVel * max(rayDist - bounds, 0.0);
//    
//    for(int i = 0; i <= steps; i++)
//    {
//        rayDist = distance(rayPos, holePos);
//        float boDist = pow(rayDist / 500.0, 2.0);
//        float diskDist = rayDist;//distance(rayPos.xy, holePos.xy);
//        
//        rayPos += rayVel;
//        
//        //vector of deflection
//        vec3 rayDefl = normalize(holePos - rayPos);
//        
//       
//       //Deflect the ray for gravity
//        rayVel += gravDis * pow(stepSize, 2.4) * vec3((rayDefl.x) * (1.0 / pow(rayDist, 4.0)),
//            (rayDefl.y) * (1.0 / pow(rayDist, 4.0)),
//            (rayDefl.z) * (1.0 / pow(rayDist, 4.0)) );
//       
//       //Distance field calculations
//        dField = distField(rayPos, holePos);
//
//        //float travel = distance(rayPos, newCam) / 300.0;
//        stepSize = min(clamp(rayDist - (holeRadius + stepVa), 0.05, stepVa), max(boDist + distance(holePos.z, rayPos.z),0.2));
//        
//        rayVel = normalize(rayVel) * stepSize;
//        
//        //Volumetric rendering of the Accretion Disk
//        occ += dField.g;
//        rayVol = rayVol + (dField.r * vol * stepSize) / occ;//mix((((dField.r) * vol) * stepSize), 0.0, 0.0);//(occ) / 2.0);
//        
//        if(rayDist >= 2.0 * bounds)
//        {return vec3((starField(rayVel) / (occ)) + rayVol, clamp(rayVol * colShift, 0.0, 1.0), rayVol);}
//        
//        if(rayDist <= holeRadius)
//        {return vec3(rayVol, clamp(rayVol * colShift, 0.0, 1.0), rayVol);}
//        
//        //Color things, fakes a subtle blueshift, but does a horrible job at it.
//        colShift += rayVol * rayPos.x / (float(steps) * 100.0);
//    }
//    if(rayDist >= holeRadius * 3.0){rayVol += starField(rayVel) / (occ);}
//    return vec3(rayVol, clamp(rayVol * colShift, 0.0, 1.0), rayVol);
//}
//
//
//
//
//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//{
//    // Normalized pixel coordinates (from 0 to 1)
//    vec2 uv = fragCoord/iResolution.xy;
//    
//    
//    //noise texture to distort rays
//    vec2 pixAxis = texture(iChannel0, uv).rg;
//    vec3 col = rayCast(pixAxis);
//    
//   
//    // Output to screen
//    fragColor = vec4(col, 1.0);
//}
//
//
////BUFFER C
//// Kelvin to RGB algorithm thanks to https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
//vec3 tempConvert( float temp)
//{
//    vec3 color;
//    float newtemp = pow((temp / (1.0)), 3.0) * 255.0;
//    if(newtemp <= 66.0)
//        {
//            color.r = 255.0;
//            color.g = 99.4708025861 * log(newtemp) - 161.1195681661;
//            if(newtemp <= 19.0) {color.b = 0.0;}
//                else
//                {
//                    color.b = newtemp - 10.0;
//                    color.b = 138.5177312231 * log(color.b) - 305.0447927307;
//                }
//        }
//        else
//        {
//            color.r = newtemp - 60.0;
//            color.r = 329.698727446 * pow(color.r, -0.13321);
//            color.g = newtemp - 60.0;
//            color.g = 288.1221695283 * pow(color.g, -0.075515);
//            color.b = 255.0;
//        }
//   
//    return clamp(color / 610.0, 0.0, 1.0);
//}
//
//
////ACES tonemapping
//vec3 ACESFilm(vec3 x)
//{
//    float a = 2.51f;
//    float b = 0.03f;
//    float c = 2.43f;
//    float d = 0.59f;
//    float e = 0.14f;
//    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
//}
//
//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//{
//    // Normalized pixel coordinates (from 0 to 1)
//    vec2 uv = fragCoord/iResolution.xy;
//    vec3 col = texture(iChannel0, uv).rgb;
//    vec3 temp = tempConvert(col.r);
//    vec3 shift = mix(vec3(1.0, 0.3, 0.1), vec3(0.55, 0.7, 1.0), vec3(col.g));
//    //shift = vec3(col.g);
//    
//    col = vec3(pow(col.r, 2.0) * (temp / (temp + 1.0)));
//    col *= shift;
//    col = pow(mix(col, ACESFilm(col), 1.0), vec3(1.0 / 2.2));
//     
//   
//    // Output to screen
//    fragColor = vec4(col,1.0);
//}
//
//
////IMAGE
//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//{
//    // Normalized pixel coordinates (from 0 to 1)
//    vec2 uv = fragCoord/iResolution.xy;
//    vec3 col = texture(iChannel0, uv).rgb;
//    
//    
//    
//    
//    // Output to screen
//    fragColor = vec4(col,1.0);
//}
