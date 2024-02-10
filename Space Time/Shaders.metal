//
//  Shaders.metal
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>


// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

#define AA 2
#define AB 2


struct Vertex{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
};


struct ColorInOut{
    float4 position [[position]];
    float2 texCoord;
};



float4 returnSPH1(){
    float4 sph1 = float4(0.0,0.0,0.0,1.0);
    return sph1;
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

float3 fancyCube(texture2d<float> texture, sampler sam,  float3 d,  float s,  float b )
{
    //calculate the sampling coordinates based on direction 'd'
    float2 coordX = 0.5 + s* d.yz/d.x;
    float2 coordY = 0.5 + s* d.zx/d.y;
    float2 coordZ = 0.5 + s* d.xy/d.z;
    
    
    //sample texture at calculated coordinates
    float3 colx = texture.sample( sam, coordX, level(b) ).xyz;
    float3 coly = texture.sample( sam, coordY, level(b) ).xyz;
    float3 colz = texture.sample( sam, coordZ, level(b) ).xyz;
    
    //calculate weighted color components
    float3 n = d*d;
    float3 resultColor = (colx * n.x + coly * n.y + colz * n.z)/ (n.x +n.y + n.z);
    
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

float3 background(  float3 d,  float3 l , 
                  texture2d<float> iChannel1,
                  sampler sam)
{
    float3 col = float3(0.0);
    
         col += 0.5*pow( fancyCube( iChannel1,sam,d, 0.05, 5.0 ).zyx, float3(2.0) );
         col += 0.2*pow( fancyCube( iChannel1,sam, d, 0.10, 3.0 ).zyx, float3(1.5) );
         col += 0.8*float3(0.80,0.5,0.6)*pow( fancyCube( iChannel1,sam, d, 0.1, 0.0 ).xxx, float3(6.0) );
    
    float stars = smoothstep( 0.3, 0.7, fancyCube( iChannel1,sam,d, 0.91, 0.0 ).x );

    float3 n = abs(d);
    n = n*n*n;
    
    float2 vxy = voronoi( 50.0*d.xy );
    float2 vyz = voronoi( 50.0*d.yz );
    float2 vzx = voronoi( 50.0*d.zx );
    float2 r = (vyz*n.x + vzx*n.y + vxy*n.z) / (n.x+n.y+n.z);
    col += 0.5 * stars * clamp(1.0-(3.0+r.y*5.0)*r.x,0.0,1.0);

    col = 1.5*col - 0.2;
    col += float3(-0.05,0.1,0.0);

    float s = clamp( dot(d,l), 0.0, 1.0 );
    col += 0.4*pow(s,5.0)*float3(1.0,0.7,0.6)*2.0;
    col += 0.4*pow(s,64.0)*float3(1.0,0.9,0.8)*2.0;
    
    return col;
}

//--------------------------------------------------------------------


float rayTrace(  float3 ro,  float3 rd )
{
    return shpIntersect( ro, rd, returnSPH1() );
}

float map(  float3 pos )
{
    float2 r = pos.xz - returnSPH1().xz;
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
        float h = map( pos );
        if( h<0.001 || t>tmax ) break;
        t += h;
    }
    return t;
}

float3 render( float3 ro, float3 rd ,texture2d<float> iChannel0, texture2d<float> iChannel1, sampler sam,float iTime)
{
    float3 lig = normalize( float3(1.0,0.2,1.0) );
    float3 col = background( rd, lig, iChannel1, sam );
    
    // raytrace stuff
    float t = rayTrace( ro, rd );

    if( t>0.0 )
    {
        float3 mat = float3( 0.18 );
        float3 pos = ro + t*rd;
        float3 nor = sphNormal( pos, returnSPH1() );
            
        float am = 0.1*iTime;
        float2 pr = float2( cos(am), sin(am) );
        float3 tnor = nor;
        tnor.xz = float2x2( float2(pr.x, -pr.y), float2(pr.y, pr.x )) * tnor.xz;


        float am2 = 0.08*iTime - 1.0*(1.0-nor.y*nor.y);
        pr = float2( cos(am2), sin(am2) );
        float3 tnor2 = nor;
        tnor2.xz = float2x2( float2(pr.x, -pr.y), float2(pr.y, pr.x )) * tnor2.xz;

        float3 ref = reflect( rd, nor );
        float fre = clamp( 1.0+dot( nor, rd ), 0.0 ,1.0 );

        float l = fancyCube( iChannel0,sam,tnor, 0.03, 0.0 ).x;
        l += -0.1 + 0.3*fancyCube( iChannel0,sam, tnor, 8.0, 0.0 ).x;

        float3 sea  = mix( float3(0.0,0.07,0.2), float3(0.0,0.01,0.3), fre );
        sea *= 0.15;

        float3 land = float3(0.02,0.04,0.0);
        land = mix( land, float3(0.05,0.1,0.0), smoothstep(0.4,1.0,fancyCube( iChannel0,sam, tnor, 0.1, 0.0 ).x ));
        land *= fancyCube( iChannel0,sam, tnor, 0.3, 0.0 ).xyz;
        land *= 0.5;

        float los = smoothstep(0.45,0.46, l);
        mat = mix( sea, land, los );

        float3 wrap = -1.0 + 2.0*fancyCube( iChannel1,sam, tnor2.xzy, 0.025, 0.0 ).xyz;
        float cc1 = fancyCube( iChannel1,sam, tnor2 + 0.2*wrap, 0.05, 0.0 ).y;
        float clouds = smoothstep( 0.3, 0.6, cc1 );

        mat = mix( mat, float3(0.93*0.15), clouds );

        float dif = clamp( dot(nor, lig), 0.0, 1.0 );
        mat *= 0.8;
        float3 lin  = float3(3.0,2.5,2.0)*dif;
        lin += 0.01;
        col = mat * lin;
        col = pow( col, float3(0.4545) );
        col += 0.6*fre*fre*float3(0.9,0.9,1.0)*(0.3+0.7*dif);

        float spe = clamp( dot(ref,lig), 0.0, 1.0 );
        float tspe = pow( spe, 3.0 ) + 0.5*pow( spe, 16.0 );
        col += (1.0-0.5*los)*clamp(1.0-2.0*clouds,0.0,1.0)*0.3*float3(0.5,0.4,0.3)*tspe*dif;;
    }
    
    // raymarch stuff
    float tmax = 20.0;
    if( t>0.0 ) tmax = t;
    t = rayMarch( ro, rd, tmax );
    if( t<tmax )
    {
        float3 pos = ro + t*rd;

        float2 scp = sin(2.0*6.2831*pos.xz);

        float3 wir = float3( 0.0 );
        wir += 1.0*exp(-12.0*abs(scp.x));
        wir += 1.0*exp(-12.0*abs(scp.y));
        wir += 0.5*exp( -4.0*abs(scp.x));
        wir += 0.5*exp( -4.0*abs(scp.y));
        wir *= 0.2 + 1.0*sphSoftShadow( pos, lig, returnSPH1(), 4.0 );

        col += wir*0.5*exp( -0.05*t*t );
    }

    // outter glow
    if( dot(rd,returnSPH1().xyz-ro)>0.0 )
    {
        float d = sphDistance( ro, rd, returnSPH1() );
        float3 glo = float3(0.0);
        glo += float3(0.6,0.7,1.0)*0.3*exp(-2.0*abs(d))*step(0.0,d);
        glo += 0.6*float3(0.6,0.7,1.0)*0.3*exp(-8.0*abs(d));
        glo += 0.6*float3(0.8,0.9,1.0)*0.4*exp(-100.0*abs(d));
        col += glo*1.5;
    }
    
    col *= smoothstep( 0.0, 6.0, iTime );

    return col;
}

float3x3 setCamera(  float3 ro,  float3 rt,  float cr )
{
    float3 cw = normalize(rt-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv = normalize( cross(cu,cw) );
    return float3x3( cu, cv, -cw );
}





struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texCoords [[attribute(2)]];};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
    float3 modelNormal;//??
    float3 RayOri;
    float3 RayDir;
};


//struct Uniforms {
//    float4x4 modelViewMatrix;
//    float4x4 projectionMatrix;
//    float time;
//};

struct PoseConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float3 cameraPosition;
};

struct InstanceConstants {
    float4x4 modelMatrix;
};

vertex VertexOut vertexShader(const VertexIn in [[stage_in]],
                              constant Uniforms &uniforms [[buffer(0)]]){
                             //constant PoseConstants &pose [[buffer(0)]],
                             //constant InstanceConstants &environment [[buffer(1)]],
                             //constant float &time [[buffer(2)]]) {
    VertexOut out;
    // Transform vertex positions to clip space
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * float4(in.position, 1.0f);
    out.RayOri = float3(0.0);//pose.cameraPosition;//shouldn;t this be inverse of view matrix?
    out.RayDir = normalize(out.RayOri - float3(0.0, 0.0, 0.0)); // Example direction
    return out;
}


fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(0)]],
                               
                              texture2d<float> iChannel0 [[texture(2)]],
                              texture2d<float> iChannel1 [[texture(1)]]) {
    
    float4 finalColor = float4(0,0,0,1);
    float zo = 1.0 + smoothstep( 5.0, 15.0, abs(uniforms.time-48.0) );
    float an = 3.0 + 0.05 * uniforms.time;
    float3 ro = zo*float3( 2.0 * cos(an), 1.0, 2.0*sin(an) );
    float3 rt = float3( 1.0, 0.0, 0.0 );
    float3x3 cam = setCamera( ro, rt, 0.35 );
    sampler sam;
    finalColor.xyz = render(in.RayOri, in.RayDir, iChannel0, iChannel1, sam, uniforms.time);
    //return finalColor;
    return float4(1.0,0,0,1);
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
