//
//  Renderer.swift
//

import CompositorServices
import Metal
import MetalKit
import simd
import Spatial

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100
let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

enum DisplayMode: String, CaseIterable, Identifiable {

    case all = "Planetary focus"
    case earth = "Earth focus"
    
    var id: String { self.rawValue }

}

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

class Renderer: ObservableObject  {

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    //var displayMode: DisplayMode = .all
    @Published var displayMode: DisplayMode = .earth

    var pipelineState: MTLRenderPipelineState!
    var pipelineStateAll: MTLRenderPipelineState!
    var pipelineStateEarth: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState
    var globe: MTLTexture
    var stars: MTLTexture
    var startTime: TimeInterval = CACurrentMediaTime()
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>
    var rotation: Float = 0
    var mesh: MTKMesh
    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    let layerRenderer: LayerRenderer
    
    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!
        self.pipelineState = nil
        self.pipelineStateAll = nil
        self.pipelineStateEarth = nil
        //self.pipelineState =
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        self.dynamicUniformBuffer = self.device.makeBuffer(length:uniformBufferSize,
                                                           options:[MTLResourceOptions.storageModeShared])!

        self.dynamicUniformBuffer.label = "UniformBuffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:UniformsArray.self, capacity:1)

        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptor.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor:depthStateDescriptor)!
        

        do {
            mesh = try Renderer.buildFullScreenQuad(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
           // mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            fatalError("Unable to build MetalKit Mesh. Error info: \(error)")
        }
        do {
            globe = try Renderer.loadTexturePNG(device: device, textureName: "globe")
        } catch {
            fatalError("Unable to load globe. Error info: \(error)")
        }
        
        do {
            stars = try Renderer.loadTextureJPG(device: device, textureName: "stars")
        } catch {
            fatalError("Unable to load stars. Error info: \(error)")
        }
        
        
        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
        
        do {
            initializePipelineStates()
//            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
//                                                                       layerRenderer: layerRenderer,
//                                                                       mtlVertexDescriptor: mtlVertexDescriptor,
//                                                                       vertexFunctionName: "String",
//                                                                       fragmentFunctionName: "String")
        } //catch {
         //   fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        //}
    }
    
    func startRenderLoop() {
        Task {
            do {
                try await arSession.run([worldTracking])
            } catch {
                fatalError("Failed to initialize ARSession")
            }
            
            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }
    // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render pipeline and how we'll layout our Model IO vertices
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {

        
        let mtlVertexDescriptor = MTLVertexDescriptor()

           // Setup for vertex positions
           mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
           mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
           mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

           // Define the layout for the position data in the buffer
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12;//MemoryLayout<SIMD3<Float>>.stride // Or simply 12
           //mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
           //mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

           return mtlVertexDescriptor
    }

        class func buildRenderPipelineWithDevice(device: MTLDevice,
                                                 layerRenderer: LayerRenderer,
                                                 mtlVertexDescriptor: MTLVertexDescriptor,
                                                 vertexFunctionName: String,
                                                 fragmentFunctionName: String) throws -> MTLRenderPipelineState {

        // Build a render state pipeline object
        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: vertexFunctionName)
        let fragmentFunction = library?.makeFunction(name: fragmentFunctionName)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat
        pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        let metalAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(4, 4, 4),
                                     segments: SIMD3<UInt32>(2, 2, 2),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        //attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate

        mdlMesh.vertexDescriptor = mdlVertexDescriptor

        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    
    class func buildFullScreenQuad(device: MTLDevice, mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        // Allocator for mesh buffers
        let metalAllocator = MTKMeshBufferAllocator(device: device)

        // Define the quad vertices (two triangles) to cover the entire screen in NDC
        let quadVertices: [Float] = [
            // First triangle
            -1.0,  1.0, 0.0, // top left
            -1.0, -1.0, 0.0, // bottom left
             1.0, -1.0, 0.0, // bottom right
            // Second triangle
            -1.0,  1.0, 0.0, // top left
             1.0, -1.0, 0.0, // bottom right
             1.0,  1.0, 0.0  // top right
        ]
        
        // Indices for the vertices above
        let indices: [UInt16] = [
            0, 1, 2,  // First triangle
            3, 4, 5   // Second triangle
        ]
        
        
        // Convert vertices and indices arrays to Data
         let vertexData = Data(bytes: quadVertices, count: quadVertices.count * MemoryLayout<Float>.size)
         let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<UInt16>.size)


        // Create vertex and index buffers
        let vertexBuffer = metalAllocator.newBuffer(with: vertexData, type: .vertex)
        let indexBuffer = metalAllocator.newBuffer(with: indexData, type: .index)
        
        // Create the MDLMesh with the vertex buffer
        let submesh = MDLSubmesh(indexBuffer: indexBuffer,
                                 indexCount: indices.count,
                                 indexType: .uInt16,
                                 geometryType: .triangles,
                                 material: nil)
        

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        let mdlMesh = MDLMesh(vertexBuffers: [vertexBuffer],
                              vertexCount: quadVertices.count / 3,
                              descriptor: (mdlVertexDescriptor),
                              submeshes: [submesh])
        
        
        return try MTKMesh(mesh: mdlMesh, device: device)
    }

    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling

        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]

        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)

    }

    class func loadTexturePNG(device: MTLDevice, textureName: String) throws -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: device)

        // Ensure the texture is in the app's main bundle
        guard let url = Bundle.main.url(forResource: textureName, withExtension: "png") else {
            throw NSError(domain: "TextureLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to locate \(textureName).png in the main bundle."])
        }

        // Options for the texture
        let textureLoaderOptions: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue),
            // Include any other options relevant to your specific texture here
        ]

        // Load the texture from the URL
        do {
            let texture = try textureLoader.newTexture(URL: url, options: textureLoaderOptions)
            return texture
        } catch {
            // Handle any errors by rethrowing
            throw error
        }
    }
    
    
    
    
    class func loadTextureJPG(device: MTLDevice, textureName: String) throws -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: device)

        // Ensure the texture is in the app's main bundle
        guard let url = Bundle.main.url(forResource: textureName, withExtension: "jpeg") else {
            throw NSError(domain: "TextureLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to locate \(textureName).jpeg in the main bundle."])
        }

        // Options for the texture
        let textureLoaderOptions: [MTKTextureLoader.Option: Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue),
            // You can include other options here as needed
        ]

        // Load the texture from the URL
        do {
            let texture = try textureLoader.newTexture(URL: url, options: textureLoaderOptions)
            return texture
        } catch {
            // If there's an error in loading the texture, it's propagated
            throw error
        }
    }
    
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:UniformsArray.self, capacity:1)
    }

    private func updateGameState(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) {
        /// Update any game state before rendering

        let currentTime = CACurrentMediaTime()
        let elapsedTime = Float(currentTime - self.startTime)
        
        //let rotationAxis = SIMD3<Float>(1, 1, 0)

       // if let drawable = layerRenderer.queryNextFrame() {
        let viewportWidth = drawable.colorTextures.first?.width ?? 0
        let viewportHeight = drawable.colorTextures.first?.height ?? 0
            // Now you have a viewport size you can pass to your shaders
       // }
        
        //let modelRotationMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        //let modelTranslationMatrix = matrix4x4_translation(0.0, 0.0, -1.0)
        //let modelMatrix = modelTranslationMatrix * modelRotationMatrix
        
        // Use identity matrices for simplification
        let modelMatrix = matrix_identity_float4x4
        //let viewMatrix = matrix_identity_float4x4

        
        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4
        
        func uniforms(forViewIndex viewIndex: Int) -> Uniforms {
            let view = drawable.views[viewIndex]
            let viewMatrix = (simdDeviceAnchor * view.transform).inverse
            let projection = ProjectiveTransform3D(leftTangent: Double(view.tangents[0]),
                                                   rightTangent: Double(view.tangents[1]),
                                                   topTangent: Double(view.tangents[2]),
                                                   bottomTangent: Double(view.tangents[3]),
                                                   nearZ: Double(drawable.depthRange.y),
                                                   farZ: Double(drawable.depthRange.x),
                                                   reverseZ: true)
            
            
            // Extract camera position from the translation components of the matrix
            let translation = viewMatrix.inverse.columns.3
            let cameraPosition = SIMD3<Float>(translation.x, translation.y, translation.z)
//            * modelMatrix
            return Uniforms(projectionMatrix: .init(projection), 
                            modelMatrix: modelMatrix ,
                            viewMatrix: viewMatrix ,
                            cameraPosition: cameraPosition,
                            time: elapsedTime,
                            viewportWidth: Int32(viewportWidth),
                            viewportHeight: Int32(viewportHeight))
        }
        
            self.uniforms[0].uniforms.0 = uniforms(forViewIndex: 0)
        if drawable.views.count > 1 {
            self.uniforms[0].uniforms.1 = uniforms(forViewIndex: 1)
        }
        
        //rotation += 0.01
    }

    
    func initializePipelineStates() {
        do {
            pipelineStateAll = try Renderer.buildRenderPipelineWithDevice(
                device: device,
                layerRenderer: layerRenderer,
                mtlVertexDescriptor: Renderer.buildMetalVertexDescriptor(),
                vertexFunctionName: "vertexShaderAll", // Update these names to match your actual shader function names
                fragmentFunctionName: "fragmentShaderAll"
            )
            pipelineStateEarth = try Renderer.buildRenderPipelineWithDevice(
                device: device,
                layerRenderer: layerRenderer,
                mtlVertexDescriptor: Renderer.buildMetalVertexDescriptor(),
                vertexFunctionName: "vertexShaderEarth", // Update these names accordingly
                fragmentFunctionName: "fragmentShaderEarth"
            )
        } catch {
            fatalError("Unable to compile render pipeline state. Error info: \(error)")
        }
    }

    func renderFrame() {
        /// Per frame updates hare

        guard let frame = layerRenderer.queryNextFrame() else { return }
        
        frame.startUpdate()
        
        // Perform frame independent work
        
        frame.endUpdate()
        
        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)
        
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }
        
        guard let drawable = frame.queryDrawable() else { return }
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        frame.startSubmission()
        
        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)
        
        drawable.deviceAnchor = deviceAnchor
        
        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }
        
        self.updateDynamicBufferState()
        
        self.updateGameState(drawable: drawable, deviceAnchor: deviceAnchor)
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
        if layerRenderer.configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = drawable.views.count
        }
        
        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
    
        renderEncoder.label = "Primary Render Encoder"
        
        renderEncoder.pushDebugGroup("Draw Box")
        
        //renderEncoder.setCullMode(.back)
        renderEncoder.setCullMode(.none)

        
        renderEncoder.setFrontFacing(.counterClockwise)
        
        //renderEncoder.setRenderPipelineState(pipelineState)
        // Determine and safely unwrap the correct pipeline state based on displayMode
        if let pipelineState = (displayMode == .all) ? pipelineStateAll : pipelineStateEarth {
            renderEncoder.setRenderPipelineState(pipelineState)
        } else {
            // Handle the case where the pipeline state could not be unwrapped (e.g., because it's nil)
            fatalError("Pipeline state is nil")
        }
        
        renderEncoder.setDepthStencilState(depthState)
        
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: 1)

        let viewports = drawable.views.map { $0.textureMap.viewport }
        
        renderEncoder.setViewports(viewports)
        
        if drawable.views.count > 1 {
            var viewMappings = (0..<drawable.views.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        
//        for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
//            guard let layout = element as? MDLVertexBufferLayout else {
//                return
//            }
//            
//            if layout.stride != 0 {
//                let buffer = mesh.vertexBuffers[index]
//                renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
//            }
//        }
        
        renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: 0)
        renderEncoder.setFragmentTexture(stars, index: 1)//
        renderEncoder.setFragmentTexture(globe, index: 2)//


        if let vertexBuffer = mesh.vertexBuffers.first?.buffer {
            // Set the vertex buffer
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            
            // Draw the quad directly without using indices
            //renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
        
        
        for submesh in mesh.submeshes {
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                indexCount: submesh.indexCount,
                                                indexType: submesh.indexType,
                                                indexBuffer: submesh.indexBuffer.buffer,
                                                indexBufferOffset: submesh.indexBuffer.offset)

        }
        
        renderEncoder.popDebugGroup()
        
        renderEncoder.endEncoding()
        
        drawable.encodePresent(commandBuffer: commandBuffer)
        
        commandBuffer.commit()
        
        frame.endSubmission()
    }
    
    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                return
            } else if layerRenderer.state == .paused {
                layerRenderer.waitUntilRunning()
                continue
            } else {
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}

// Helper function to create an orthographic projection matrix
func matrix_orthographic_projection(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> matrix_float4x4 {
    let ral = right + left
    let rsl = right - left
    let tab = top + bottom
    let tsb = top - bottom
    let fan = far + near
    let fsn = far - near
    
    return matrix_float4x4(columns: (
        vector_float4(2.0 / rsl, 0, 0, 0),
        vector_float4(0, 2.0 / tsb, 0, 0),
        vector_float4(0, 0, -2.0 / fsn, 0),
        vector_float4(-ral / rsl, -tab / tsb, -fan / fsn, 1)
    ))
}

//enum RendererError: Error {
//    case badVertexDescriptor
//    case textureLoadingFailed(resourceName: String)
//    case pipelineStateCreationFailed
//    case commandQueueCreationFailed
//    case bufferAllocationFailed
//}
//
//class Renderer {
//    var device: MTLDevice
//    var commandQueue: MTLCommandQueue
//    var dynamicUniformBuffer: MTLBuffer
//    var displayMode: DisplayMode = .all
//    var pipelineStateAll: MTLRenderPipelineState?
//    var pipelineStateEarth: MTLRenderPipelineState?
//    var depthState: MTLDepthStencilState?
//    var mesh: MTKMesh
//    var globe: MTLTexture?
//    var stars: MTLTexture?
//    let inFlightSemaphore: DispatchSemaphore
//    var uniformBufferOffset = 0
//    var uniformBufferIndex = 0
//    var uniforms: UnsafeMutablePointer<UniformsArray>?
//    var rotation: Float = 0
//    let maxBuffersInFlight = 3
//
//    init(device: MTLDevice) throws {
//        self.device = device
//        guard let commandQueue = device.makeCommandQueue() else {
//            throw RendererError.commandQueueCreationFailed
//        }
//        self.commandQueue = commandQueue
//        self.inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
//        self.mesh = try Renderer.buildFullScreenQuad(device: device)
//
//        try self.initializeBuffers()
//        try self.initializeDepthState()
//        try self.loadAssets()
//        try self.initializePipelineStates()
//    }
//
//    private func initializeBuffers() throws {
//        let alignedUniformsSize = (MemoryLayout<UniformsArray>.size + 0xFF) & ~0xFF
//        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
//        guard let buffer = device.makeBuffer(length: uniformBufferSize, options: [.storageModeShared]) else {
//            throw RendererError.bufferAllocationFailed
//        }
//        self.dynamicUniformBuffer = buffer
//        self.uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to: UniformsArray.self, capacity: 1)
//    }
//
//    private func initializeDepthState() throws {
//        let depthStateDescriptor = MTLDepthStencilDescriptor()
//        depthStateDescriptor.depthCompareFunction = .greater
//        depthStateDescriptor.isDepthWriteEnabled = true
//        guard let state = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
//            throw RendererError.pipelineStateCreationFailed
//        }
//        self.depthState = state
//    }
//
//    private func loadAssets() throws {
//        self.globe = try Renderer.loadTexture(device: device, textureName: "globe", extension: "png")
//        self.stars = try Renderer.loadTexture(device: device, textureName: "stars", extension: "jpg")
//    }
//
//    private func initializePipelineStates() throws {
//        self.pipelineStateAll = try buildRenderPipelineState(vertexFunctionName: "vertexShaderAll", fragmentFunctionName: "fragmentShaderAll")
//        self.pipelineStateEarth = try buildRenderPipelineState(vertexFunctionName: "vertexShaderEarth", fragmentFunctionName: "fragmentShaderEarth")
//    }
//
//    private func buildRenderPipelineState(vertexFunctionName: String, fragmentFunctionName: String) throws -> MTLRenderPipelineState {
//        guard let library = device.makeDefaultLibrary() else {
//            throw RendererError.pipelineStateCreationFailed
//        }
//        guard let vertexFunction = library.makeFunction(name: vertexFunctionName),
//              let fragmentFunction = library.makeFunction(name: fragmentFunctionName) else {
//            throw RendererError.pipelineStateCreationFailed
//        }
//        let descriptor = MTLRenderPipelineDescriptor()
//        descriptor.vertexFunction = vertexFunction
//        descriptor.fragmentFunction = fragmentFunction
//        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
//        descriptor.depthAttachmentPixelFormat = .depth32Float
//        return try device.makeRenderPipelineState(descriptor: descriptor)
//    }
//
//    class func loadTexture(device: MTLDevice, textureName: String, extension ext: String) throws -> MTLTexture {
//        let textureLoader = MTKTextureLoader(device: device)
//        guard let url = Bundle.main.url(forResource: textureName, withExtension: ext) else {
//            throw RendererError.textureLoadingFailed(resourceName: textureName)
//        }
//        let options: [MTKTextureLoader.Option: Any] = [
//            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
//            .textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
//        ]
//        return try textureLoader.newTexture(URL: url, options: options)
//    }
//    
//    func updateDynamicBufferState() {
//        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
//        uniformBufferOffset = ((MemoryLayout<UniformsArray>.size + 0xFF) & ~0xFF) * uniformBufferIndex
//        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to: UniformsArray.self, capacity: 1)
//    }
//
//    
//
//    class func buildFullScreenQuad(device: MTLDevice) throws -> MTKMesh {
//        // Implementation of full screen quad creation
//    }
//
//    // Additional methods like rendering, updating, etc.
//}
