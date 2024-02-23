//
//  ContentView.swift
//  Space Time
//
//  Created by User1 on 10/25/23.
//

import SwiftUI
import RealityKit
import RealityKitContent


import AVFoundation

// Music player manager
class MusicPlayerManager {
    static let shared = MusicPlayerManager() // Singleton instance
    var player: AVAudioPlayer?

    func startPlaying() {
        guard let url = Bundle.main.url(forResource: "SoftMusic", withExtension: "mp3") else { return }
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.numberOfLoops = -1 // Loop indefinitely
            player?.play()
        } catch {
            print("Couldn't load the music file.")
        }
    }

    func stopPlaying() {
        player?.stop()
    }
}

//struct ContentView: View {
//
//    @State private var showImmersiveSpace = false
//    @State private var immersiveSpaceIsShown = false
//    @State private var isMusicPlaying = false
//    @State private var selectedDisplayMode = "All planets" // Default selection
//
//    @Environment(\.openImmersiveSpace) var openImmersiveSpace
//    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
//
//    var body: some View {
//        VStack {
//            Model3D(named: "Scene", bundle: realityKitContentBundle)
//                .padding(.bottom, 50)
//
//            Text("Hello, from Space!")
//
//            Toggle(isOn: $showImmersiveSpace) {
//                Text(immersiveSpaceIsShown ? "Leave Immersive" : "Enter Cosmos")
//            }
//            .toggleStyle(.button)
//            .padding(.top, 50)
//
//            // Music toggle
//            Toggle("Play Soft Music", isOn: $isMusicPlaying)
//                .onChange(of: isMusicPlaying) { newValue in
//                    if newValue {
//                        MusicPlayerManager.shared.startPlaying()
//                    } else {
//                        MusicPlayerManager.shared.stopPlaying()
//                    }
//                }
//                .padding(.top, 20)
//        }
//        .padding()
//        .onChange(of: showImmersiveSpace) { _, newValue in
//            Task {
//                if newValue {
//                    switch await openImmersiveSpace(id: "ImmersiveSpace") {
//                    case .opened:
//                        immersiveSpaceIsShown = true
//                    case .error, .userCancelled:
//                        fallthrough
//                    @unknown default:
//                        immersiveSpaceIsShown = false
//                        showImmersiveSpace = false
//                    }
//                } else if immersiveSpaceIsShown {
//                    await dismissImmersiveSpace()
//                    immersiveSpaceIsShown = false
//                }
//            }
//        }
//    }
//}

struct ContentView: View {
    @State private var showImmersiveSpace = false
    @State private var immersiveSpaceIsShown = false
    @State private var isMusicPlaying = false
    //@ObservedObject var renderer: Renderer // Assume it's passed in or initialized here
    @EnvironmentObject var renderer: Renderer

    @State private var selectedDisplayMode: DisplayMode = .all // Default selection

    let displayModes = ["Planetary focus", "Earth focus"]

    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    var body: some View {
        VStack {
            Model3D(named: "Scene", bundle: realityKitContentBundle)
                .padding(.bottom, 50)

            Text("Welcome! ")

            Toggle(isOn: $showImmersiveSpace) {
                Text(immersiveSpaceIsShown ? "Exit" : "Explore the cosmos")
            }
            .toggleStyle(.button)
            .padding(.top, 50)

            // Music toggle
            Toggle("Play Soft Music", isOn: $isMusicPlaying)
                .onChange(of: isMusicPlaying) {
                    if isMusicPlaying {
                        MusicPlayerManager.shared.startPlaying()
                    } else {
                        MusicPlayerManager.shared.stopPlaying()
                    }
                }
                .padding(.top, 20)
            
            // Display Mode Picker
            Picker("Select Display Mode", selection: $selectedDisplayMode) {
                ForEach(DisplayMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented) // You can choose another style if you prefer
            .padding()

        }
        .padding()
        .onChange(of: showImmersiveSpace) { _, newValue in
            Task {
                if newValue {
                    switch await openImmersiveSpace(id: "ImmersiveSpace") {
                    case .opened:
                        immersiveSpaceIsShown = true
                    case .error, .userCancelled:
                        fallthrough
                    @unknown default:
                        immersiveSpaceIsShown = false
                        showImmersiveSpace = false
                    }
                } else if immersiveSpaceIsShown {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
        }
        .onChange(of: selectedDisplayMode) {
            RendererManager.shared.updateDisplayMode(selectedDisplayMode)
        }
    }
}
