import SwiftUI
import MLX
import Hub
import MLXNN
import MLXRandom
import Tokenizers
import FluxSwift
import AppKit

struct ContentView: View {
    @State private var prompt: String = "A cat sitting on a tree"
    @FocusState private var isPromptFocused: Bool
    @State private var width: Int = 512
    @State private var height: Int = 512
    @State private var steps: Int = 4
    @State private var guidance: Float = 3.5
    @State private var outputImage: MLXImage?
    @State private var isGenerating: Bool = false
    @State private var progressValue: Float = 0.0

    var body: some View {
        VStack {
            TextField("Enter prompt", text: $prompt)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
                .focused($isPromptFocused)

            HStack {
                VStack {
                    Text("Width: \(width)")
                    Slider(value: Binding(
                        get: { Double(width) },
                        set: { width = Int($0) }
                    ), in: 256...1024, step: 64)
                }
                VStack {
                    Text("Height: \(height)")
                    Slider(value: Binding(
                        get: { Double(height) },
                        set: { height = Int($0) }
                    ), in: 256...1024, step: 64)
                }
            }.padding()

            HStack {
                VStack {
                    Text("Steps: \(steps)")
                    Slider(value: Binding(
                        get: { Double(steps) },
                        set: { steps = Int($0) }
                    ), in: 1...50, step: 1)
                }
                VStack {
                    Text("Guidance: \(String(format: "%.1f", guidance))")
                    Slider(value: $guidance, in: 1...10, step: 0.1)
                }
            }.padding()

            Button(action: generateImage) {
                Text("Generate Image")
            }
            .disabled(isGenerating)
            .padding()

            if isGenerating {
                ProgressView(value: progressValue)
                    .progressViewStyle(LinearProgressViewStyle())
                    .padding()
            }
            if let image = outputImage {
                Image(image.asCGImage(), scale: 1.0, label: Text(""))
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 300, maxHeight: 300)
            }
        }
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.isPromptFocused = true
                self.forceActivateWindow()
            }
        }
    }

    func generateImage() {
        isGenerating = true
        progressValue = 0.0

        Task {
            do {
                // Download model
                try await FluxConfiguration.flux1Schnell.download { progress in
                    DispatchQueue.main.async {
                        self.progressValue = Float(progress.fractionCompleted)
                    }
                }

                // Rest of the image generation code
                let loadConfiguration = LoadConfiguration(float16: true, quantize: false)
                let generator = try FluxConfiguration.flux1Schnell.textToImageGenerator(
                    configuration: loadConfiguration)
                generator?.ensureLoaded()
                
                var parameters = FluxConfiguration.flux1Schnell.defaultParameters()
                parameters.height = height
                parameters.width = width
                parameters.prompt = prompt
                parameters.numInferenceSteps = steps

                var denoiser = generator?.generateLatents(parameters: parameters)
                var lastXt: MLXArray!
                while let xt = denoiser!.next() {
                    progressValue = Float(denoiser!.i) / Float(parameters.numInferenceSteps)
                    eval(xt)
                    lastXt = xt
                }

                let unpackedLatents = unpackLatents(lastXt, height: parameters.height, width: parameters.width)
                let decoded = generator?.decode(xt: unpackedLatents)
                var imageData = decoded?.squeezed()
                imageData = imageData!.transposed(1, 2, 0)
                
                let raster = (imageData! * 255).asType(.uint8)
                
                DispatchQueue.main.async {
                    self.outputImage = MLXImage(raster)
                    self.isGenerating = false
                }
            } catch {
                print("Error generating image: \(error)")
                DispatchQueue.main.async {
                    self.isGenerating = false
                }
            }
        }
    }

    func unpackLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let reshaped = latents.reshaped(1, height / 16, width / 16, 16, 2, 2)
        let transposed = reshaped.transposed(0, 3, 1, 4, 2, 5)
        return transposed.reshaped(1, 16, height / 16 * 2, width / 16 * 2)
    }

    private func forceActivateWindow() {
        NSApplication.shared.activate(ignoringOtherApps: true)
        if let window = NSApplication.shared.windows.first {
            window.makeKeyAndOrderFront(nil)
            window.becomeKey()
        }
    }
}
