// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML
import UIKit

/// Extension to AppContextManager for CoreML setup and integration
extension AppContextManager {
    
    /// Initialize CoreML for app-wide use
    func setupCoreML() {
        Debug.shared.log(message: "Setting up CoreML integration", type: .info)
        
        // Start model loading in background to avoid blocking UI
        DispatchQueue.global(qos: .userInitiated).async {
            self.prepareAndLoadMLModel()
        }
        
        // Register the model-related commands
        registerCoreMLCommands()
    }
    
    /// Prepare and load the ML model
    private func prepareAndLoadMLModel() {
        // Ensure model file is available
        ModelFileManager.shared.prepareMLModel { result in
            switch result {
            case .success(let modelURL):
                Debug.shared.log(message: "CoreML model prepared successfully at: \(modelURL.path)", type: .info)
                
                // Preload the model to avoid delay during first use
                CoreMLManager.shared.loadModel { success in
                    if success {
                        Debug.shared.log(message: "CoreML model preloaded successfully", type: .info)
                        
                        // Update AI services with ML availability
                        NotificationCenter.default.post(
                            name: Notification.Name("CoreMLModelLoaded"),
                            object: nil
                        )
                    } else {
                        Debug.shared.log(message: "Failed to preload CoreML model", type: .warning)
                    }
                }
                
            case .failure(let error):
                Debug.shared.log(message: "Failed to prepare CoreML model: \(error.localizedDescription)", type: .error)
            }
        }
    }
    
    /// Register commands related to CoreML and AI features
    private func registerCoreMLCommands() {
        // Command to analyze text with ML model
        registerCommand("analyze text") { text, completion in
            CoreMLManager.shared.predictIntent(from: text) { result in
                switch result {
                case .success(let prediction):
                    var responseText = "Analysis results:\n"
                    responseText += "Intent: \(prediction.intent)\n"
                    responseText += "Confidence: \(Int(prediction.confidence * 100))%\n"
                    
                    if !prediction.parameters.isEmpty {
                        responseText += "\nDetected parameters:\n"
                        for (key, value) in prediction.parameters {
                            responseText += "- \(key): \(value)\n"
                        }
                    }
                    
                    completion(responseText)
                    
                case .failure(let error):
                    completion("Failed to analyze text: \(error.localizedDescription)")
                }
            }
        }
        
        // Command to get sentiment analysis
        registerCommand("sentiment") { text, completion in
            CoreMLManager.shared.analyzeSentiment(from: text) { result in
                switch result {
                case .success(let sentiment):
                    let sentimentText: String
                    switch sentiment.sentiment {
                    case .positive:
                        sentimentText = "positive 😊"
                    case .negative:
                        sentimentText = "negative 😞"
                    case .neutral:
                        sentimentText = "neutral 😐"
                    }
                    
                    completion("The sentiment of the text is \(sentimentText) (score: \(Int(sentiment.score * 100))%)")
                    
                case .failure(let error):
                    completion("Failed to analyze sentiment: \(error.localizedDescription)")
                }
            }
        }
        
        // Command to check ML model status
        registerCommand("ml status") { _, completion in
            let isLoaded = CoreMLManager.shared.isModelLoaded
            let status = isLoaded ? "loaded and active" : "not loaded"
            completion("CoreML model is \(status)")
        }
    }
}
