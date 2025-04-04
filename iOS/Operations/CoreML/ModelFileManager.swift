// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// Manages ML model file operations and ensures it's available for the app
final class ModelFileManager {
    static let shared = ModelFileManager()
    
    private init() {}
    
    // Model file name and extension
    private let modelFileName = "coreml_model"
    private let modelExtension = "mlmodel"
    
    /// Copy the model from project directory to Documents directory
    func prepareMLModel(completion: @escaping (Result<URL, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                // Check if model already exists in Documents
                if let existingModelURL = self.getModelURLInDocuments() {
                    DispatchQueue.main.async {
                        Debug.shared.log(message: "ML model already exists in Documents directory", type: .info)
                        completion(.success(existingModelURL))
                    }
                    return
                }
                
                // Find model in project directory
                guard let sourceModelURL = self.findModelInProjectDirectory() else {
                    throw ModelError.modelNotFound
                }
                
                // Create destination URL in Documents directory
                let docsURL = try FileManager.default.url(
                    for: .documentDirectory,
                    in: .userDomainMask,
                    appropriateFor: nil,
                    create: true
                )
                let modelsDir = docsURL.appendingPathComponent("Models", isDirectory: true)
                
                // Create Models directory if it doesn't exist
                try FileManager.default.createDirectory(
                    at: modelsDir,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
                
                let destinationURL = modelsDir.appendingPathComponent("\(modelFileName).\(modelExtension)")
                
                // Copy the file
                try FileManager.default.copyItem(at: sourceModelURL, to: destinationURL)
                
                DispatchQueue.main.async {
                    Debug.shared.log(message: "ML model successfully copied to Documents directory", type: .info)
                    completion(.success(destinationURL))
                }
                
            } catch {
                DispatchQueue.main.async {
                    Debug.shared.log(message: "Failed to prepare ML model: \(error.localizedDescription)", type: .error)
                    completion(.failure(error))
                }
            }
        }
    }
    
    /// Get URL to model in Documents directory if it exists
    func getModelURLInDocuments() -> URL? {
        do {
            let docsURL = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            )
            let modelsDir = docsURL.appendingPathComponent("Models", isDirectory: true)
            let modelURL = modelsDir.appendingPathComponent("\(modelFileName).\(modelExtension)")
            
            if FileManager.default.fileExists(atPath: modelURL.path) {
                return modelURL
            }
            return nil
        } catch {
            Debug.shared.log(message: "Error checking Documents directory for model: \(error.localizedDescription)", type: .error)
            return nil
        }
    }
    
    /// Find the model file in various possible locations in the project
    private func findModelInProjectDirectory() -> URL? {
        // Possible locations to check
        let possibleLocations = [
            // Root model directory
            URL(fileURLWithPath: "./model/\(modelFileName).\(modelExtension)"),
            
            // Absolute path from project root
            URL(fileURLWithPath: "/workspace/im-a-test-bdg_Backdoor-v1/model/\(modelFileName).\(modelExtension)"),
            
            // Try to find relative to bundle path
            Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("model").appendingPathComponent("\(modelFileName).\(modelExtension)")
        ]
        
        // Check each location
        for url in possibleLocations {
            if FileManager.default.fileExists(atPath: url.path) {
                Debug.shared.log(message: "Found ML model at: \(url.path)", type: .info)
                return url
            }
        }
        
        Debug.shared.log(message: "ML model not found in any expected location", type: .error)
        return nil
    }
}

/// Errors that can occur during model file operations
enum ModelError: Error, LocalizedError {
    case modelNotFound
    case copyFailed
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "CoreML model file not found in expected locations"
        case .copyFailed:
            return "Failed to copy CoreML model to Documents directory"
        }
    }
}
