// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML

/// Extension to AILearningManager for model upload functionality
extension AILearningManager {
    
    /// Upload the locally trained CoreML model to the server for ensemble training
    func uploadTrainedModelToServer() async -> (success: Bool, message: String) {
        Debug.shared.log(message: "Starting trained model upload process", type: .info)
        
        // Verify server sync is enabled
        guard isServerSyncEnabled else {
            Debug.shared.log(message: "Server sync is disabled. Cannot upload model.", type: .warning)
            return (false, "Server sync is disabled in settings. Please enable it to upload models.")
        }
        
        // Get the latest model URL - use async version for proper await
        guard let modelURL = await getLatestModelURLAsync() else {
            Debug.shared.log(message: "No trained model found to upload", type: .error)
            return (false, "No trained model found. Please train a model first.")
        }
        
        // Upload the model
        do {
            let message = try await uploadModelToServer(modelURL: modelURL)
            Debug.shared.log(message: "Model upload successful: \(message)", type: .info)
            return (true, message)
        } catch {
            Debug.shared.log(message: "Model upload failed: \(error.localizedDescription)", type: .error)
            return (false, "Upload failed: \(error.localizedDescription)")
        }
    }
    
    /// Check if a trained model exists and is ready for upload
    func isTrainedModelAvailableForUpload() -> Bool {
        if let modelURL = getLatestModelURL() {
            return FileManager.default.fileExists(atPath: modelURL.path)
        }
        return false
    }
    
    /// Get information about the trained model
    func getTrainedModelInfo() -> (version: String, date: Date?) {
        let version = currentModelVersion
        let date = UserDefaults.standard.object(forKey: lastTrainingKey) as? Date
        return (version, date)
    }
}
