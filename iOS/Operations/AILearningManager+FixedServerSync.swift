// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML
import UIKit

/// Extension to AILearningManager specifically for model upload to the server
extension AILearningManager {
    
    /// URL for the model upload endpoint
    private var modelUploadEndpoint: URL {
        return URL(string: "https://database-iupv.onrender.com/api/ai/upload-model")!
    }
    
    /// Upload a CoreML model to the server with proper multipart/form-data format
    func uploadModelToServer(modelURL: URL) async throws -> String {
        Debug.shared.log(message: "Starting model upload to server endpoint", type: .info)
        
        // Create a multipart request
        var request = URLRequest(url: modelUploadEndpoint)
        request.httpMethod = "POST"
        
        // API key authentication removed as it's no longer needed
        
        // Generate boundary for multipart request
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        // Get device ID
        let deviceId = await UIDevice.current.identifierForVendor?.uuidString ?? "unknown-device"
        
        // Get app version
        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        
        // Create model description with timestamp
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let modelDescription = "Device trained model from \(deviceId) on \(timestamp)"
        
        // Create multipart form data
        var data = Data()
        
        // Add device ID
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"deviceId\"\r\n\r\n".data(using: .utf8)!)
        data.append("\(deviceId)\r\n".data(using: .utf8)!)
        
        // Add app version
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"appVersion\"\r\n\r\n".data(using: .utf8)!)
        data.append("\(appVersion)\r\n".data(using: .utf8)!)
        
        // Add description
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"description\"\r\n\r\n".data(using: .utf8)!)
        data.append("\(modelDescription)\r\n".data(using: .utf8)!)
        
        // Add model file
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"model\"; filename=\"\(modelURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        
        // Read model file
        let modelData = try Data(contentsOf: modelURL)
        data.append(modelData)
        data.append("\r\n".data(using: .utf8)!)
        
        // End the multipart form
        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        // Make the request
        let (responseData, response) = try await URLSession.shared.upload(for: request, from: data)
        
        // Check response
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            Debug.shared.log(message: "Server returned error status code: \(statusCode)", type: .error)
            throw NSError(domain: "HTTP", code: statusCode, userInfo: [NSLocalizedDescriptionKey: "Server returned error status code: \(statusCode)"])
        }
        
        // Parse response
        guard let json = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let success = json["success"] as? Bool,
              let message = json["message"] as? String else {
            throw NSError(domain: "Response", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
        }
        
        // Check success
        if success {
            Debug.shared.log(message: "Model upload successful: \(message)", type: .info)
            return message
        } else {
            Debug.shared.log(message: "Model upload failed: \(message)", type: .error)
            throw NSError(domain: "Server", code: 0, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}
