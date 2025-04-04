// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML

/// Client for interacting with the Backdoor AI learning server
class BackdoorAIClient {
    // Singleton instance
    static let shared = BackdoorAIClient()
    
    // Server configuration
    private let baseURL: URL
    
    // We no longer need an API key for server authentication
    // Previously encrypted key and passphrase have been removed
    
    // Server endpoints
    private let learnEndpoint = "api/ai/learn"
    private let latestModelEndpoint = "api/ai/latest-model"
    private let modelDownloadEndpoint = "api/ai/models"
    
    // Fixed learn endpoint - direct path provided by server admin
    private let fixedLearnEndpointURL = URL(string: "https://database-iupv.onrender.com/api/ai/learn")!
    
    // User defaults keys
    private let currentModelVersionKey = "currentModelVersion"
    
    /// Initialize the client with server URL
    private init() {
        // Always use the fixed endpoint to ensure reliability
        let serverURL = "https://database-iupv.onrender.com"
        self.baseURL = URL(string: serverURL)!
        
        Debug.shared.log(message: "BackdoorAIClient initialized", type: .info)
    }
    
    // No configuration update functionality - using secure hardcoded values
    
    // Common headers for all requests
    // API key authentication has been removed as it's no longer needed
    private var headers: [String: String] {
        return [
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Backdoor-App/\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")"
        ]
    }
    
    // MARK: - Data Upload
    
    /// Upload interaction data to the server
    func uploadInteractions(interactions: [AIInteraction], behaviors: [UserBehavior] = [], patterns: [AppUsagePattern] = []) async throws -> ModelInfo {
        // Use the fixed learn endpoint URL provided by the server admin
        let url = fixedLearnEndpointURL
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Add headers
        headers.forEach { request.addValue($0.value, forHTTPHeaderField: $0.key) }
        
        // Convert our internal models to API models
        let apiInteractions = interactions.map { interaction -> Interaction in
            let feedback = interaction.feedback.map { 
                Feedback(rating: $0.rating, comment: $0.comment)
            }
            
            // Format timestamp to match server expectation: "2023-06-15T14:30:00Z"
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            
            return Interaction(
                id: interaction.id,
                timestamp: formatter.string(from: interaction.timestamp),
                userMessage: interaction.userMessage,
                aiResponse: interaction.aiResponse,
                detectedIntent: interaction.detectedIntent,
                confidenceScore: interaction.confidenceScore,
                feedback: feedback
            )
        }
        
        // Convert behaviors to API models
        let apiBehaviors = behaviors.map { behavior -> AppBehavior in
            return AppBehavior(
                id: behavior.id,
                timestamp: ISO8601DateFormatter().string(from: behavior.timestamp),
                action: behavior.action,
                screen: behavior.screen,
                duration: behavior.duration,
                details: behavior.details
            )
        }
        
        // Convert patterns to API models
        let apiPatterns = patterns.map { pattern -> UsagePattern in
            return UsagePattern(
                id: pattern.id,
                timestamp: ISO8601DateFormatter().string(from: pattern.timestamp),
                feature: pattern.feature,
                timeSpent: pattern.timeSpent,
                actionSequence: pattern.actionSequence,
                completedTask: pattern.completedTask
            )
        }
        
        // Create device data package (using properties that don't require await)
        let deviceId = await UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
        let deviceData = DeviceData(
            deviceId: deviceId,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0",
            modelVersion: UserDefaults.standard.string(forKey: currentModelVersionKey) ?? "1.0.0",
            osVersion: "iOS \(await UIDevice.current.systemVersion)",
            interactions: apiInteractions,
            behaviors: apiBehaviors,
            patterns: apiPatterns
        )
        
        // Encode data
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted]
            
            let jsonData = try encoder.encode(deviceData)
            
            // Log the JSON for debugging
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                Debug.shared.log(message: "Request JSON: \(jsonString)", type: .debug)
            }
            
            request.httpBody = jsonData
        } catch {
            Debug.shared.log(message: "Failed to encode device data: \(error)", type: .error)
            throw APIError.encodingFailed
        }
        
        // Make request
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                Debug.shared.log(message: "Server returned error status: \((response as? HTTPURLResponse)?.statusCode ?? 0)", type: .error)
                throw APIError.invalidResponse
            }
            
            // Decode response
            do {
                let modelInfo = try JSONDecoder().decode(ModelInfo.self, from: data)
                Debug.shared.log(message: "Successfully uploaded \(interactions.count) interactions, \(behaviors.count) behaviors, and \(patterns.count) patterns", type: .info)
                return modelInfo
            } catch {
                Debug.shared.log(message: "Failed to decode model info: \(error)", type: .error)
                throw APIError.decodingFailed
            }
        } catch {
            Debug.shared.log(message: "Network error during upload: \(error)", type: .error)
            throw APIError.networkError(error)
        }
    }
    
    // MARK: - Model Management
    
    /// Get information about the latest available model
    func getLatestModelInfo() async throws -> ModelInfo {
        Debug.shared.log(message: "Fetching latest model info with direct networking", type: .info)
        
        // Construct the URL for latest model info
        let latestModelURL = baseURL.appendingPathComponent(latestModelEndpoint)
        
        // Create URL request
        var request = URLRequest(url: latestModelURL)
        request.httpMethod = "GET"
        
        // Add headers
        headers.forEach { request.addValue($0.value, forHTTPHeaderField: $0.key) }
        
        do {
            // Perform the request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                Debug.shared.log(message: "Invalid response format", type: .error)
                throw APIError.invalidResponse
            }
            
            // Check status code
            guard (200...299).contains(httpResponse.statusCode) else {
                Debug.shared.log(message: "Server returned error status: \(httpResponse.statusCode)", type: .error)
                
                if httpResponse.statusCode == 404 {
                    throw APIError.modelNotFound
                } else if httpResponse.statusCode >= 500 {
                    throw APIError.invalidResponse
                } else {
                    throw APIError.invalidResponse
                }
            }
            
            // Decode the response
            do {
                let modelInfo = try JSONDecoder().decode(ModelInfo.self, from: data)
                return modelInfo
            } catch {
                Debug.shared.log(message: "Failed to decode model info: \(error)", type: .error)
                throw APIError.decodingFailed
            }
        } catch {
            if error is APIError {
                throw error // Rethrow our custom APIErrors
            } else {
                Debug.shared.log(message: "Network error during model info request: \(error)", type: .error)
                throw APIError.networkError(error)
            }
        }
    }
    
    /// Download a specific model version from the server
    func downloadModel(version: String) async throws -> URL {
        Debug.shared.log(message: "Downloading model version \(version) with direct networking", type: .info)
        
        // Create temporary file to store the model
        let tempDir = FileManager.default.temporaryDirectory
        let modelURL = tempDir.appendingPathComponent("model_\(version).mlmodel")
        
        // Remove existing file if it exists
        if FileManager.default.fileExists(atPath: modelURL.path) {
            try FileManager.default.removeItem(at: modelURL)
        }
        
        // Construct download URL
        let downloadURL = baseURL.appendingPathComponent(modelDownloadEndpoint).appendingPathComponent(version)
        var request = URLRequest(url: downloadURL)
        
        // Add headers
        headers.forEach { request.addValue($0.value, forHTTPHeaderField: $0.key) }
        
        do {
            // Show download progress with UI notification
            let notificationName = Notification.Name("ModelDownloadProgress")
            NotificationCenter.default.post(name: notificationName, object: nil, userInfo: ["status": "started", "version": version])
            
            // Download file with URLSession
            let (tempFileURL, response) = try await URLSession.shared.download(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                Debug.shared.log(message: "Invalid response format during download", type: .error)
                throw APIError.invalidResponse
            }
            
            // Check status code
            guard (200...299).contains(httpResponse.statusCode) else {
                Debug.shared.log(message: "Server returned error status during download: \(httpResponse.statusCode)", type: .error)
                
                if httpResponse.statusCode == 404 {
                    throw APIError.modelNotFound
                } else if httpResponse.statusCode >= 500 {
                    throw APIError.invalidResponse
                } else {
                    throw APIError.downloadFailed
                }
            }
            
            // Move downloaded file to the target location
            try FileManager.default.moveItem(at: tempFileURL, to: modelURL)
            
            // Verify the downloaded file integrity with CRC32 checksum
            let fileData = try Data(contentsOf: modelURL)
            let checksum = CryptoHelper.shared.crc32(of: fileData)
            
            Debug.shared.log(message: "Model downloaded successfully with checksum: \(checksum)", type: .info)
            NotificationCenter.default.post(name: notificationName, object: nil, userInfo: ["status": "completed", "version": version])
            
            return modelURL
        } catch {
            // Enhanced error handling
            Debug.shared.log(message: "Failed to download model: \(error)", type: .error)
            
            // Notify UI of failure
            NotificationCenter.default.post(
                name: Notification.Name("ModelDownloadProgress"),
                object: nil,
                userInfo: ["status": "failed", "version": version, "error": error.localizedDescription]
            )
            
            if error is APIError {
                throw error // Rethrow our custom APIErrors
            } else {
                throw APIError.downloadFailed
            }
        }
    }
    
    /// Compile and save model to the app's documents directory
    func compileAndSaveModel(at tempURL: URL) async throws -> URL {
        // Get documents directory for persistent storage
        let documentsDir = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        
        // Get models directory
        let modelsDir = documentsDir.appendingPathComponent("AIModels", isDirectory: true)
        if !FileManager.default.fileExists(atPath: modelsDir.path) {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true, attributes: nil)
        }
        
        // Define destination
        let modelFileName = tempURL.lastPathComponent
        let compiledModelName = modelFileName.replacingOccurrences(of: ".mlmodel", with: ".mlmodelc")
        let compiledModelURL = modelsDir.appendingPathComponent(compiledModelName)
        
        // Check if model already exists
        if FileManager.default.fileExists(atPath: compiledModelURL.path) {
            Debug.shared.log(message: "Model already compiled at \(compiledModelURL.path)", type: .info)
            return compiledModelURL
        }
        
        // Compile model (this is CPU intensive - do on background thread)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    Debug.shared.log(message: "Compiling model at \(tempURL.path)", type: .info)
                    let compiledURL = try MLModel.compileModel(at: tempURL)
                    
                    // Save to documents directory
                    if FileManager.default.fileExists(atPath: compiledModelURL.path) {
                        try FileManager.default.removeItem(at: compiledModelURL)
                    }
                    try FileManager.default.copyItem(at: compiledURL, to: compiledModelURL)
                    
                    Debug.shared.log(message: "Model successfully compiled and saved to \(compiledModelURL.path)", type: .info)
                    continuation.resume(returning: compiledModelURL)
                } catch {
                    Debug.shared.log(message: "Failed to compile model: \(error)", type: .error)
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Check for model updates, download and update if needed
    func checkAndUpdateModel() async -> Bool {
        do {
            let modelInfo = try await getLatestModelInfo()
            let currentVersion = UserDefaults.standard.string(forKey: currentModelVersionKey) ?? "1.0.0"
            
            // If we have a newer version available
            if modelInfo.latestModelVersion != currentVersion {
                Debug.shared.log(message: "New model version available: \(modelInfo.latestModelVersion) (current: \(currentVersion))", type: .info)
                
                // Download and update
                let tempModelURL = try await downloadModel(version: modelInfo.latestModelVersion)
                let compiledModelURL = try await compileAndSaveModel(at: tempModelURL)
                
                // Update current version
                UserDefaults.standard.set(modelInfo.latestModelVersion, forKey: currentModelVersionKey)
                
                // Post notification for other components
                DispatchQueue.main.async {
                    NotificationCenter.default.post(name: Notification.Name("AIModelUpdated"), object: nil)
                }
                
                return true
            } else {
                Debug.shared.log(message: "Model is already up to date (version \(currentVersion))", type: .info)
                return false
            }
        } catch {
            Debug.shared.log(message: "Error checking for model updates: \(error)", type: .error)
            return false
        }
    }
    
    /// Get the URL to the latest model
    func getLatestModelURL() -> URL? {
        // Get the current model version
        let version = UserDefaults.standard.string(forKey: currentModelVersionKey) ?? "1.0.0"
        
        // Get documents directory
        guard let documentsDir = try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) else {
            return nil
        }
        
        // Get the compiled model directory
        let modelsDir = documentsDir.appendingPathComponent("AIModels", isDirectory: true)
        let modelName = "model_\(version).mlmodelc"
        let modelURL = modelsDir.appendingPathComponent(modelName)
        
        // Check if the model exists
        if FileManager.default.fileExists(atPath: modelURL.path) {
            return modelURL
        }
        
        return nil
    }
    
    /// Get the URL to the latest model asynchronously with proper async/await APIs
    func getLatestModelURLAsync() async -> URL? {
        // First try the synchronous version for speed
        if let localModelURL = getLatestModelURL() {
            return localModelURL
        }
        
        // If no local model, check if we need to download one
        do {
            let modelInfo = try await getLatestModelInfo()
            let currentVersion = UserDefaults.standard.string(forKey: currentModelVersionKey) ?? "1.0.0"
            
            // Check if server has a newer version
            if modelInfo.latestModelVersion != currentVersion {
                Debug.shared.log(message: "Downloading newer model version \(modelInfo.latestModelVersion)", type: .info)
                
                // Use an elegant UI loader animation to show progress
                DispatchQueue.main.async {
                    NotificationCenter.default.post(
                        name: Notification.Name("ShowModelLoadingAnimation"),
                        object: nil
                    )
                }
                
                // Download and compile the model
                let tempModelURL = try await downloadModel(version: modelInfo.latestModelVersion)
                let compiledModelURL = try await compileAndSaveModel(at: tempModelURL)
                
                // Update current version
                UserDefaults.standard.set(modelInfo.latestModelVersion, forKey: currentModelVersionKey)
                
                // Remove loading animation
                DispatchQueue.main.async {
                    NotificationCenter.default.post(
                        name: Notification.Name("HideModelLoadingAnimation"),
                        object: nil
                    )
                }
                
                return compiledModelURL
            }
        } catch {
            Debug.shared.log(message: "Failed to get latest model async: \(error)", type: .error)
            // Fall through to return nil
        }
        
        return nil
    }
}

// MARK: - Data Structures for API Interaction

extension BackdoorAIClient {
    /// User feedback on an AI interaction
    struct Feedback: Codable {
        let rating: Int
        let comment: String?
    }
    
    /// A single user interaction with the AI
    struct Interaction: Codable {
        let id: String
        let timestamp: String // ISO8601 formatted date
        let userMessage: String
        let aiResponse: String
        let detectedIntent: String
        let confidenceScore: Double
        let feedback: Feedback?
    }
    
    /// A single user behavior within the app
    struct AppBehavior: Codable {
        let id: String
        let timestamp: String // ISO8601 formatted date
        let action: String
        let screen: String
        let duration: TimeInterval
        let details: [String: String]
    }
    
    /// A pattern of app usage
    struct UsagePattern: Codable {
        let id: String
        let timestamp: String // ISO8601 formatted date
        let feature: String
        let timeSpent: TimeInterval
        let actionSequence: [String]
        let completedTask: Bool
    }
    
    /// Complete data package to send to the server
    struct DeviceData: Codable {
        let deviceId: String
        let appVersion: String
        let modelVersion: String
        let osVersion: String
        let interactions: [Interaction]
        let behaviors: [AppBehavior]?
        let patterns: [UsagePattern]?
        
        init(deviceId: String, appVersion: String, modelVersion: String, osVersion: String, 
             interactions: [Interaction], behaviors: [AppBehavior], patterns: [UsagePattern]) {
            self.deviceId = deviceId
            self.appVersion = appVersion
            self.modelVersion = modelVersion
            self.osVersion = osVersion
            self.interactions = interactions
            
            // Only include non-empty arrays for better compatibility
            self.behaviors = behaviors.isEmpty ? nil : behaviors
            self.patterns = patterns.isEmpty ? nil : patterns
        }
    }
    
    /// Response from the server containing model information
    /// Matches the example response:
    /// {
    ///   "success": true,
    ///   "message": "Data received successfully",
    ///   "latestModelVersion": "1.0.1712052481",
    ///   "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
    /// }
    struct ModelInfo: Codable {
        let success: Bool
        let message: String
        let latestModelVersion: String
        let modelDownloadURL: String?
    }
    
    /// Errors that can occur during API operations
    enum APIError: Error, LocalizedError {
        case invalidResponse
        case modelNotFound
        case encodingFailed
        case decodingFailed
        case downloadFailed
        case networkError(Error)
        
        var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "The server returned an invalid response"
            case .modelNotFound:
                return "The requested model was not found"
            case .encodingFailed:
                return "Failed to encode data for upload"
            case .decodingFailed:
                return "Failed to decode server response"
            case .downloadFailed:
                return "Failed to download model"
            case .networkError(let error):
                return "Network error: \(error.localizedDescription)"
            }
        }
    }
}
