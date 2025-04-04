// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit
import CoreML
import CreateML

/// Manager for on-device AI learning and improvement
class AILearningManager {
    // Singleton instance
    static let shared = AILearningManager()
    
    // Local storage for interactions
    internal var storedInteractions: [AIInteraction] = []
    internal var userBehaviors: [UserBehavior] = []
    internal var appUsagePatterns: [AppUsagePattern] = []
    
    // Lock for thread-safe access
    internal let interactionsLock = NSLock()
    internal let behaviorsLock = NSLock()
    internal let patternsLock = NSLock()
    
    // Settings keys
    private let learningEnabledKey = "AILearningEnabled"
    internal let lastTrainingKey = "AILastTrainingDate"
    internal let modelVersionKey = "AILocalModelVersion"
    private let exportPasswordKey = "ExportPasswordHash"
    
    // Export password
    private let correctPasswordHash = "2B4D5G".sha256()
    
    // Model paths
    private let interactionsPath: URL
    private let behaviorsPath: URL
    private let patternsPath: URL
    internal let modelsDirectory: URL  // Changed to internal for extension access
    private let exportsDirectory: URL
    
    // Training configuration
    private let minInteractionsForTraining = 10
    private let minDaysBetweenTraining = 1
    
    // Current model version
    private(set) var currentModelVersion: String = "1.0.0"
    
    private init() {
        // Set up storage locations
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        interactionsPath = documentsDirectory.appendingPathComponent("ai_interactions.json")
        behaviorsPath = documentsDirectory.appendingPathComponent("user_behaviors.json")
        patternsPath = documentsDirectory.appendingPathComponent("app_usage.json")
        modelsDirectory = documentsDirectory.appendingPathComponent("AIModels", isDirectory: true)
        exportsDirectory = documentsDirectory.appendingPathComponent("AIExports", isDirectory: true)
        
        // Create directories if needed
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: exportsDirectory, withIntermediateDirectories: true)
        
        // Load stored data
        loadInteractions()
        loadBehaviors()
        loadPatterns()
        
        // Get current model version
        if let savedVersion = UserDefaults.standard.string(forKey: modelVersionKey) {
            currentModelVersion = savedVersion
        }
        
        // Schedule periodic model training evaluation
        scheduleTrainingEvaluation()
    }
    
    // MARK: - Public Interface
    
    /// Check if AI learning is enabled
    var isLearningEnabled: Bool {
        return UserDefaults.standard.bool(forKey: learningEnabledKey)
    }
    
    /// Set whether AI learning is enabled
    func setLearningEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: learningEnabledKey)
        Debug.shared.log(message: "AI learning \(enabled ? "enabled" : "disabled")", type: .info)
    }
    
    /// Check if server sync is enabled (sending data to remote server)
    var isServerSyncEnabled: Bool {
        // Default to enabled for better out-of-box experience
        if UserDefaults.standard.object(forKey: "AIServerSyncEnabled") == nil {
            UserDefaults.standard.set(true, forKey: "AIServerSyncEnabled")
            return true
        }
        return UserDefaults.standard.bool(forKey: "AIServerSyncEnabled")
    }
    
    /// Set whether server sync is enabled
    func setServerSyncEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: "AIServerSyncEnabled")
        Debug.shared.log(message: "AI server sync \(enabled ? "enabled" : "disabled")", type: .info)
    }
    
    /// Verify export password
    func verifyExportPassword(_ password: String) -> Bool {
        return password.sha256() == correctPasswordHash
    }
    
    /// Get the URL for the latest trained model (locally trained or server-provided)
    func getLatestModelURL() -> URL? {
        // We'll only use the locally trained model in the synchronous version for safety
        // Server model is only checked in the async version
        
        // Fall back to locally trained model
        let modelPath = modelsDirectory.appendingPathComponent("model_\(currentModelVersion).mlmodel")
        
        // Check if file exists
        if FileManager.default.fileExists(atPath: modelPath.path) {
            return modelPath
        }
        
        return nil
    }
    
    /// Async version that properly awaits backend calls
    func getLatestModelURLAsync() async -> URL? {
        // First check for server-provided model - with proper await
        if isServerSyncEnabled, let serverModelURL = await BackdoorAIClient.shared.getLatestModelURLAsync() {
            return serverModelURL
        }
        
        // Fall back to locally trained model
        let modelPath = modelsDirectory.appendingPathComponent("model_\(currentModelVersion).mlmodel")
        
        // Check if file exists
        if FileManager.default.fileExists(atPath: modelPath.path) {
            return modelPath
        }
        
        return nil
    }
    
    /// Export the latest trained model
    func exportModel(password: String) -> Result<URL, ExportError> {
        // Verify password
        guard verifyExportPassword(password) else {
            return .failure(.invalidPassword)
        }
        
        // Get the latest model URL
        guard let modelURL = getLatestModelURL() else {
            return .failure(.modelNotFound)
        }
        
        do {
            // Create a date formatter for file naming
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
            let dateString = dateFormatter.string(from: Date())
            
            // Create export filename
            let exportFileName = "backdoor_ai_model_\(dateString).mlmodel"
            let exportURL = exportsDirectory.appendingPathComponent(exportFileName)
            
            // Copy the model file
            try FileManager.default.copyItem(at: modelURL, to: exportURL)
            
            // Log the export
            Debug.shared.log(message: "Successfully exported AI model to \(exportURL.path)", type: .info)
            
            return .success(exportURL)
        } catch {
            Debug.shared.log(message: "Failed to export model: \(error)", type: .error)
            return .failure(.exportFailed(error))
        }
    }
    
    /// Record a user interaction with the AI for learning purposes
    func recordInteraction(userMessage: String, aiResponse: String, intent: String, confidence: Double) {
        // Skip if learning is disabled
        guard isLearningEnabled else {
            return
        }
        
        // Create interaction record
        let interaction = AIInteraction(
            id: UUID().uuidString,
            timestamp: Date(),
            userMessage: userMessage,
            aiResponse: aiResponse,
            detectedIntent: intent,
            confidenceScore: confidence,
            feedback: nil,
            context: getCurrentContext(),
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown",
            modelVersion: currentModelVersion
        )
        
        // Add to stored interactions
        interactionsLock.lock()
        storedInteractions.append(interaction)
        interactionsLock.unlock()
        
        // Save to disk
        saveInteractions()
        
        // If server sync is enabled, queue for sync
        if isServerSyncEnabled {
            queueForServerSync()
        } else {
            // Otherwise, check if we should train locally
            evaluateTraining()
        }
    }
    
    /// Record user behavior within the app
    func recordUserBehavior(action: String, screen: String, duration: TimeInterval, details: [String: String] = [:]) {
        // Skip if learning is disabled
        guard isLearningEnabled else {
            return
        }
        
        // Create behavior record
        let behavior = UserBehavior(
            id: UUID().uuidString,
            timestamp: Date(),
            action: action,
            screen: screen,
            duration: duration,
            details: details
        )
        
        // Add to stored behaviors
        behaviorsLock.lock()
        userBehaviors.append(behavior)
        behaviorsLock.unlock()
        
        // Save to disk
        saveBehaviors()
    }
    
    /// Record app usage pattern
    func recordAppUsage(feature: String, timeSpent: TimeInterval, sequence: [String], completed: Bool) {
        // Skip if learning is disabled
        guard isLearningEnabled else {
            return
        }
        
        // Create usage pattern record
        let pattern = AppUsagePattern(
            id: UUID().uuidString,
            timestamp: Date(),
            feature: feature,
            timeSpent: timeSpent,
            actionSequence: sequence,
            completedTask: completed
        )
        
        // Add to stored patterns
        patternsLock.lock()
        appUsagePatterns.append(pattern)
        patternsLock.unlock()
        
        // Save to disk
        savePatterns()
    }
    
    /// Add feedback to a specific interaction
    func recordFeedback(for interactionId: String, rating: Int, comment: String? = nil) {
        interactionsLock.lock()
        
        // Find the interaction
        if let index = storedInteractions.firstIndex(where: { $0.id == interactionId }) {
            // Add feedback
            storedInteractions[index].feedback = AIFeedback(rating: rating, comment: comment)
            
            // Save
            saveInteractions()
            
            // If server sync is enabled, queue for sync
            if isServerSyncEnabled {
                queueForServerSync()
            } else {
                // Otherwise, consider training if this is highly-rated feedback
                if rating >= 4 {
                    DispatchQueue.global(qos: .background).async { [weak self] in
                        self?.evaluateTraining()
                    }
                }
            }
        }
        
        interactionsLock.unlock()
    }
    
    /// Get statistics about stored interactions and learning data
    func getLearningStatistics() -> LearningStatistics {
        interactionsLock.lock()
        behaviorsLock.lock()
        patternsLock.lock()
        defer { 
            interactionsLock.unlock()
            behaviorsLock.unlock()
            patternsLock.unlock()
        }
        
        let total = storedInteractions.count
        let withFeedback = storedInteractions.filter { $0.feedback != nil }.count
        let averageRating = calculateAverageRating()
        let lastTrainingDate = UserDefaults.standard.object(forKey: lastTrainingKey) as? Date
        let behaviorCount = userBehaviors.count
        let patternCount = appUsagePatterns.count
        
        // Calculate total data points
        let totalDataPoints = total + behaviorCount + patternCount
        
        return LearningStatistics(
            totalInteractions: total,
            interactionsWithFeedback: withFeedback,
            averageFeedbackRating: averageRating,
            behaviorCount: behaviorCount,
            patternCount: patternCount,
            totalDataPoints: totalDataPoints,
            modelVersion: currentModelVersion,
            lastTrainingDate: lastTrainingDate
        )
    }
    
    /// Manually trigger model training
    func trainModelNow(completion: @escaping (Bool, String) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    completion(false, "Manager deallocated")
                }
                return
            }
            
            // Check if we have enough data
            if self.storedInteractions.count < self.minInteractionsForTraining {
                DispatchQueue.main.async {
                    completion(false, "Not enough interactions for training (need at least \(self.minInteractionsForTraining))")
                }
                return
            }
            
            // Perform training
            let result = self.trainNewModel()
            
            DispatchQueue.main.async {
                if result.success {
                    completion(true, "Successfully trained model version \(result.version)")
                } else {
                    completion(false, "Training failed: \(result.errorMessage ?? "Unknown error")")
                }
            }
        }
    }
    
    /// Clear all stored interactions
    func clearAllInteractions() {
        interactionsLock.lock()
        storedInteractions.removeAll()
        interactionsLock.unlock()
        
        saveInteractions()
        
        Debug.shared.log(message: "Cleared all stored AI interactions", type: .info)
    }
    
    // MARK: - Private Methods
    
    /// Schedule periodic evaluation for training
    private func scheduleTrainingEvaluation() {
        // Check once per day if training should be performed
        let timer = Timer(timeInterval: 24 * 60 * 60, target: self, selector: #selector(timerFired), userInfo: nil, repeats: true)
        RunLoop.main.add(timer, forMode: .common)
    }
    
    @objc private func timerFired() {
        evaluateTraining()
    }
    
    /// Evaluate if a new model should be trained
    private func evaluateTraining() {
        // Only train if learning is enabled
        guard isLearningEnabled else {
            return
        }
        
        // Check if we have enough interactions
        interactionsLock.lock()
        let interactionCount = storedInteractions.count
        interactionsLock.unlock()
        
        guard interactionCount >= minInteractionsForTraining else {
            return
        }
        
        // Check when we last trained
        let lastTraining = UserDefaults.standard.object(forKey: lastTrainingKey) as? Date ?? Date.distantPast
        let daysSinceLastTraining = Calendar.current.dateComponents([.day], from: lastTraining, to: Date()).day ?? Int.max
        
        guard daysSinceLastTraining >= minDaysBetweenTraining else {
            return
        }
        
        // We meet all criteria, start training
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self = self else { return }
            let result = self.trainNewModel()
            if !result.success {
                Debug.shared.log(message: "Training failed: \(result.errorMessage ?? "Unknown error")", type: .error)
            }
        }
    }
    
    /// Train a new model using all collected data
    private func trainNewModel() -> (success: Bool, version: String, errorMessage: String?) {
        Debug.shared.log(message: "Starting comprehensive AI model training", type: .info)
        
        do {
            // Lock and copy all data
            interactionsLock.lock()
            behaviorsLock.lock()
            patternsLock.lock()
            
            let interactionsToUse = storedInteractions
            let behaviorsToUse = userBehaviors
            let patternsToUse = appUsagePatterns
            
            interactionsLock.unlock()
            behaviorsLock.unlock()
            patternsLock.unlock()
            
            // Generate new version
            let timestamp = Int(Date().timeIntervalSince1970)
            let newVersion = "1.0.\(timestamp)"
            
            // Prepare training data
            // Focus on interactions with positive feedback for better quality
            let trainingData = interactionsToUse.filter { 
                if let feedback = $0.feedback {
                    return feedback.rating >= 3  // Only use moderate to positive examples
                }
                return false
            }
            
            // Handle case where we don't have enough quality examples
            if trainingData.count < 5 {
                Debug.shared.log(message: "Not enough quality training examples", type: .warning)
                return (false, newVersion, "Not enough quality examples (with good feedback)")
            }
            
            // Create MLDataTable from interactions
            var textInput: [String] = []
            var intentOutput: [String] = []
            var contextData: [[String: String]] = []
            
            // Add message data
            for interaction in trainingData {
                textInput.append(interaction.userMessage)
                intentOutput.append(interaction.detectedIntent)
                
                // Add context when available
                if let context = interaction.context {
                    contextData.append(context)
                } else {
                    contextData.append([:])
                }
            }
            
            // Add behavior data to enhance the model
            if !behaviorsToUse.isEmpty {
                Debug.shared.log(message: "Incorporating \(behaviorsToUse.count) behavior records into training", type: .info)
                
                // Convert behaviors to training features
                for behavior in behaviorsToUse {
                    // Create a composite feature from the behavior
                    let behaviorText = "User performed \(behavior.action) on \(behavior.screen) screen"
                    let behaviorIntent = getIntentFromBehavior(behavior)
                    
                    textInput.append(behaviorText)
                    intentOutput.append(behaviorIntent)
                    contextData.append(behavior.details)
                }
            }
            
            // Add usage patterns to enhance the model
            if !patternsToUse.isEmpty {
                Debug.shared.log(message: "Incorporating \(patternsToUse.count) usage patterns into training", type: .info)
                
                // Convert patterns to training features
                for pattern in patternsToUse where pattern.completedTask {
                    // Only use completed tasks as they represent successful flows
                    let patternText = "User worked with \(pattern.feature) feature"
                    let patternIntent = "use:\(pattern.feature)"
                    
                    textInput.append(patternText)
                    intentOutput.append(patternIntent)
                    contextData.append(["sequence": pattern.actionSequence.joined(separator: ","),
                                        "completed": String(pattern.completedTask)])
                }
            }
            
            // Create enhanced text classifier with context features
            let modelURL = modelsDirectory.appendingPathComponent("model_\(newVersion).mlmodel")
            
            // Create data table for CreateML - Using the appropriate constructor format
            let dataTable = try MLDataTable(
                dictionary: [
                    "text": textInput,
                    "label": intentOutput
                ]
            )
            
            // Train model with simplified approach
            let textClassifier = try MLTextClassifier(
                trainingData: dataTable,
                textColumn: "text",
                labelColumn: "label"
            )
            
            // Save the model
            try textClassifier.write(to: modelURL)
            
            // Update current version
            currentModelVersion = newVersion
            UserDefaults.standard.set(newVersion, forKey: modelVersionKey)
            UserDefaults.standard.set(Date(), forKey: lastTrainingKey)
            
            Debug.shared.log(message: "Successfully trained new comprehensive model version: \(newVersion)", type: .info)
            
            // Notify that a new model is available
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: Notification.Name("AIModelUpdated"), object: nil)
            }
            
            return (true, newVersion, nil)
            
        } catch {
            Debug.shared.log(message: "Failed to train model: \(error)", type: .error)
            return (false, currentModelVersion, error.localizedDescription)
        }
    }
    
    /// Calculate average rating from feedbacks
    private func calculateAverageRating() -> Double {
        let feedbacks = storedInteractions.compactMap { $0.feedback }
        
        if feedbacks.isEmpty {
            return 0.0
        }
        
        let sum = feedbacks.reduce(0) { $0 + $1.rating }
        return Double(sum) / Double(feedbacks.count)
    }
    
    /// Save interactions to disk
    internal func saveInteractions() {
        interactionsLock.lock()
        defer { interactionsLock.unlock() }
        
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(storedInteractions)
            try data.write(to: interactionsPath)
        } catch {
            Debug.shared.log(message: "Failed to save interactions: \(error)", type: .error)
        }
    }
    
    /// Save user behaviors to disk
    internal func saveBehaviors() {
        behaviorsLock.lock()
        defer { behaviorsLock.unlock() }
        
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(userBehaviors)
            try data.write(to: behaviorsPath)
        } catch {
            Debug.shared.log(message: "Failed to save user behaviors: \(error)", type: .error)
        }
    }
    
    /// Save app usage patterns to disk
    internal func savePatterns() {
        patternsLock.lock()
        defer { patternsLock.unlock() }
        
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(appUsagePatterns)
            try data.write(to: patternsPath)
        } catch {
            Debug.shared.log(message: "Failed to save app usage patterns: \(error)", type: .error)
        }
    }
    
    /// Get the current app context
    private func getCurrentContext() -> [String: String]? {
        // Convert AppContext to a string dictionary
        let context = AppContextManager.shared.currentContext()
        
        // Extract relevant fields as strings
        var contextDict: [String: String] = [:]
        contextDict["screen"] = context.currentScreen
        
        // Get session ID from additional data if available
        if let sessionId = context.additionalData["currentChatSession"] as? String {
            contextDict["session"] = sessionId
        }
        
        // Only return if we have at least one valid value
        return contextDict.isEmpty ? nil : contextDict
    }
    
    /// Map user behavior to an intent
    internal func getIntentFromBehavior(_ behavior: UserBehavior) -> String {
        switch behavior.action {
        case "open":
            return "navigate:\(behavior.screen)"
        case "search":
            return "search:\(behavior.screen)"
        case "download":
            return "download"
        case "install":
            return "install"
        case "sign":
            return "sign"
        default:
            return "action:\(behavior.action)"
        }
    }
    
    /// Load interactions from disk
    private func loadInteractions() {
        guard FileManager.default.fileExists(atPath: interactionsPath.path) else {
            return
        }
        
        do {
            let data = try Data(contentsOf: interactionsPath)
            let decoder = JSONDecoder()
            let interactions = try decoder.decode([AIInteraction].self, from: data)
            
            interactionsLock.lock()
            storedInteractions = interactions
            interactionsLock.unlock()
            
            Debug.shared.log(message: "Loaded \(interactions.count) stored interactions", type: .info)
        } catch {
            Debug.shared.log(message: "Failed to load stored interactions: \(error)", type: .error)
        }
    }
    
    /// Load user behaviors from disk
    private func loadBehaviors() {
        guard FileManager.default.fileExists(atPath: behaviorsPath.path) else {
            return
        }
        
        do {
            let data = try Data(contentsOf: behaviorsPath)
            let decoder = JSONDecoder()
            let behaviors = try decoder.decode([UserBehavior].self, from: data)
            
            behaviorsLock.lock()
            userBehaviors = behaviors
            behaviorsLock.unlock()
            
            Debug.shared.log(message: "Loaded \(behaviors.count) stored user behaviors", type: .info)
        } catch {
            Debug.shared.log(message: "Failed to load stored user behaviors: \(error)", type: .error)
        }
    }
    
    /// Load app usage patterns from disk
    private func loadPatterns() {
        guard FileManager.default.fileExists(atPath: patternsPath.path) else {
            return
        }
        
        do {
            let data = try Data(contentsOf: patternsPath)
            let decoder = JSONDecoder()
            let patterns = try decoder.decode([AppUsagePattern].self, from: data)
            
            patternsLock.lock()
            appUsagePatterns = patterns
            patternsLock.unlock()
            
            Debug.shared.log(message: "Loaded \(patterns.count) stored app usage patterns", type: .info)
        } catch {
            Debug.shared.log(message: "Failed to load stored app usage patterns: \(error)", type: .error)
        }
    }
}

// MARK: - Model Types

/// Represents a single user interaction with the AI
struct AIInteraction: Codable, Identifiable, Equatable {
    let id: String
    let timestamp: Date
    let userMessage: String
    let aiResponse: String
    let detectedIntent: String
    let confidenceScore: Double
    var feedback: AIFeedback?
    let context: [String: String]?
    let appVersion: String
    let modelVersion: String
    
    static func == (lhs: AIInteraction, rhs: AIInteraction) -> Bool {
        return lhs.id == rhs.id
    }
}

/// User feedback on an AI interaction
struct AIFeedback: Codable {
    let rating: Int // 1-5 rating
    let comment: String?
}

/// Represents a user behavior within the app
struct UserBehavior: Codable, Identifiable {
    let id: String
    let timestamp: Date
    let action: String
    let screen: String
    let duration: TimeInterval
    let details: [String: String]
}

/// Represents a pattern of app usage
struct AppUsagePattern: Codable, Identifiable {
    let id: String
    let timestamp: Date
    let feature: String
    let timeSpent: TimeInterval
    let actionSequence: [String]
    let completedTask: Bool
}

/// Statistics about stored interactions and learning
struct LearningStatistics {
    let totalInteractions: Int
    let interactionsWithFeedback: Int
    let averageFeedbackRating: Double
    let behaviorCount: Int
    let patternCount: Int
    let totalDataPoints: Int
    let modelVersion: String
    let lastTrainingDate: Date?
}

/// Errors that can occur during model export
enum ExportError: Error, LocalizedError {
    case invalidPassword
    case modelNotFound
    case exportFailed(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidPassword:
            return "Invalid password provided for model export"
        case .modelNotFound:
            return "No trained model found to export"
        case .exportFailed(let error):
            return "Export failed: \(error.localizedDescription)"
        }
    }
}
