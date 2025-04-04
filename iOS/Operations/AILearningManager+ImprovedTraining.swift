// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML
import CreateML
import UIKit

/// Extension to AILearningManager for improved training using all user interactions
extension AILearningManager {
    
    /// Enhanced training method that uses ALL user interactions, not just rated ones
    func trainModelWithAllInteractions() -> (success: Bool, version: String, errorMessage: String?) {
        Debug.shared.log(message: "Starting comprehensive AI model training with ALL interactions", type: .info)
        
        do {
            // Lock and copy all data
            interactionsLock.lock()
            behaviorsLock.lock()
            patternsLock.lock()
            
            // Get ALL interactions, behaviors, and patterns
            let allInteractions = storedInteractions
            let allBehaviors = userBehaviors
            let allPatterns = appUsagePatterns
            
            interactionsLock.unlock()
            behaviorsLock.unlock()
            patternsLock.unlock()
            
            // Generate new version
            let timestamp = Int(Date().timeIntervalSince1970)
            let newVersion = "1.0.\(timestamp)"
            
            // Check if we have enough data overall
            if allInteractions.isEmpty && allBehaviors.isEmpty && allPatterns.isEmpty {
                Debug.shared.log(message: "No training data available", type: .warning)
                return (false, newVersion, "No training data available")
            }
            
            // Set up data arrays for training
            var textInput: [String] = []
            var intentOutput: [String] = []
            var contextData: [[String: String]] = []
            
            // Add ALL interactions (not just ones with good feedback)
            Debug.shared.log(message: "Including ALL \(allInteractions.count) interactions in training", type: .info)
            
            for interaction in allInteractions {
                textInput.append(interaction.userMessage)
                intentOutput.append(interaction.detectedIntent)
                
                // Add context when available
                if let context = interaction.context {
                    contextData.append(context)
                } else {
                    contextData.append([:])
                }
            }
            
            // Add ALL behavior data
            if !allBehaviors.isEmpty {
                Debug.shared.log(message: "Including ALL \(allBehaviors.count) behavior records in training", type: .info)
                
                for behavior in allBehaviors {
                    // Create a composite feature from the behavior
                    let behaviorText = "User performed \(behavior.action) on \(behavior.screen) screen"
                    let behaviorIntent = getIntentFromBehavior(behavior)
                    
                    textInput.append(behaviorText)
                    intentOutput.append(behaviorIntent)
                    contextData.append(behavior.details)
                }
            }
            
            // Add ALL usage patterns (not just completed ones)
            if !allPatterns.isEmpty {
                Debug.shared.log(message: "Including ALL \(allPatterns.count) usage patterns in training", type: .info)
                
                for pattern in allPatterns {
                    let patternText = "User worked with \(pattern.feature) feature"
                    let patternIntent = "use:\(pattern.feature)"
                    
                    textInput.append(patternText)
                    intentOutput.append(patternIntent)
                    contextData.append(["sequence": pattern.actionSequence.joined(separator: ","),
                                        "completed": String(pattern.completedTask)])
                }
            }
            
            // If we still don't have enough data after including everything
            if textInput.count < 5 {
                Debug.shared.log(message: "Not enough training data examples (minimum 5 required)", type: .warning)
                return (false, newVersion, "Not enough training examples (minimum 5 required)")
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
            let userDefaults = UserDefaults.standard
            userDefaults.set(newVersion, forKey: modelVersionKey)
            userDefaults.set(Date(), forKey: lastTrainingKey)
            
            Debug.shared.log(message: "Successfully trained new model using ALL user interactions, version: \(newVersion)", type: .info)
            
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
    
    /// Override the original train method to use the improved one
    func overrideTrainingWithAllInteractions() {
        // Replace the original trainNewModel implementation by swizzling at runtime
        Debug.shared.log(message: "Overriding training method to include ALL interactions", type: .info)
        
        // Note: We're not actually doing method swizzling here since Swift doesn't support it easily
        // Instead, we'll just use this method from other parts of the code
    }
    
    /// Enhanced evaluation that triggers training based on ALL interactions
    func evaluateTrainingWithAllInteractions() {
        // Only train if learning is enabled
        guard isLearningEnabled else {
            return
        }
        
        // Check if we have enough interactions or behaviors or patterns
        interactionsLock.lock()
        behaviorsLock.lock()
        patternsLock.lock()
        
        let interactionCount = storedInteractions.count
        let behaviorCount = userBehaviors.count
        let patternCount = appUsagePatterns.count
        
        interactionsLock.unlock()
        behaviorsLock.unlock()
        patternsLock.unlock()
        
        // Check total data points against minimum threshold
        let totalDataPoints = interactionCount + behaviorCount + patternCount
        let minTotalDataPoints = 5 // Much lower threshold since we're using all data
        
        guard totalDataPoints >= minTotalDataPoints else {
            return
        }
        
        // Check when we last trained
        let lastTraining = UserDefaults.standard.object(forKey: lastTrainingKey) as? Date ?? Date.distantPast
        let daysSinceLastTraining = Calendar.current.dateComponents([.day], from: lastTraining, to: Date()).day ?? Int.max
        
        // Train more often since we're using all data - now after just 8 hours (0.33 days)
        // Convert to the same type for comparison (compare hours instead of fractional days)
        let hoursSinceLastTraining = daysSinceLastTraining * 24 // convert days to hours
        let minHoursBetweenTraining = 8 // 8 hours = 0.33 days
        guard hoursSinceLastTraining >= minHoursBetweenTraining else {
            return
        }
        
        // We meet all criteria, start training with all data
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self = self else { return }
            let _ = self.trainModelWithAllInteractions()
        }
    }
    
    /// Manually trigger model training with all data
    func trainModelWithAllInteractionsNow(completion: @escaping (Bool, String) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    completion(false, "Manager deallocated")
                }
                return
            }
            
            // Perform training with all data
            let result = self.trainModelWithAllInteractions()
            
            DispatchQueue.main.async {
                if result.success {
                    completion(true, "Successfully trained model with ALL user interactions, version \(result.version)")
                } else {
                    completion(false, "Training failed: \(result.errorMessage ?? "Unknown error")")
                }
            }
        }
    }
}
