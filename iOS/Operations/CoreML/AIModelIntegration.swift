// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML
import UIKit

/// Extension to CustomAIService that integrates CoreML model predictions
extension CustomAIService {
    
    /// Enhanced analyze user intent using CoreML
    func analyzeUserIntentWithML(message: String, completion: @escaping (MessageIntent) -> Void) {
        // Start with traditional pattern matching as a fallback
        let patternBasedIntent = analyzeUserIntent(message: message)
        
        // Try to enhance with ML model
        CoreMLManager.shared.predictIntent(from: message) { result in
            switch result {
            case .success(let prediction):
                // Only use ML prediction if confidence is high enough
                if prediction.confidence > 0.6 {
                    // Convert ML model's intent to our MessageIntent type
                    let enhancedIntent = self.convertMLIntentToMessageIntent(
                        intent: prediction.intent,
                        parameters: prediction.parameters,
                        message: message
                    )
                    completion(enhancedIntent)
                } else {
                    // Fall back to pattern-based intent if ML confidence is low
                    Debug.shared.log(message: "ML confidence too low (\(prediction.confidence)), using pattern matching", type: .debug)
                    completion(patternBasedIntent)
                }
                
            case .failure(let error):
                // Log error and fall back to pattern matching
                Debug.shared.log(message: "ML intent prediction failed: \(error.localizedDescription), using pattern matching", type: .warning)
                completion(patternBasedIntent)
            }
        }
    }
    
    /// Convert ML model intent format to our MessageIntent enum
    private func convertMLIntentToMessageIntent(intent: String, parameters: [String: Any], message: String) -> MessageIntent {
        // Map the ML model's intent to our MessageIntent format
        switch intent.lowercased() {
        case "sign_app", "signing":
            if let appName = parameters["appName"] as? String {
                return .appSign(appName: appName)
            }
            // Fall through to generic case if no app name
            return .unknown
            
        case "navigate", "navigation":
            if let destination = parameters["destination"] as? String {
                return .appNavigation(destination: destination)
            }
            // Fall through to generic case if no destination
            return .unknown
            
        case "add_source", "source":
            if let url = parameters["url"] as? String {
                return .sourceAdd(url: url)
            }
            // Fall through to generic case if no URL
            return .unknown
            
        case "install_app", "install":
            if let appName = parameters["appName"] as? String {
                return .appInstall(appName: appName)
            }
            // Fall through to generic case if no app name
            return .unknown
            
        case "greeting", "hello":
            return .greeting
            
        case "help", "assistance":
            return .generalHelp
            
        case "question", "query":
            if let topic = parameters["topic"] as? String {
                return .question(topic: topic)
            } else {
                // If no specific topic found, extract from message
                let topic = message.replacing(regularExpression: "\\?|what|how|when|where|why|who|is|are|can|could|would|will|should", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                return .question(topic: topic)
            }
            
        default:
            // Return unknown intent, will be handled by the main service
            return .unknown
        }
    }
    
    /// Enhanced generate response using CoreML
    func generateResponseWithML(
        intent: MessageIntent,
        userMessage: String,
        conversationHistory: [AIMessagePayload],
        conversationContext: String,
        appContext: AppContext,
        completion: @escaping (String) -> Void
    ) {
        // First, analyze the sentiment of the message to adapt our response tone
        CoreMLManager.shared.analyzeSentiment(from: userMessage) { result in
            // Default to neutral sentiment if analysis fails
            let sentiment: SentimentType = result.map { $0.sentiment }.getOrElse(.neutral)
            
            // Get the standard response from our rule-based system
            let standardResponse = self.generateResponse(
                intent: intent,
                userMessage: userMessage,
                conversationHistory: conversationHistory,
                conversationContext: conversationContext,
                appContext: appContext
            )
            
            // Adapt the response based on sentiment
            let enhancedResponse = self.adaptResponseToSentiment(
                response: standardResponse,
                sentiment: sentiment
            )
            
            completion(enhancedResponse)
        }
    }
    
    /// Adapt response based on detected sentiment
    private func adaptResponseToSentiment(response: String, sentiment: SentimentType) -> String {
        switch sentiment {
        case .positive:
            // For positive sentiment, keep the response enthusiastic
            return response
            
        case .negative:
            // For negative sentiment, add a more empathetic prefix
            let empathyPrefixes = [
                "I understand your frustration. ",
                "I'm sorry to hear that. ",
                "Let me help resolve that for you. ",
                "I'll do my best to help with this issue. "
            ]
            
            // Only add prefix if it doesn't already have one
            if !response.contains("I understand") && !response.contains("I'm sorry") {
                let prefix = empathyPrefixes.randomElement() ?? ""
                return prefix + response
            }
            return response
            
        case .neutral:
            // For neutral sentiment, use the standard response
            return response
        }
    }
}

// MARK: - Helper Extensions

extension Result {
    /// Get the success value or return a default
    func getOrElse(_ defaultValue: Success) -> Success {
        switch self {
        case .success(let value):
            return value
        case .failure:
            return defaultValue
        }
    }
}
