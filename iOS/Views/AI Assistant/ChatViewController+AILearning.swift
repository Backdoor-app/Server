// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

// MARK: - AI Learning Extension
extension ChatViewController {
    
    /// Record an AI interaction for learning
    func recordAIInteraction(userMessage: String, aiResponse: String, messageId: String) {
        // Extract intent and confidence
        let intent = extractIntent(from: aiResponse)
        let confidence = extractConfidence(from: aiResponse)
        
        // Record the interaction
        AILearningManager.shared.recordInteraction(
            userMessage: userMessage,
            aiResponse: aiResponse, 
            intent: intent,
            confidence: confidence
        )
        
        // Show feedback prompt after delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            self?.showFeedbackPrompt(for: messageId)
        }
    }
    
    /// Extract intent from AI response
    private func extractIntent(from response: String) -> String {
        // Look for intent in square brackets like [navigate to:settings]
        if let range = response.range(of: "\\[([a-zA-Z0-9_\\s]+):.*?\\]", options: .regularExpression) {
            let match = String(response[range])
            if let colonIndex = match.firstIndex(of: ":") {
                let startIndex = match.index(after: match.startIndex) // Skip the opening bracket
                return String(match[startIndex..<colonIndex])
            }
        }
        
        // Default intent if not found
        return "conversation"
    }
    
    /// Extract confidence score (placeholder implementation)
    private func extractConfidence(from response: String) -> Double {
        // In a real implementation, confidence could be embedded in response or derived from context
        return 0.85 // Default reasonable confidence
    }
    
    /// Show feedback prompt for AI response
    private func showFeedbackPrompt(for messageId: String) {
        // Only show if learning is enabled
        guard AILearningManager.shared.isLearningEnabled else { return }
        
        // Create feedback view
        let feedbackView = AIFeedbackView(frame: CGRect(x: 0, y: 0, width: self.view.bounds.width - 60, height: 240))
        feedbackView.translatesAutoresizingMaskIntoConstraints = false
        feedbackView.alpha = 0
        
        // Set callback
        feedbackView.onFeedbackSubmitted = { [weak self] rating, comment in
            // Record the feedback
            AILearningManager.shared.recordFeedback(
                for: messageId,
                rating: rating,
                comment: comment
            )
            
            // Optionally show a thank you message for high ratings
            if rating >= 4 {
                self?.showThankYouToast()
            }
        }
        
        // Add to view
        self.view.addSubview(feedbackView)
        
        // Set constraints
        NSLayoutConstraint.activate([
            feedbackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            feedbackView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            feedbackView.widthAnchor.constraint(equalToConstant: view.bounds.width - 60),
            feedbackView.heightAnchor.constraint(greaterThanOrEqualToConstant: 240)
        ])
        
        // Animate in
        UIView.animate(withDuration: 0.3) {
            feedbackView.alpha = 1
        }
    }
    
    /// Show a thank you toast message
    private func showThankYouToast() {
        let toast = UILabel()
        toast.text = "Thanks for your feedback!"
        toast.backgroundColor = .systemGray6
        toast.textAlignment = .center
        toast.alpha = 0
        toast.layer.cornerRadius = 10
        toast.clipsToBounds = true
        toast.padding = UIEdgeInsets(top: 8, left: 16, bottom: 8, right: 16)
        toast.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(toast)
        
        NSLayoutConstraint.activate([
            toast.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            toast.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])
        
        UIView.animate(withDuration: 0.3, animations: {
            toast.alpha = 1
        }, completion: { _ in
            UIView.animate(withDuration: 0.3, delay: 2, options: [], animations: {
                toast.alpha = 0
            }, completion: { _ in
                toast.removeFromSuperview()
            })
        })
    }
}
