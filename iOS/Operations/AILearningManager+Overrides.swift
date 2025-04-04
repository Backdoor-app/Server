// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// Extension that patches AILearningManager's training to use all user interactions
extension AILearningManager {
    
    /// Apply overrides to make the AI learn from ALL user interactions
    func applyAllInteractionTrainingOverrides() {
        // Replace the standard training evaluation with our enhanced version
        // that triggers training based on ALL interactions
        
        // Register a notification observer for app startup
        NotificationCenter.default.addObserver(
            forName: UIApplication.didFinishLaunchingNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.setupImprovedTraining()
        }
        
        // Try to set up immediately as well
        setupImprovedTraining()
        
        Debug.shared.log(message: "Applied overrides to train AI with ALL user interactions", type: .info)
    }
    
    /// Set up improved training that uses all interactions
    private func setupImprovedTraining() {
        // Configure shorter training intervals
        UserDefaults.standard.set(0.33, forKey: "AITrainingDayInterval") // Train every 8 hours
        UserDefaults.standard.set(5, forKey: "AIMinimumInteractions") // Only need 5 data points
        
        // Schedule more frequent training evaluation
        Timer.scheduledTimer(
            timeInterval: 4 * 60 * 60, // Check every 4 hours
            target: self,
            selector: #selector(improvedTrainingTimerFired),
            userInfo: nil,
            repeats: true
        )
        
        // Run an initial evaluation
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 60) { [weak self] in
            self?.evaluateTrainingWithAllInteractions()
        }
    }
    
    /// Timer handler for improved training
    @objc private func improvedTrainingTimerFired() {
        evaluateTrainingWithAllInteractions()
    }
    
    /// Override the standard trainModelNow with our enhanced version
    func trainModelNowWithAllData(completion: @escaping (Bool, String) -> Void) {
        trainModelWithAllInteractionsNow(completion: completion)
    }
}
