// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// Extension to ensure proper async/await usage in view controllers
extension ModelServerIntegrationViewController {
    
    /// Safe wrapper for async tasks that ensures proper await usage
    func performAsyncSafely(_ task: @escaping () async -> Void) {
        Task {
            await task()
        }
    }
    
    /// Safe method to check server status with proper async/await handling
    func checkServerStatusSafely() {
        performAsyncSafely { [weak self] in
            do {
                let modelInfo = try await BackdoorAIClient.shared.getLatestModelInfo()
                DispatchQueue.main.async {
                    self?.updateServerStatusUI(status: "Online", message: "Latest model: \(modelInfo.latestModelVersion)", isError: false)
                }
            } catch {
                DispatchQueue.main.async {
                    self?.updateServerStatusUI(status: "Error", message: error.localizedDescription, isError: true)
                }
            }
        }
    }
    
    /// Updates the server status UI using the public method instead of direct label access
    private func updateServerStatusUI(status: String, message: String, isError: Bool) {
        let statusText = "Server status: \(status)\n\(message)"
        updateStatusLabel(text: statusText, isError: isError)
    }
    
    /// Updates the UI status label with the provided text and styling
    func updateStatusLabel(text: String, isError: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // Find the label by tag if it's not directly accessible
            if let statusLabel = self.view.viewWithTag(1001) as? UILabel {
                statusLabel.text = text
                statusLabel.textColor = isError ? .systemRed : .systemGreen
            } else {
                // Create the label if it doesn't exist yet
                let label = UILabel()
                label.tag = 1001
                label.text = text
                label.textColor = isError ? .systemRed : .systemGreen
                label.numberOfLines = 0
                label.textAlignment = .center
                label.font = .systemFont(ofSize: 16, weight: .medium)
                label.translatesAutoresizingMaskIntoConstraints = false
                
                self.view.addSubview(label)
                
                NSLayoutConstraint.activate([
                    label.centerXAnchor.constraint(equalTo: self.view.centerXAnchor),
                    label.topAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.topAnchor, constant: 16),
                    label.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                    label.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20)
                ])
            }
        }
    }
    
    /// Safe wrapper for model uploads using proper async/await
    func uploadModelSafely(completion: @escaping (Bool, String) -> Void) {
        performAsyncSafely { [weak self] in
            guard let self = self else { return }
            
            let result = await AILearningManager.shared.uploadTrainedModelToServer()
            
            DispatchQueue.main.async {
                completion(result.success, result.message)
            }
        }
    }
}
