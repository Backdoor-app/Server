// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// View controller for integrating with the AI server, including model upload
class ModelServerIntegrationViewController: UIViewController {
    
    // UI Elements
    private let scrollView = UIScrollView()
    private let contentView = UIView()
    private let serverSyncSwitch = UISwitch()
    private let uploadModelButton = UIButton(type: .system)
    private let statusLabel = UILabel()
    private let activityIndicator = UIActivityIndicatorView(style: .medium)
    private let lastUploadLabel = UILabel()
    private let modelInfoLabel = UILabel()
    private let serverStatusLabel = UILabel()
    private let explanationLabel = UILabel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "AI Server Integration"
        view.backgroundColor = UIColor(named: "Background") ?? .systemBackground
        
        setupUI()
        updateStatus()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        updateStatus()
    }
    
    private func setupUI() {
        // Setup scroll view
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        contentView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scrollView)
        scrollView.addSubview(contentView)
        
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            
            contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            contentView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
            contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor)
        ])
        
        // Server sync section
        let serverSyncLabel = UILabel()
        serverSyncLabel.text = "Server Synchronization"
        serverSyncLabel.font = .systemFont(ofSize: 17, weight: .bold)
        
        serverSyncSwitch.isOn = AILearningManager.shared.isServerSyncEnabled
        serverSyncSwitch.addTarget(self, action: #selector(serverSyncSwitchChanged), for: .valueChanged)
        
        let serverSyncDescriptionLabel = UILabel()
        serverSyncDescriptionLabel.text = "Enable synchronization with the AI server to improve model performance"
        serverSyncDescriptionLabel.font = .systemFont(ofSize: 14)
        serverSyncDescriptionLabel.textColor = .secondaryLabel
        serverSyncDescriptionLabel.numberOfLines = 0
        
        // Upload model section
        let uploadSectionLabel = UILabel()
        uploadSectionLabel.text = "Model Upload"
        uploadSectionLabel.font = .systemFont(ofSize: 17, weight: .bold)
        
        uploadModelButton.setTitle("Upload Trained Model", for: .normal)
        uploadModelButton.addTarget(self, action: #selector(uploadModelButtonTapped), for: .touchUpInside)
        uploadModelButton.backgroundColor = .systemBlue
        uploadModelButton.setTitleColor(.white, for: .normal)
        uploadModelButton.layer.cornerRadius = 10
        
        statusLabel.font = .systemFont(ofSize: 14)
        statusLabel.textColor = .secondaryLabel
        statusLabel.numberOfLines = 0
        statusLabel.text = "No status"
        
        activityIndicator.hidesWhenStopped = true
        
        lastUploadLabel.font = .systemFont(ofSize: 14)
        lastUploadLabel.textColor = .secondaryLabel
        lastUploadLabel.numberOfLines = 0
        lastUploadLabel.text = "Never uploaded"
        
        modelInfoLabel.font = .systemFont(ofSize: 14)
        modelInfoLabel.textColor = .secondaryLabel
        modelInfoLabel.numberOfLines = 0
        modelInfoLabel.text = "No model information"
        
        serverStatusLabel.font = .systemFont(ofSize: 14)
        serverStatusLabel.textColor = .secondaryLabel
        serverStatusLabel.numberOfLines = 0
        serverStatusLabel.text = "Server status: Unknown"
        
        explanationLabel.font = .systemFont(ofSize: 14)
        explanationLabel.textColor = .secondaryLabel
        explanationLabel.numberOfLines = 0
        explanationLabel.text = "When you upload your trained model to the server, it is combined with models from other users to create an improved ensemble model. This helps everyone get better AI performance. Your data remains private, only the model parameters are shared."
        
        // Add views to hierarchy
        [serverSyncLabel, serverSyncSwitch, serverSyncDescriptionLabel, 
         uploadSectionLabel, uploadModelButton, statusLabel, activityIndicator,
         lastUploadLabel, modelInfoLabel, serverStatusLabel, explanationLabel].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            contentView.addSubview($0)
        }
        
        // Setup constraints
        NSLayoutConstraint.activate([
            serverSyncLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 20),
            serverSyncLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            
            serverSyncSwitch.centerYAnchor.constraint(equalTo: serverSyncLabel.centerYAnchor),
            serverSyncSwitch.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            serverSyncDescriptionLabel.topAnchor.constraint(equalTo: serverSyncLabel.bottomAnchor, constant: 10),
            serverSyncDescriptionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            serverSyncDescriptionLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            uploadSectionLabel.topAnchor.constraint(equalTo: serverSyncDescriptionLabel.bottomAnchor, constant: 30),
            uploadSectionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            
            uploadModelButton.topAnchor.constraint(equalTo: uploadSectionLabel.bottomAnchor, constant: 15),
            uploadModelButton.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            uploadModelButton.widthAnchor.constraint(equalToConstant: 200),
            uploadModelButton.heightAnchor.constraint(equalToConstant: 44),
            
            activityIndicator.topAnchor.constraint(equalTo: uploadModelButton.bottomAnchor, constant: 15),
            activityIndicator.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            
            statusLabel.topAnchor.constraint(equalTo: activityIndicator.bottomAnchor, constant: 15),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            modelInfoLabel.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 20),
            modelInfoLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            modelInfoLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            lastUploadLabel.topAnchor.constraint(equalTo: modelInfoLabel.bottomAnchor, constant: 15),
            lastUploadLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            lastUploadLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            serverStatusLabel.topAnchor.constraint(equalTo: lastUploadLabel.bottomAnchor, constant: 15),
            serverStatusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            serverStatusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            explanationLabel.topAnchor.constraint(equalTo: serverStatusLabel.bottomAnchor, constant: 30),
            explanationLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            explanationLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            explanationLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -30)
        ])
    }
    
    private func updateStatus() {
        let isServerSyncEnabled = AILearningManager.shared.isServerSyncEnabled
        serverSyncSwitch.isOn = isServerSyncEnabled
        
        // Update UI based on server sync status
        if isServerSyncEnabled {
            // Check if model is available
            if AILearningManager.shared.isTrainedModelAvailableForUpload() {
                uploadModelButton.isEnabled = true
                
                // Display model info
                let modelInfo = AILearningManager.shared.getTrainedModelInfo()
                var modelInfoText = "Model version: \(modelInfo.version)"
                if let trainDate = modelInfo.date {
                    let dateFormatter = DateFormatter()
                    dateFormatter.dateStyle = .medium
                    dateFormatter.timeStyle = .short
                    modelInfoText += "\nTrained on: \(dateFormatter.string(from: trainDate))"
                }
                modelInfoLabel.text = modelInfoText
                
                statusLabel.text = "Ready to upload"
                statusLabel.textColor = .systemGreen
            } else {
                uploadModelButton.isEnabled = false
                modelInfoLabel.text = "No trained model available yet"
                statusLabel.text = "Train a model in AI Learning settings before uploading"
                statusLabel.textColor = .secondaryLabel
            }
            
            // Last upload date
            if let lastUploadDate = UserDefaults.standard.object(forKey: "lastModelUploadDate") as? Date {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                dateFormatter.timeStyle = .short
                lastUploadLabel.text = "Last uploaded: \(dateFormatter.string(from: lastUploadDate))"
            } else {
                lastUploadLabel.text = "Never uploaded"
            }
            
            // Server status
            serverStatusLabel.text = "Checking server status..."
            
            // Check server status
            Task {
                do {
                    let modelInfo = try await BackdoorAIClient.shared.getLatestModelInfo()
                    DispatchQueue.main.async { [weak self] in
                        self?.serverStatusLabel.text = "Server status: Online\nLatest model: \(modelInfo.latestModelVersion)"
                        self?.serverStatusLabel.textColor = .systemGreen
                    }
                } catch {
                    DispatchQueue.main.async { [weak self] in
                        self?.serverStatusLabel.text = "Server status: Error - \(error.localizedDescription)"
                        self?.serverStatusLabel.textColor = .systemRed
                    }
                }
            }
        } else {
            // Server sync disabled
            uploadModelButton.isEnabled = false
            statusLabel.text = "Server sync is disabled"
            statusLabel.textColor = .systemRed
            modelInfoLabel.text = "Enable server sync to upload models"
            lastUploadLabel.text = "Sync disabled"
            serverStatusLabel.text = "Server status: Connection disabled"
            serverStatusLabel.textColor = .systemRed
        }
    }
    
    @objc private func serverSyncSwitchChanged() {
        // Update the server sync setting
        AILearningManager.shared.setServerSyncEnabled(serverSyncSwitch.isOn)
        
        // Update UI
        updateStatus()
    }
    
    @objc private func uploadModelButtonTapped() {
        // Disable UI during upload
        uploadModelButton.isEnabled = false
        activityIndicator.startAnimating()
        statusLabel.text = "Uploading model to server..."
        statusLabel.textColor = .systemOrange
        
        // Upload the model
        Task {
            let result = await AILearningManager.shared.uploadTrainedModelToServer()
            
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                
                if result.success {
                    self?.statusLabel.text = "Upload successful: \(result.message)"
                    self?.statusLabel.textColor = .systemGreen
                    
                    // Save upload date
                    UserDefaults.standard.set(Date(), forKey: "lastModelUploadDate")
                    
                    // Show success alert
                    let alert = UIAlertController(
                        title: "Upload Successful",
                        message: result.message,
                        preferredStyle: .alert
                    )
                    alert.addAction(UIAlertAction(title: "OK", style: .default))
                    self?.present(alert, animated: true)
                } else {
                    self?.statusLabel.text = "Upload failed: \(result.message)"
                    self?.statusLabel.textColor = .systemRed
                    self?.uploadModelButton.isEnabled = true
                    
                    // Show error alert
                    let alert = UIAlertController(
                        title: "Upload Failed",
                        message: result.message,
                        preferredStyle: .alert
                    )
                    alert.addAction(UIAlertAction(title: "Try Again", style: .default) { _ in
                        self?.uploadModelButtonTapped()
                    })
                    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
                    self?.present(alert, animated: true)
                }
                
                // Update UI
                self?.updateStatus()
            }
        }
    }
}
