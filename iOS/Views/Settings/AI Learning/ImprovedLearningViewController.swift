// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// View controller for improved AI learning settings
class ImprovedLearningViewController: UIViewController {
    
    // UI Elements
    private let tableView = UITableView(frame: .zero, style: .insetGrouped)
    private let headerView = UIView()
    private let headerLabel = UILabel()
    private let trainButton = UIButton(type: .system)
    private let activityIndicator = UIActivityIndicatorView(style: .medium)
    
    // Settings sections
    private enum Section: Int, CaseIterable {
        case main
        case learning
        case upload
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "Advanced AI Learning"
        view.backgroundColor = UIColor(named: "SettingsBackground") ?? .systemGroupedBackground
        
        setupUI()
        
        // Apply the overrides to ensure AI learns from ALL interactions
        AILearningManager.shared.applyAllInteractionTrainingOverrides()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Refresh the table view
        tableView.reloadData()
        
        // Update the stats header
        updateStatsHeader()
    }
    
    private func setupUI() {
        // Configure table view
        tableView.delegate = self
        tableView.dataSource = self
        tableView.backgroundColor = .clear
        tableView.register(ImprovedLearningSettingsCell.self, forCellReuseIdentifier: "SettingsCell")
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "DefaultCell")
        
        // Configure header view
        headerLabel.font = .systemFont(ofSize: 14)
        headerLabel.textColor = .secondaryLabel
        headerLabel.numberOfLines = 0
        headerLabel.textAlignment = .center
        
        // Configure train button
        trainButton.setTitle("Train Model Now", for: .normal)
        trainButton.backgroundColor = .systemBlue
        trainButton.setTitleColor(.white, for: .normal)
        trainButton.layer.cornerRadius = 10
        trainButton.addTarget(self, action: #selector(trainButtonPressed), for: .touchUpInside)
        
        // Configure activity indicator
        activityIndicator.hidesWhenStopped = true
        
        // Add subviews
        headerView.addSubview(headerLabel)
        headerView.addSubview(trainButton)
        headerView.addSubview(activityIndicator)
        view.addSubview(tableView)
        
        // Configure constraints
        headerLabel.translatesAutoresizingMaskIntoConstraints = false
        trainButton.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        tableView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerLabel.topAnchor.constraint(equalTo: headerView.topAnchor, constant: 16),
            headerLabel.leadingAnchor.constraint(equalTo: headerView.leadingAnchor, constant: 16),
            headerLabel.trailingAnchor.constraint(equalTo: headerView.trailingAnchor, constant: -16),
            
            trainButton.topAnchor.constraint(equalTo: headerLabel.bottomAnchor, constant: 16),
            trainButton.centerXAnchor.constraint(equalTo: headerView.centerXAnchor),
            trainButton.widthAnchor.constraint(equalToConstant: 200),
            trainButton.heightAnchor.constraint(equalToConstant: 44),
            
            activityIndicator.topAnchor.constraint(equalTo: trainButton.bottomAnchor, constant: 8),
            activityIndicator.centerXAnchor.constraint(equalTo: headerView.centerXAnchor),
            activityIndicator.bottomAnchor.constraint(equalTo: headerView.bottomAnchor, constant: -16),
            
            tableView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor)
        ])
        
        // Setup table header
        headerView.frame = CGRect(x: 0, y: 0, width: view.bounds.width, height: 150)
        tableView.tableHeaderView = headerView
        
        // Update the stats header
        updateStatsHeader()
    }
    
    private func updateStatsHeader() {
        // Get learning statistics
        let stats = AILearningManager.shared.getLearningStatistics()
        
        // Format date
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        let lastTrainingText = stats.lastTrainingDate != nil ? 
            dateFormatter.string(from: stats.lastTrainingDate!) : "Never"
        
        // Update header text
        headerLabel.text = """
        AI learning is currently: \(AILearningManager.shared.isLearningEnabled ? "Enabled" : "Disabled")
        
        Current model version: \(stats.modelVersion)
        Last trained: \(lastTrainingText)
        
        Total interactions: \(stats.totalInteractions)
        User behaviors: \(stats.behaviorCount)
        Usage patterns: \(stats.patternCount)
        
        Total data points: \(stats.totalDataPoints)
        """
    }
    
    @objc private func trainButtonPressed() {
        // Disable button and show activity indicator
        trainButton.isEnabled = false
        activityIndicator.startAnimating()
        
        // Train model with all interactions
        AILearningManager.shared.trainModelWithAllInteractionsNow { [weak self] success, message in
            // Re-enable button and hide activity indicator
            self?.trainButton.isEnabled = true
            self?.activityIndicator.stopAnimating()
            
            // Update UI
            self?.updateStatsHeader()
            
            // Show alert with result
            let alert = UIAlertController(
                title: success ? "Training Successful" : "Training Failed",
                message: message,
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            self?.present(alert, animated: true)
        }
    }
}

// MARK: - UITableViewDelegate & UITableViewDataSource

extension ImprovedLearningViewController: UITableViewDelegate, UITableViewDataSource {
    
    func numberOfSections(in tableView: UITableView) -> Int {
        return Section.allCases.count
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        switch Section(rawValue: section) {
        case .main:
            return 1
        case .learning:
            return 1
        case .upload:
            return 1
        default:
            return 0
        }
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let section = Section(rawValue: indexPath.section) else {
            return UITableViewCell()
        }
        
        switch section {
        case .main:
            let cell = tableView.dequeueReusableCell(withIdentifier: "SettingsCell", for: indexPath) as! ImprovedLearningSettingsCell
            cell.configure(
                title: "Enable AI Learning",
                description: "Enable on-device AI learning from ALL your interactions with the app",
                isOn: AILearningManager.shared.isLearningEnabled
            )
            cell.toggleAction = { [weak self] isOn in
                AILearningManager.shared.setLearningEnabled(isOn)
                self?.updateStatsHeader()
            }
            return cell
            
        case .learning:
            let cell = tableView.dequeueReusableCell(withIdentifier: "SettingsCell", for: indexPath) as! ImprovedLearningSettingsCell
            cell.configure(
                title: "Enhanced Learning",
                description: "Learn from ALL interactions, not just rated ones. The AI will continuously improve as you use the app.",
                isOn: true,
                status: "Active - Learning from all interactions"
            )
            return cell
            
        case .upload:
            let cell = tableView.dequeueReusableCell(withIdentifier: "SettingsCell", for: indexPath) as! ImprovedLearningSettingsCell
            cell.configure(
                title: "Server Sync",
                description: "Upload your trained AI model to the server for ensemble training. Your personal data stays private, only the model is shared.",
                isOn: AILearningManager.shared.isServerSyncEnabled
            )
            cell.toggleAction = { [weak self] isOn in
                AILearningManager.shared.setServerSyncEnabled(isOn)
                self?.updateStatsHeader()
            }
            return cell
        }
    }
    
    func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        guard let section = Section(rawValue: section) else {
            return nil
        }
        
        switch section {
        case .main:
            return "AI Learning"
        case .learning:
            return "Enhanced Learning"
        case .upload:
            return "Server Integration"
        }
    }
    
    func tableView(_ tableView: UITableView, titleForFooterInSection section: Int) -> String? {
        guard let section = Section(rawValue: section) else {
            return nil
        }
        
        switch section {
        case .main:
            return "When enabled, the AI will learn from your interactions with the app."
        case .learning:
            return "The enhanced learning system uses ALL your interactions, behaviors, and usage patterns to continuously improve the AI model."
        case .upload:
            return "When enabled, your trained model will be uploaded to the server and combined with other users' models to create an improved ensemble model."
        }
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        
        guard let section = Section(rawValue: indexPath.section) else {
            return
        }
        
        switch section {
        case .upload:
            if AILearningManager.shared.isServerSyncEnabled {
                // Navigate to server integration view
                let serverVC = ModelServerIntegrationViewController()
                navigationController?.pushViewController(serverVC, animated: true)
            }
        default:
            break
        }
    }
}
