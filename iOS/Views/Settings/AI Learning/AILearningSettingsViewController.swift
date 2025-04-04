// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// View controller for managing AI learning settings
class AILearningSettingsViewController: UITableViewController {
    
    // MARK: - Properties
    
    private let cellReuseIdentifier = "AILearningSettingCell"
    private let switchCellReuseIdentifier = "AILearningSwitchCell"
    
    // Activity indicator for training
    private let activityIndicator = UIActivityIndicatorView(style: .medium)
    
    // Section indices
    private enum Section: Int {
        case about = 0
        case settings = 1
        case serverSettings = 2
        case status = 3
        case actions = 4
        case export = 5
    }
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "AI Learning"
        
        // Configure table view
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: cellReuseIdentifier)
        tableView.register(SwitchTableViewCell.self, forCellReuseIdentifier: switchCellReuseIdentifier)
        tableView.tableFooterView = UIView()
        
        // Add observer for model updates
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(modelUpdated),
            name: Notification.Name("AIModelUpdated"),
            object: nil
        )
        
        // Set up activity indicator in navigation bar
        activityIndicator.hidesWhenStopped = true
        let barButton = UIBarButtonItem(customView: activityIndicator)
        navigationItem.rightBarButtonItem = barButton
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        tableView.reloadData()
    }
    
    @objc private func modelUpdated() {
        DispatchQueue.main.async {
            self.activityIndicator.stopAnimating()
            self.tableView.reloadSections(IndexSet(integer: Section.status.rawValue), with: .automatic)
            
            // Show a notification
            let alert = UIAlertController(
                title: "AI Model Updated",
                message: "The AI has been updated with your feedback on previous interactions.",
                preferredStyle: .alert
            )
            
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            
            self.present(alert, animated: true)
        }
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - Table View Data Source
    
    override func numberOfSections(in tableView: UITableView) -> Int {
        return 6
    }
    
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        switch Section(rawValue: section) {
        case .about:
            return 1
        case .settings:
            return 1
        case .serverSettings:
            return 1 // Removed server configuration option
        case .status:
            return 5
        case .actions:
            return 3
        case .export:
            return 1
        case .none:
            return 0
        }
    }
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        switch Section(rawValue: indexPath.section) {
        case .about:
            let cell = tableView.dequeueReusableCell(withIdentifier: cellReuseIdentifier, for: indexPath)
            cell.textLabel?.text = "The AI assistant can learn from your feedback and improve over time."
            cell.textLabel?.numberOfLines = 0
            cell.accessoryType = .none
            cell.selectionStyle = .none
            return cell
            
        case .settings:
            let cell = tableView.dequeueReusableCell(withIdentifier: switchCellReuseIdentifier, for: indexPath) as! SwitchTableViewCell
            cell.textLabel?.text = "Enable AI Learning"
            cell.switchControl.isOn = AILearningManager.shared.isLearningEnabled
            cell.switchValueChanged = { isOn in
                AILearningManager.shared.setLearningEnabled(isOn)
                self.tableView.reloadSections(IndexSet(integer: Section.actions.rawValue), with: .automatic)
            }
            return cell
            
        case .serverSettings:
            let cell = tableView.dequeueReusableCell(withIdentifier: switchCellReuseIdentifier, for: indexPath) as! SwitchTableViewCell
            cell.textLabel?.text = "Enable Server Sync"
            cell.switchControl.isOn = AILearningManager.shared.isServerSyncEnabled
            cell.switchValueChanged = { isOn in
                AILearningManager.shared.setServerSyncEnabled(isOn)
                self.tableView.reloadSections([Section.serverSettings.rawValue, Section.actions.rawValue], with: .automatic)
            }
            return cell
            
        case .status:
            let cell = tableView.dequeueReusableCell(withIdentifier: cellReuseIdentifier, for: indexPath)
            cell.selectionStyle = .none
            
            let stats = AILearningManager.shared.getLearningStatistics()
            
            if indexPath.row == 0 {
                cell.textLabel?.text = "Current model version: \(stats.modelVersion)"
            } else if indexPath.row == 1 {
                cell.textLabel?.text = "Stored interactions: \(stats.totalInteractions)"
            } else if indexPath.row == 2 {
                cell.textLabel?.text = "Average feedback rating: \(String(format: "%.1f", stats.averageFeedbackRating))"
            } else if indexPath.row == 3 {
                cell.textLabel?.text = "User behaviors tracked: \(stats.behaviorCount)"
            } else {
                cell.textLabel?.text = "Total learning data points: \(stats.totalDataPoints)"
            }
            
            return cell
            
        case .actions:
            let cell = tableView.dequeueReusableCell(withIdentifier: cellReuseIdentifier, for: indexPath)
            
            if indexPath.row == 0 {
                cell.textLabel?.text = "Train Model Now"
                cell.accessoryType = .disclosureIndicator
                cell.selectionStyle = .default
                // Disable if learning is disabled or server sync is enabled
                let enabled = AILearningManager.shared.isLearningEnabled && !AILearningManager.shared.isServerSyncEnabled
                cell.isUserInteractionEnabled = enabled
                cell.textLabel?.isEnabled = enabled
            } else if indexPath.row == 1 {
                cell.textLabel?.text = "Sync with Server Now"
                cell.accessoryType = .disclosureIndicator
                cell.selectionStyle = .default
                // Disable if server sync is disabled
                let enabled = AILearningManager.shared.isServerSyncEnabled
                cell.isUserInteractionEnabled = enabled
                cell.textLabel?.isEnabled = enabled
            } else {
                cell.textLabel?.text = "Clear Stored Interactions"
                cell.accessoryType = .disclosureIndicator
                cell.selectionStyle = .default
            }
            
            return cell
            
        case .export:
            let cell = tableView.dequeueReusableCell(withIdentifier: cellReuseIdentifier, for: indexPath)
            cell.textLabel?.text = "Export Trained Model"
            cell.accessoryType = .disclosureIndicator
            cell.textLabel?.textColor = .systemBlue
            return cell
            
        case .none:
            return UITableViewCell()
        }
    }
    
    override func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        switch Section(rawValue: section) {
        case .about:
            return "About AI Learning"
        case .settings:
            return "Settings"
        case .serverSettings:
            return "Server Settings"
        case .status:
            return "Status"
        case .actions:
            return "Actions"
        case .export:
            return "Advanced"
        case .none:
            return nil
        }
    }
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        
        if indexPath.section == Section.actions.rawValue {
            if indexPath.row == 0 {
                trainModelNow()
            } else if indexPath.row == 1 {
                syncWithServerNow()
            } else {
                promptClearInteractions()
            }
        } else if indexPath.section == Section.export.rawValue {
            promptExportModel()
        }
    }
    
    // MARK: - Actions
    
    // Server configuration is now secured with hardcoded values and not user-configurable
    
    private func syncWithServerNow() {
        // Show activity indicator
        activityIndicator.startAnimating()
        
        // Start sync
        Task {
            await AILearningManager.shared.syncWithServer()
            
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                self.showInfoAlert(title: "Sync Complete", message: "Data has been synchronized with the server.")
                self.tableView.reloadSections(IndexSet(integer: Section.status.rawValue), with: .automatic)
            }
        }
    }
    
    private func promptExportModel() {
        let alert = UIAlertController(
            title: "Export Trained Model",
            message: "This feature allows exporting your trained AI model. Please enter the required password to continue.",
            preferredStyle: .alert
        )
        
        // Add text field for password
        alert.addTextField { textField in
            textField.placeholder = "Enter password"
            textField.isSecureTextEntry = true
            textField.clearButtonMode = .whileEditing
        }
        
        // Cancel action
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        
        // Export action
        alert.addAction(UIAlertAction(title: "Export", style: .default) { [weak self, weak alert] _ in
            guard let password = alert?.textFields?.first?.text, !password.isEmpty else {
                self?.showErrorAlert(message: "Password is required")
                return
            }
            
            self?.exportModel(password: password)
        })
        
        present(alert, animated: true)
    }
    
    private func exportModel(password: String) {
        // Show activity indicator
        activityIndicator.startAnimating()
        
        // Perform export
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = AILearningManager.shared.exportModel(password: password)
            
            DispatchQueue.main.async {
                self?.activityIndicator.stopAnimating()
                
                switch result {
                case .success(let exportURL):
                    self?.showExportSuccess(exportURL: exportURL)
                case .failure(let error):
                    self?.showErrorAlert(message: error.localizedDescription)
                }
            }
        }
    }
    
    private func showExportSuccess(exportURL: URL) {
        let alert = UIAlertController(
            title: "Export Successful",
            message: "Model successfully exported to:\n\(exportURL.lastPathComponent)\n\nYou can find this file in the app's documents directory.",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        
        present(alert, animated: true)
    }
    
    private func showErrorAlert(message: String) {
        let alert = UIAlertController(
            title: "Error",
            message: message,
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        
        present(alert, animated: true)
    }
    
    private func showInfoAlert(title: String, message: String) {
        let alert = UIAlertController(
            title: title,
            message: message,
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        
        present(alert, animated: true)
    }
    
    private func trainModelNow() {
        // Show activity indicator
        activityIndicator.startAnimating()
        
        // Request model training
        AILearningManager.shared.trainModelNow { success, message in
            self.activityIndicator.stopAnimating()
            
            // Show result
            let title = success ? "Training Successful" : "Training Failed"
            let alert = UIAlertController(
                title: title,
                message: message,
                preferredStyle: .alert
            )
            
            alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                // Reload status section
                self.tableView.reloadSections(IndexSet(integer: Section.status.rawValue), with: .automatic)
            })
            
            self.present(alert, animated: true)
        }
    }
    
    private func promptClearInteractions() {
        let alert = UIAlertController(
            title: "Clear Stored Interactions",
            message: "This will delete all stored AI interactions and feedback. This action cannot be undone.",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        
        alert.addAction(UIAlertAction(title: "Clear", style: .destructive) { _ in
            // Clear interactions
            AILearningManager.shared.clearAllInteractions()
            
            // Reload status section
            self.tableView.reloadSections(IndexSet(integer: Section.status.rawValue), with: .automatic)
            
            // Show confirmation
            let confirmation = UIAlertController(
                title: "Interactions Cleared",
                message: "All stored AI interactions have been deleted.",
                preferredStyle: .alert
            )
            
            confirmation.addAction(UIAlertAction(title: "OK", style: .default))
            
            self.present(confirmation, animated: true)
        })
        
        present(alert, animated: true)
    }
}

// MARK: - Switch Cell

class SwitchTableViewCell: UITableViewCell {
    
    let switchControl = UISwitch()
    var switchValueChanged: ((Bool) -> Void)?
    
    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        
        switchControl.addTarget(self, action: #selector(switchChanged), for: .valueChanged)
        accessoryView = switchControl
        selectionStyle = .none
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    @objc private func switchChanged() {
        switchValueChanged?(switchControl.isOn)
    }
}
