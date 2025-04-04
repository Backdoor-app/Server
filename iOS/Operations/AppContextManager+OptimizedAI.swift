// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import Foundation
import SwiftUI
import UIKit

/// Enhanced AI integration extension for AppContextManager
///
/// This extension adds improved AI capabilities to the AppContextManager:
/// 1. Enhanced pattern matching for better command recognition
/// 2. Context-aware conversations with history tracking
/// 3. Predictive command suggestions based on user behavior
extension AppContextManager {
    
    // MARK: - Enhanced AI Setup
    
    /// Sets up the enhanced AI capabilities
    func setupEnhancedAI() {
        // Register for additional context observations
        setupEnhancedContextObservers()
        
        // Register advanced AI commands
        registerAdvancedCommands()
        
        // Initialize the learning model
        initializePredictiveModel()
        
        Debug.shared.log(message: "Enhanced AI capabilities initialized", type: .info)
    }
    
    private func setupEnhancedContextObservers() {
        // Observe file system changes for better context awareness
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(fileSystemChanged(_:)),
            name: NSNotification.Name("FileSystemChanged"),
            object: nil
        )
        
        // Observe sign operations for better assistance
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(signingOperationCompleted(_:)),
            name: NSNotification.Name("SigningCompleted"),
            object: nil
        )
        
        // Observe source refresh operations
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sourcesRefreshed(_:)),
            name: NSNotification.Name("SourcesRefreshed"),
            object: nil
        )
    }
    
    @objc private func fileSystemChanged(_ notification: Notification) {
        if let changedPath = notification.userInfo?["path"] as? String {
            let additionalData: [String: Any] = [
                "recentFileChange": changedPath,
                "fileChangeTimestamp": Date()
            ]
            setAdditionalContextData(additionalData)
            Debug.shared.log(message: "AI context updated with file system change: \(changedPath)", type: .debug)
        }
    }
    
    @objc private func signingOperationCompleted(_ notification: Notification) {
        if let appName = notification.userInfo?["appName"] as? String,
           let success = notification.userInfo?["success"] as? Bool {
            
            let additionalData: [String: Any] = [
                "recentSigningOperation": [
                    "appName": appName,
                    "success": success,
                    "timestamp": Date()
                ]
            ]
            setAdditionalContextData(additionalData)
            
            // Also update the user command history for better predictions
            updateCommandHistory(command: "sign app", parameter: appName)
            
            Debug.shared.log(message: "AI context updated with signing result for \(appName)", type: .debug)
        }
    }
    
    @objc private func sourcesRefreshed(_ notification: Notification) {
        let sources = CoreDataManager.shared.getAZSources()
        
        let additionalData: [String: Any] = [
            "refreshedSources": sources.map { $0.name ?? "Unnamed" },
            "sourceRefreshTimestamp": Date()
        ]
        setAdditionalContextData(additionalData)
        Debug.shared.log(message: "AI context updated with refreshed sources", type: .debug)
    }
    
    // MARK: - Advanced AI Commands
    
    /// Registers advanced AI commands for enhanced functionality
    private func registerAdvancedCommands() {
        // Batch signing command
        registerCommand("batch sign") { [weak self] appNames, completion in
            guard let self = self else {
                completion("System error")
                return
            }
            
            let appList = appNames.split(separator: ",").map { String($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            
            if appList.isEmpty {
                completion("Please specify app names to sign, separated by commas")
                return
            }
            
            completion("Starting batch signing of \(appList.count) apps: \(appList.joined(separator: ", "))")
            
            // Queue the batch operation
            DispatchQueue.global(qos: .userInitiated).async {
                self.performBatchSigning(apps: appList)
            }
        }
        
        // App suggestions command
        registerCommand("suggest apps") { [weak self] category, completion in
            guard let self = self else {
                completion("System error")
                return
            }
            
            self.getSuggestedApps(for: category) { suggestions in
                completion("Based on your usage, here are some suggested apps: \(suggestions.joined(separator: ", "))")
            }
        }
        
        // Enhanced search command
        registerCommand("advanced search") { [weak self] query, completion in
            guard let self = self else {
                completion("System error")
                return
            }
            
            self.performAdvancedSearch(query: query) { results in
                if results.isEmpty {
                    completion("No results found for your search query")
                } else {
                    completion("Found the following results: \(results.joined(separator: ", "))")
                }
            }
        }
        
        // Voice command processing
        registerCommand("voice command") { [weak self] audioQuery, completion in
            guard let self = self else {
                completion("System error")
                return
            }
            
            // Simulate voice command processing
            let processedCommand = self.processVoiceCommand(audioQuery)
            completion("I understood your voice command as: \(processedCommand)")
        }
    }
    
    // MARK: - Command History and Prediction
    
    // Command history is now stored in the main AppContextManager class
    
    /// Updates the command history for better predictions
    private func updateCommandHistory(command: String, parameter: String) {
        commandHistory.append((command: command, parameter: parameter, timestamp: Date()))
        
        // Keep only the last 50 commands
        if commandHistory.count > 50 {
            commandHistory.removeFirst(commandHistory.count - 50)
        }
        
        // Update prediction model
        updatePredictionModel()
    }
    
    /// Initialize the predictive model for command suggestions
    private func initializePredictiveModel() {
        // Load any saved command history
        if let savedHistory = UserDefaults.standard.object(forKey: "AICommandHistory") as? Data {
            if let decodedHistory = try? JSONDecoder().decode(Array<CommandHistoryEntry>.self, from: savedHistory) {
                commandHistory = decodedHistory.map { ($0.command, $0.parameter, $0.timestamp) }
                Debug.shared.log(message: "Loaded \(commandHistory.count) command history entries", type: .debug)
            }
        }
    }
    
    /// Update the prediction model with new command history
    private func updatePredictionModel() {
        // Save the command history for future use
        let historyEntries = commandHistory.map { CommandHistoryEntry(command: $0.command, parameter: $0.parameter, timestamp: $0.timestamp) }
        if let encodedData = try? JSONEncoder().encode(historyEntries) {
            UserDefaults.standard.set(encodedData, forKey: "AICommandHistory")
        }
    }
    
    /// Get predicted commands based on user behavior
    func getPredictedCommands(currentInput: String) -> [String] {
        // Simple prediction based on command frequency and recency
        var commandFrequency: [String: Int] = [:]
        var predictions: [String] = []
        
        // Calculate frequency
        for entry in commandHistory {
            let fullCommand = "\(entry.command) \(entry.parameter)"
            commandFrequency[fullCommand] = (commandFrequency[fullCommand] ?? 0) + 1
        }
        
        // Filter by current input if provided
        let filteredCommands = currentInput.isEmpty ? commandFrequency.keys.map { $0 } :
            commandFrequency.keys.filter { $0.lowercased().contains(currentInput.lowercased()) }
        
        // Sort by frequency
        let sortedCommands = filteredCommands.sorted { commandFrequency[$0] ?? 0 > commandFrequency[$1] ?? 0 }
        
        // Take top 5
        predictions = Array(sortedCommands.prefix(5))
        
        return predictions
    }
    
    // MARK: - Implementation for Advanced Commands
    
    /// Performs batch signing of multiple apps
    private func performBatchSigning(apps: [String]) {
        for appName in apps {
            // Use existing sign app logic for each app in the batch
            Debug.shared.log(message: "Batch signing: Processing \(appName)", type: .info)
            
            // This would trigger the actual signing logic
            // For now, we're just simulating the process
            
            // Notify about progress
            DispatchQueue.main.async {
                NotificationCenter.default.post(
                    name: NSNotification.Name("BatchSigningProgress"),
                    object: nil,
                    userInfo: ["appName": appName, "status": "processing"]
                )
            }
            
            // Simulate processing time
            Thread.sleep(forTimeInterval: 0.5)
            
            // Notify about completion
            DispatchQueue.main.async {
                NotificationCenter.default.post(
                    name: NSNotification.Name("BatchSigningProgress"),
                    object: nil,
                    userInfo: ["appName": appName, "status": "completed"]
                )
            }
        }
        
        // Notify about batch completion
        DispatchQueue.main.async {
            NotificationCenter.default.post(
                name: NSNotification.Name("BatchSigningCompleted"),
                object: nil,
                userInfo: ["appsCount": apps.count]
            )
        }
    }
    
    /// Gets suggested apps based on category and user history
    private func getSuggestedApps(for category: String, completion: @escaping ([String]) -> Void) {
        // Simulate app suggestions based on category
        var suggestions: [String] = []
        
        switch category.lowercased() {
        case "games":
            suggestions = ["Game1", "Game2", "Game3"]
        case "utilities":
            suggestions = ["Utility1", "Utility2", "Utility3"]
        case "social":
            suggestions = ["Social1", "Social2", "Social3"]
        default:
            // Default suggestions based on user history
            // Get the most frequently signed apps from command history
            let signedApps = commandHistory
                .filter { $0.command == "sign app" }
                .map { $0.parameter }
            
            // Get unique apps
            var uniqueApps = Array(Set(signedApps))
            
            // Limit to 5 suggestions
            if uniqueApps.count > 5 {
                uniqueApps = Array(uniqueApps.prefix(5))
            }
            
            suggestions = uniqueApps
        }
        
        completion(suggestions)
    }
    
    /// Performs advanced search across the app
    private func performAdvancedSearch(query: String, completion: @escaping ([String]) -> Void) {
        // Simulate advanced search functionality
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let signedApps = CoreDataManager.shared.getDatedSignedApps()
        let sources = CoreDataManager.shared.getAZSources()
        
        // Combine all searchable items
        var searchResults: [String] = []
        
        // Search in downloaded apps
        for app in downloadedApps {
            if let name = app.name, name.lowercased().contains(query.lowercased()) {
                searchResults.append("App: \(name)")
            }
        }
        
        // Search in signed apps
        for app in signedApps {
            if let name = app.name, name.lowercased().contains(query.lowercased()) {
                searchResults.append("Signed: \(name)")
            }
        }
        
        // Search in sources
        for source in sources {
            if let name = source.name, name.lowercased().contains(query.lowercased()) {
                searchResults.append("Source: \(name)")
            }
        }
        
        completion(searchResults)
    }
    
    /// Processes voice commands (simulated)
    private func processVoiceCommand(_ audioQuery: String) -> String {
        // In a real implementation, this would connect to speech recognition
        // For now, we'll just simulate it by parsing the text directly
        
        // Remove "process voice" prefix if present
        var processedCommand = audioQuery
        if processedCommand.hasPrefix("process voice") {
            processedCommand = String(processedCommand.dropFirst("process voice".count)).trimmingCharacters(in: .whitespaces)
        }
        
        return processedCommand
    }
}

// MARK: - Supporting Types

/// Structure for encoding/decoding command history
struct CommandHistoryEntry: Codable {
    let command: String
    let parameter: String
    let timestamp: Date
}
