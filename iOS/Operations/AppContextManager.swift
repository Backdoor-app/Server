// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import SwiftUI
import UIKit

/// Protocol for view controllers to explicitly identify their screen name.
protocol ScreenIdentifiable {
    var screenName: String { get }
}

/// Manages the app's context and command execution for the AI assistant.
final class AppContextManager {
    // Singleton instance
    static let shared = AppContextManager()

    // Properties
    var currentState: AppContext? // Refers to AppContext from AppContext.swift
    private var commandHandlers: [String: (String, @escaping (String) -> Void) -> Void] = [:]
    private let commandQueue = DispatchQueue(label: "com.app.commandQueue")
    
    // Command history for AI prediction
    var commandHistory: [(command: String, parameter: String, timestamp: Date)] = []

    private init() {}

    /// Updates the app's context based on the current view controller.
    func updateContext(_ viewController: UIViewController) {
        var additionalData: [String: Any] = [:]

        // Fetch data from CoreDataManager for app state
        let sources = CoreDataManager.shared.getAZSources()
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let signedApps = CoreDataManager.shared.getDatedSignedApps()
        let certificates = CoreDataManager.shared.getDatedCertificate()
        let currentCertificate = CoreDataManager.shared.getCurrentCertificate()

        // Populate additional data with formatted information
        additionalData["sources"] = sources.map { $0.name ?? "Unnamed" }
        additionalData["downloadedApps"] = downloadedApps.map { AppInfo(name: $0.name ?? "Unnamed", version: $0.version ?? "Unknown").description }
        additionalData["signedApps"] = signedApps.map { SignedAppInfo(name: $0.name ?? "Unnamed", bundleIdentifier: $0.bundleidentifier ?? "Unknown", teamName: $0.teamName ?? "N/A").description }
        additionalData["certificates"] = certificates.map { $0.certData?.name ?? "Unnamed" }
        additionalData["currentCertificate"] = currentCertificate?.certData?.name ?? "None"
        additionalData["signingOptions"] = Preferences.signingOptions.description

        // Determine the current screen
        if let identifiable = viewController as? ScreenIdentifiable {
            additionalData["currentScreen"] = identifiable.screenName
        } else {
            switch viewController {
                case _ as UIHostingController<TabbarView>:
                    let selectedTab = UserDefaults.standard.string(forKey: "selectedTab") ?? "home"
                    additionalData["currentTab"] = selectedTab
                    switch selectedTab {
                        case "home": additionalData["currentScreen"] = "Home"
                        case "sources": additionalData["currentScreen"] = "Sources"
                        case "library": additionalData["currentScreen"] = "Library"
                        case "settings": additionalData["currentScreen"] = "Settings"
                        case "bdgHub": additionalData["currentScreen"] = "BDG Hub"
                        default: additionalData["currentScreen"] = "Unknown"
                    }
                default:
                    let screenName = String(describing: type(of: viewController))
                    if screenName.contains("Home") {
                        additionalData["currentScreen"] = "Home"
                    } else if screenName.contains("Sources") {
                        additionalData["currentScreen"] = "Sources"
                    } else if screenName.contains("Library") {
                        additionalData["currentScreen"] = "Library"
                    } else if screenName.contains("Settings") {
                        additionalData["currentScreen"] = "Settings"
                    } else if screenName.contains("WebView") || screenName.contains("BDGHub") {
                        additionalData["currentScreen"] = "BDG Hub"
                    } else {
                        additionalData["currentScreen"] = "Unknown"
                    }
            }
        }

        // Add chat-specific context if in a chat view
        if let chatVC = viewController as? ChatViewController {
            additionalData["currentChatSession"] = chatVC.currentSession.title
            // Example: Add chat history summary (assuming CoreDataManager provides this)
            let chatHistory = CoreDataManager.shared.fetchChatHistory(for: chatVC.currentSession)
            additionalData["chatHistorySummary"] = chatHistory.map { "\($0.timestamp ?? Date()): \($0.content ?? "")" }
        }

        // Update the current state
        currentState = AppContext(
            currentScreen: additionalData["currentScreen"] as? String ?? "Unknown",
            additionalData: additionalData
        )
    }

    /// Returns the current app context.
    func currentContext() -> AppContext {
        return currentState ?? AppContext(currentScreen: "Unknown", additionalData: [:])
    }

    /// Registers a command with its handler, including a completion callback.
    func registerCommand(_ command: String, handler: @escaping (String, @escaping (String) -> Void) -> Void) {
        commandQueue.sync {
            commandHandlers[command.lowercased()] = handler
        }
    }

    /// Executes a command with the given parameter and returns the result via completion.
    func executeCommand(_ command: String, parameter: String, completion: @escaping (CommandResult) -> Void) {
        commandQueue.sync {
            let commandKey = command.lowercased()
            if let handler = commandHandlers[commandKey] {
                handler(parameter) { result in
                    completion(.successWithResult(result))
                }
            } else {
                Debug.shared.log(message: "Unknown command: \(command)", type: .warning)
                completion(.unknownCommand(command))
            }
        }
    }

    /// Returns a list of available commands.
    func availableCommands() -> [String] {
        commandQueue.sync {
            Array(commandHandlers.keys)
        }
    }

    /// Sets additional context data dynamically.
    func setAdditionalContextData(_ data: [String: Any]) {
        // Since AppContext is immutable, create a new instance
        if let current = currentState {
            currentState = AppContext(
                currentScreen: current.currentScreen,
                additionalData: current.additionalData.merging(data) { _, new in new }
            )
        }
    }
}

/// Result type for command execution.
enum CommandResult {
    case successWithResult(String)
    case unknownCommand(String)
}

/// Struct to format app information.
struct AppInfo {
    let name: String
    let version: String
    var description: String { "\(name) (\(version))" }
}

/// Struct to format signed app information.
struct SignedAppInfo {
    let name: String
    let bundleIdentifier: String
    let teamName: String
    var description: String { "\(name) (\(bundleIdentifier)) - Team: \(teamName)" }
}
