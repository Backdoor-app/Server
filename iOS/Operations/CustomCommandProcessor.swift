// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// Processes commands extracted from AI responses
class CustomCommandProcessor {
    // Singleton instance
    static let shared = CustomCommandProcessor()

    // Private initializer for singleton
    private init() {
        Debug.shared.log(message: "CustomCommandProcessor initialized", type: .debug)
    }

    /// Process a command extracted from an AI response
    /// - Parameters:
    ///   - commandString: The full command string containing command and parameter
    ///   - completion: Callback with result information
    func processCommand(_ commandString: String, completion: @escaping (CommandResult) -> Void) {
        // Extract command and parameter
        let components = commandString.split(separator: ":", maxSplits: 1).map(String.init)

        guard components.count >= 1 else {
            completion(.unknownCommand(commandString))
            return
        }

        let command = components[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let parameter = components.count > 1 ? components[1].trimmingCharacters(in: .whitespacesAndNewlines) : ""

        // Execute the command using AppContextManager
        executeCommand(command, parameter: parameter, completion: completion)
    }

    /// Execute a command with the given parameter
    /// - Parameters:
    ///   - command: The command to execute
    ///   - parameter: The parameter for the command
    ///   - completion: Callback with result information
    private func executeCommand(_ command: String, parameter: String, completion: @escaping (CommandResult) -> Void) {
        // Log command execution
        Debug.shared.log(message: "Executing command: \(command) with parameter: \(parameter)", type: .info)

        // Check if the command is registered with AppContextManager
        if AppContextManager.shared.availableCommands().contains(command.lowercased()) {
            // Execute through AppContextManager
            AppContextManager.shared.executeCommand(command, parameter: parameter) { result in
                completion(result)
            }
        } else {
            // Handle special commands not registered with AppContextManager

            switch command.lowercased() {
                case "sign":
                    // Handle app signing request
                    signApp(named: parameter, completion: completion)

                case "install":
                    // Handle app installation request
                    installApp(named: parameter, completion: completion)

                case "open":
                    // Handle app opening request
                    openApp(named: parameter, completion: completion)

                case "help":
                    // Show help information
                    showHelp(topic: parameter, completion: completion)

                default:
                    // Command not recognized
                    completion(.unknownCommand(command))
            }
        }
    }

    // MARK: - Command Implementations

    private func signApp(named appName: String, completion: @escaping (CommandResult) -> Void) {
        // Find app in downloaded apps
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let matchingApps = downloadedApps.filter {
            ($0.name ?? "").lowercased().contains(appName.lowercased())
        }

        if matchingApps.isEmpty {
            completion(.successWithResult("Could not find app '\(appName)' to sign. Please download it first."))
            return
        }

        // For now, just return a confirmation that would normally lead to signing screen
        completion(.successWithResult("Found app '\(matchingApps[0].name ?? appName)'. In a real implementation, this would navigate to the signing screen."))
    }

    private func installApp(named appName: String, completion: @escaping (CommandResult) -> Void) {
        // Simulate app installation
        completion(.successWithResult("Installation command received for '\(appName)'. In a real implementation, this would begin the installation process."))
    }

    private func openApp(named appName: String, completion: @escaping (CommandResult) -> Void) {
        // Find app in signed apps
        let signedApps = CoreDataManager.shared.getDatedSignedApps()
        let matchingApps = signedApps.filter {
            ($0.name ?? "").lowercased().contains(appName.lowercased())
        }

        if matchingApps.isEmpty {
            completion(.successWithResult("Could not find signed app '\(appName)' to open. Please sign it first."))
            return
        }

        // Simulate app opening
        completion(.successWithResult("Opening app '\(matchingApps[0].name ?? appName)'. In a real implementation, this would launch the app."))
    }

    private func showHelp(topic: String, completion: @escaping (CommandResult) -> Void) {
        var helpText = "Backdoor AI Assistant Help"

        if topic.isEmpty {
            // General help
            helpText = """
            Backdoor AI Assistant Help:

            Available commands:
            - [sign:app_name] - Sign an app with your certificate
            - [install:app_name] - Install an app from your sources
            - [open:app_name] - Open an installed app
            - [navigate to:screen_name] - Navigate to a specific screen
            - [add source:url] - Add a new source repository
            - [list sources] - Show available sources
            - [list downloaded apps] - Show downloaded apps
            - [list signed apps] - Show signed apps
            - [list certificates] - Show available certificates
            - [help:topic] - Show help on a specific topic

            Type 'help:signing' for information about signing apps.
            """
        } else if topic.lowercased() == "signing" {
            // Help about signing
            helpText = """
            App Signing Help:

            To sign an app:
            1. Make sure you have a valid certificate in Settings > Certificates
            2. Download the app you want to sign
            3. Use the command [sign:app_name] or go to Library tab
            4. Select the app and tap "Sign"

            You can view your signed apps with [list signed apps].
            """
        }

        completion(.successWithResult(helpText))
    }
}
