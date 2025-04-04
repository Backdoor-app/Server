// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import SwiftUI
import UIKit

// MARK: - AI Integration Extension for AppContextManager

extension AppContextManager {
    /// Initialize AI integration - call this at app startup
    func setupAIIntegration() {
        // Register all available commands
        registerAllCommands()

        // Setup observers for context updates
        setupContextObservers()
        
        // Setup CoreML integration
        setupCoreML()
        
        // Register for CoreML model load completion
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(coreMLModelLoaded),
            name: Notification.Name("CoreMLModelLoaded"),
            object: nil
        )

        // Log successful initialization
        Debug.shared.log(message: "Custom AI Assistant integration initialized with \(availableCommands().count) commands", type: .info)
    }
    
    /// Handle CoreML model load completion
    @objc private func coreMLModelLoaded() {
        Debug.shared.log(message: "CoreML model load completed, enhancing AI capabilities", type: .info)
        
        // Update any AI components that depend on the model
        let additionalData: [String: Any] = [
            "mlModelLoaded": true,
            "mlCapabilities": ["intent recognition", "sentiment analysis", "parameter extraction"]
        ]
        setAdditionalContextData(additionalData)
        
        // Notify custom AI service that ML is available
        NotificationCenter.default.post(
            name: Notification.Name("AICapabilitiesEnhanced"),
            object: nil
        )
    }

    /// Setup context observation to keep AI updated with app state
    private func setupContextObservers() {
        // Observe tab changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleTabChange(_:)),
            name: Notification.Name("changeTab"),
            object: nil
        )

        // Observe app state changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appEnteredForeground),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )

        // Observe certificate changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(certificatesUpdated),
            name: Notification.Name("CertificatesUpdated"),
            object: nil
        )

        // Observe library changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(libraryUpdated),
            name: Notification.Name("lfetch"),
            object: nil
        )

        // Add observer for settings changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(settingsUpdated),
            name: UserDefaults.didChangeNotification,
            object: nil
        )
    }

    @objc private func handleTabChange(_ notification: Notification) {
        if let newTab = notification.userInfo?["tab"] as? String {
            var screenName: String
            switch newTab {
                case "home": screenName = "Home"
                case "sources": screenName = "Sources"
                case "library": screenName = "Library"
                case "settings": screenName = "Settings"
                case "bdgHub": screenName = "BDG Hub"
                default: screenName = "Unknown"
            }

            // Update AI context with new screen
            let additionalData: [String: Any] = ["currentScreen": screenName]
            setAdditionalContextData(additionalData)

            Debug.shared.log(message: "AI context updated: Screen changed to \(screenName)", type: .debug)
        }
    }

    @objc private func appEnteredForeground() {
        // Refresh context when app becomes active
        if let topVC = UIApplication.shared.topMostViewController() {
            updateContext(topVC)
            Debug.shared.log(message: "AI context refreshed after app became active", type: .debug)
        }
    }

    @objc private func certificatesUpdated() {
        // Update AI context with new certificate information
        let certificates = CoreDataManager.shared.getDatedCertificate()
        let currentCert = CoreDataManager.shared.getCurrentCertificate()

        let additionalData: [String: Any] = [
            "certificates": certificates.map { $0.certData?.name ?? "Unnamed" },
            "currentCertificate": currentCert?.certData?.name ?? "None",
            "certificateCount": certificates.count,
        ]

        setAdditionalContextData(additionalData)
        Debug.shared.log(message: "AI context updated: Certificate information refreshed", type: .debug)
    }

    @objc private func libraryUpdated() {
        // Update AI context with library information
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let signedApps = CoreDataManager.shared.getDatedSignedApps()

        let additionalData: [String: Any] = [
            "downloadedApps": downloadedApps.map { AppInfo(name: $0.name ?? "Unnamed", version: $0.version ?? "Unknown").description },
            "signedApps": signedApps.map { SignedAppInfo(name: $0.name ?? "Unnamed", bundleIdentifier: $0.bundleidentifier ?? "Unknown", teamName: $0.teamName ?? "N/A").description },
            "downloadedAppCount": downloadedApps.count,
            "signedAppCount": signedApps.count,
        ]

        setAdditionalContextData(additionalData)
        Debug.shared.log(message: "AI context updated: Library information refreshed", type: .debug)
    }

    @objc private func settingsUpdated() {
        // Update AI context with relevant settings changes
        let additionalData: [String: Any] = [
            "appTintColor": Preferences.appTintColor.uiColor.toHexString(),
            "interfaceStyle": UIUserInterfaceStyle(rawValue: Preferences.preferredInterfaceStyle)?.styleName ?? "unspecified",
            "preferredLanguage": Preferences.preferredLanguageCode ?? "system default",
        ]

        setAdditionalContextData(additionalData)
        Debug.shared.log(message: "AI context updated: Settings changes detected", type: .debug)
    }

    /// Process user input through our enhanced natural language understanding
    func processUserInput(_ text: String) -> (intent: String, parameter: String, confidence: Float)? {
        // First try the pattern-based intent recognition for high confidence matches
        if let patternMatch = enhanceContextWithNLU(text) {
            return (patternMatch.intent, patternMatch.parameter, 0.9)
        }

        // Next, try keyword matching with lower confidence
        let lowercasedText = text.lowercased()

        // Keyword sets for different intents
        let signingKeywords = ["sign", "signing", "certificate", "install"]
        let navigationKeywords = ["go to", "navigate", "show me", "open", "screen", "tab"]
        let sourceKeywords = ["source", "repo", "repository", "add"]
        let helpKeywords = ["help", "how to", "tutorial", "guide", "explain"]

        // Check for keyword matches
        if signingKeywords.first(where: { lowercasedText.contains($0) }) != nil {
            return ("app assistance", "signing", 0.7)
        } else if navigationKeywords.first(where: { lowercasedText.contains($0) }) != nil {
            return ("app assistance", "navigation", 0.7)
        } else if sourceKeywords.first(where: { lowercasedText.contains($0) }) != nil {
            return ("app assistance", "sources", 0.7)
        } else if helpKeywords.first(where: { lowercasedText.contains($0) }) != nil {
            return ("general help", "", 0.8)
        }

        // No confident match - return generic conversation intent
        return ("conversation", "", 0.5)
    }

    /// Enhanced context with natural language understanding capabilities
    func enhanceContextWithNLU(_ textInput: String) -> (intent: String, parameter: String)? {
        let lowercasedInput = textInput.lowercased()

        // Check for app signing intent
        if lowercasedInput.matches(pattern: "(?i)sign\\s+(the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$") {
            let appName = textInput.extractMatch(pattern: "(?i)sign\\s+(the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$", groupIndex: 2)
            return ("sign app", appName ?? "")
        }

        // Check for navigation intent
        if lowercasedInput.matches(pattern: "(?i)(?:go\\s+to|navigate\\s+to|open|show)\\s+(?:the\\s+)?(.+?)\\s+(?:tab|screen|page|section)") {
            let screen = textInput.extractMatch(pattern: "(?i)(?:go\\s+to|navigate\\s+to|open|show)\\s+(?:the\\s+)?(.+?)\\s+(?:tab|screen|page|section)", groupIndex: 1)
            return ("navigate to", screen ?? "")
        }

        // Check for source adding intent
        if lowercasedInput.matches(pattern: "(?i)add\\s+(?:a\\s+)?(?:new\\s+)?source\\s+(?:with\\s+url\\s+|at\\s+|from\\s+)?(.+?)\\s*$") {
            let url = textInput.extractMatch(pattern: "(?i)add\\s+(?:a\\s+)?(?:new\\s+)?source\\s+(?:with\\s+url\\s+|at\\s+|from\\s+)?(.+?)\\s*$", groupIndex: 1)
            return ("add source", url ?? "")
        }

        // Check for installing app intent
        if lowercasedInput.matches(pattern: "(?i)install\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$") {
            let appName = textInput.extractMatch(pattern: "(?i)install\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$", groupIndex: 1)
            return ("install app", appName ?? "")
        }

        // Check for opening app intent
        if lowercasedInput.matches(pattern: "(?i)open\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$") {
            let appName = textInput.extractMatch(pattern: "(?i)open\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?(.+?)\\s*$", groupIndex: 1)
            return ("open app", appName ?? "")
        }

        // If no intent matched
        return nil
    }
}

// Helper extensions for string pattern matching
extension String {
    func matches(pattern: String) -> Bool {
        do {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(self.startIndex..., in: self)
            return regex.firstMatch(in: self, range: range) != nil
        } catch {
            return false
        }
    }

    func extractMatch(pattern: String, groupIndex: Int) -> String? {
        do {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(self.startIndex..., in: self)
            if let match = regex.firstMatch(in: self, range: range) {
                if match.numberOfRanges > groupIndex {
                    let group = match.range(at: groupIndex)
                    if group.location != NSNotFound,
                       let groupRange = Range(group, in: self)
                    {
                        return String(self[groupRange])
                    }
                }
            }
            return nil
        } catch {
            return nil
        }
    }
}

// Extension for UIUserInterfaceStyle friendly name
extension UIUserInterfaceStyle {
    var styleName: String {
        switch self {
            case .unspecified: return "unspecified"
            case .light: return "light"
            case .dark: return "dark"
            @unknown default: return "unknown"
        }
    }
}
