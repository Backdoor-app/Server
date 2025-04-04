// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import UIKit

/// Class responsible for providing rich context to the custom AI system
class CustomAIContextProvider {
    // Singleton instance
    static let shared = CustomAIContextProvider()

    // Cache for storing computed context information
    private var contextCache: [String: Any] = [:]
    private var lastContextUpdateTime: Date = .init()

    // Refresh interval for context cache (5 minutes)
    private let cacheRefreshInterval: TimeInterval = 300

    private init() {
        // Register for app state notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(clearCache),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
    }

    @objc private func clearCache() {
        contextCache.removeAll()
        lastContextUpdateTime = Date()
    }

    /// Get comprehensive context information about the app state
    func getContextInformation() -> [String: Any] {
        // Check if we need to refresh the cache
        let now = Date()
        if now.timeIntervalSince(lastContextUpdateTime) > cacheRefreshInterval {
            updateContextCache()
        }

        return contextCache
    }

    /// Force refresh of the context cache
    func refreshContext() {
        updateContextCache()
    }

    /// Update the context cache with fresh data
    private func updateContextCache() {
        // Start with basic app information
        var context: [String: Any] = [
            "appVersion": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown",
            "buildNumber": Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown",
            "deviceModel": UIDevice.current.model,
            "systemVersion": UIDevice.current.systemVersion,
            "interfaceStyle": UITraitCollection.current.userInterfaceStyle == .dark ? "dark" : "light",
            "timestamp": ISO8601DateFormatter().string(from: Date()),
        ]

        // Add preference information
        context["preferences"] = [
            "tintColor": Preferences.appTintColor.uiColor.toHexString(),
            "interfaceStyle": Preferences.preferredInterfaceStyle,
            "language": Preferences.preferredLanguageCode,
        ]

        // Add certificate information
        let certificates = CoreDataManager.shared.getDatedCertificate()
        context["certificates"] = [
            "count": certificates.count,
            "names": certificates.map { $0.certData?.name ?? "Unnamed" },
            "currentCertificate": CoreDataManager.shared.getCurrentCertificate()?.certData?.name ?? "None",
        ]

        // Add library information
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let signedApps = CoreDataManager.shared.getDatedSignedApps()
        context["library"] = [
            "downloadedApps": [
                "count": downloadedApps.count,
                "names": downloadedApps.map { $0.name ?? "Unnamed" },
            ],
            "signedApps": [
                "count": signedApps.count,
                "names": signedApps.map { $0.name ?? "Unnamed" },
            ],
        ]

        // Add sources information
        let sources = CoreDataManager.shared.getAZSources()
        context["sources"] = [
            "count": sources.count,
            "names": sources.map { $0.name ?? "Unnamed" },
        ]

        // Add current screen information if available
        if let topVC = UIApplication.shared.topMostViewController() {
            context["currentScreen"] = String(describing: type(of: topVC))
        }

        // Add any additional context data from the AppContextManager
        if let contextManager = AppContextManager.shared as? AppContextManager {
            context["additionalContext"] = contextManager.currentContext().toString()
        }

        // Update the cache
        contextCache = context
        lastContextUpdateTime = Date()
    }

    /// Get a natural language summary of the current app state
    func getContextSummary() -> String {
        let context = getContextInformation()

        // Create a user-friendly summary
        var summary = "You're using Backdoor version \(context["appVersion"] ?? "Unknown")."

        // Add certificate info
        if let certificates = context["certificates"] as? [String: Any],
           let count = certificates["count"] as? Int
        {
            if count > 0 {
                summary += " You have \(count) certificate(s) available."
                if let currentCert = certificates["currentCertificate"] as? String, currentCert != "None" {
                    summary += " Currently using '\(currentCert)'."
                }
            } else {
                summary += " You don't have any certificates set up yet."
            }
        }

        // Add library info
        if let library = context["library"] as? [String: Any],
           let downloadedApps = library["downloadedApps"] as? [String: Any],
           let downloadedCount = downloadedApps["count"] as? Int,
           let signedApps = library["signedApps"] as? [String: Any],
           let signedCount = signedApps["count"] as? Int
        {
            summary += " Your library has \(downloadedCount) downloaded app(s) and \(signedCount) signed app(s)."
        }

        // Add current screen
        if let currentScreen = context["currentScreen"] as? String {
            summary += " You're currently on the \(currentScreen.replacingOccurrences(of: "ViewController", with: "")) screen."
        }

        return summary
    }
}
