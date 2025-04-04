// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import SwiftUI
import UIKit

// MARK: - Command Implementation Extension for AppContextManager

extension AppContextManager {
    // MARK: - Navigation

    /// Navigate to a specific screen in the app
    func navigateToScreen(_ screen: String, completion: @escaping (String) -> Void) {
        // We don't strictly need to check if we're on the tab bar view
        // Just attempt to navigate and provide appropriate feedback
        guard UIApplication.shared.topMostViewController() != nil else {
            Debug.shared.log(message: "Cannot navigate: No visible view controller", type: .error)
            completion("Cannot navigate: No visible view controller")
            return
        }

        var targetTab: String
        switch screen.lowercased() {
            case "home", "main":
                targetTab = "home"
            case "sources", "repos", "repositories":
                targetTab = "sources"
            case "library", "apps", "applications":
                targetTab = "library"
            case "settings", "preferences", "options":
                targetTab = "settings"
            case "bdg hub", "bdghub", "hub", "web":
                targetTab = "bdgHub"
            default:
                Debug.shared.log(message: "Unknown screen: \(screen)", type: .warning)
                completion("Unknown screen: \(screen)")
                return
        }

        UserDefaults.standard.set(targetTab, forKey: "selectedTab")
        NotificationCenter.default.post(name: .changeTab, object: nil, userInfo: ["tab": targetTab])
        completion("Navigated to \(screen)")
    }

    // MARK: - Source Management

    /// Add a source to the app
    func addSource(_ sourceURL: String, completion: @escaping (String) -> Void) {
        guard URL(string: sourceURL) != nil else {
            Debug.shared.log(message: "Invalid source URL: \(sourceURL)", type: .error)
            completion("Invalid source URL format. Please provide a valid URL.")
            return
        }

        CoreDataManager.shared.saveSource(name: "AI Added Source", id: UUID().uuidString, iconURL: nil, url: sourceURL) { error in
            if let error = error {
                Debug.shared.log(message: "Failed to add source: \(error)", type: .error)
                completion("Failed to add source: \(error.localizedDescription)")
            } else {
                Debug.shared.log(message: "Added source: \(sourceURL)", type: .success)
                completion("Source added successfully. You can find it in the Sources tab.")
            }
        }
    }

    /// List all sources in the app
    func listSources(completion: @escaping (String) -> Void) {
        let sources = CoreDataManager.shared.getAZSources()
        if sources.isEmpty {
            completion("No sources available. You can add sources using the [add source:url] command.")
            return
        }

        let sourceList = sources.enumerated().map { index, source -> String in
            let name = source.name ?? "Unnamed Source"
            let url = source.sourceURL?.absoluteString ?? "No URL"
            return "\(index + 1). \(name) - \(url)"
        }.joined(separator: "\n")

        completion("Sources:\n\(sourceList)")
    }

    /// Refresh all sources
    func refreshSources(completion: @escaping (String) -> Void) {
        let backgroundQueue = OperationQueue()
        backgroundQueue.qualityOfService = .background
        let operation = SourceRefreshOperation()

        operation.completionBlock = {
            DispatchQueue.main.async {
                if operation.isCancelled {
                    completion("Source refresh was cancelled")
                } else {
                    completion("Sources have been refreshed successfully")
                }
            }
        }

        backgroundQueue.addOperation(operation)
        completion("Refreshing sources in the background...")
    }

    // MARK: - App Management

    /// Download an app from a source
    func downloadApp(_: String, completion: @escaping (String) -> Void) {
        // This is a simplified implementation - would need to be expanded based on app structure
        completion("App download functionality requires user interaction. Please navigate to the Sources tab and select the app you want to download.")
    }

    /// Sign an app
    func signApp(_ appName: String, completion: @escaping (String) -> Void) {
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()

        guard !downloadedApps.isEmpty else {
            completion("No downloaded apps found. Please download an app first.")
            return
        }

        // Find the app by name
        let matchingApps = downloadedApps.filter {
            ($0.name?.lowercased() ?? "").contains(appName.lowercased())
        }

        guard !matchingApps.isEmpty else {
            completion("No downloaded app found with name: \(appName). Please check the app name and try again.")
            return
        }

        if matchingApps.count > 1 {
            let appList = matchingApps.map { $0.name ?? "Unnamed" }.joined(separator: ", ")
            completion("Multiple matching apps found: \(appList). Please be more specific.")
            return
        }

        // Found exactly one app
        let app = matchingApps[0]

        // Trigger app signing through notification center
        DispatchQueue.main.async {
            NotificationCenter.default.post(
                name: Notification.Name("InstallDownloadedApp"),
                object: nil,
                userInfo: ["downloadedApp": app]
            )
            completion("Started signing process for \(app.name ?? "the app"). Please follow the on-screen instructions.")
        }
    }

    /// Install a signed app
    func installApp(_ appName: String, completion: @escaping (String) -> Void) {
        let signedApps = CoreDataManager.shared.getDatedSignedApps()

        guard !signedApps.isEmpty else {
            completion("No signed apps found. Please sign an app first.")
            return
        }

        // Find the app by name
        let matchingApps = signedApps.filter {
            ($0.name?.lowercased() ?? "").contains(appName.lowercased())
        }

        guard !matchingApps.isEmpty else {
            completion("No signed app found with name: \(appName). Please check the app name and try again.")
            return
        }

        if matchingApps.count > 1 {
            let appList = matchingApps.map { $0.name ?? "Unnamed" }.joined(separator: ", ")
            completion("Multiple matching apps found: \(appList). Please be more specific.")
            return
        }

        // This is a placeholder - the actual installation would require access to the LibraryViewController
        completion("Installation requires user interaction. Please go to the Library tab, find the app, and select 'Install'.")
    }

    /// List all downloaded apps
    func listDownloadedApps(completion: @escaping (String) -> Void) {
        let apps = CoreDataManager.shared.getDatedDownloadedApps()

        if apps.isEmpty {
            completion("No downloaded apps found.")
            return
        }

        let appsList = apps.enumerated().map { index, app -> String in
            let name = app.name ?? "Unnamed App"
            let version = app.version ?? "Unknown version"
            return "\(index + 1). \(name) (v\(version))"
        }.joined(separator: "\n")

        completion("Downloaded apps:\n\(appsList)")
    }

    /// List all signed apps
    func listSignedApps(completion: @escaping (String) -> Void) {
        let apps = CoreDataManager.shared.getDatedSignedApps()

        if apps.isEmpty {
            completion("No signed apps found.")
            return
        }

        let appsList = apps.enumerated().map { index, app -> String in
            let name = app.name ?? "Unnamed App"
            let bundleId = app.bundleidentifier ?? "Unknown bundle ID"
            let expiryDate = app.timeToLive ?? Date()

            let dateFormatter = DateFormatter()
            dateFormatter.dateStyle = .medium

            return "\(index + 1). \(name) (\(bundleId)) - Expires: \(dateFormatter.string(from: expiryDate))"
        }.joined(separator: "\n")

        completion("Signed apps:\n\(appsList)")
    }

    /// Delete an app (downloaded or signed)
    func deleteApp(_: String, completion: @escaping (String) -> Void) {
        // This is a placeholder - the actual implementation would need to be expanded
        completion("App deletion requires user interaction. Please go to the Library tab, swipe left on the app, and select 'Delete'.")
    }

    // MARK: - Certificate Management

    /// List all certificates
    func listCertificates(completion: @escaping (String) -> Void) {
        let certificates = CoreDataManager.shared.getDatedCertificate()

        if certificates.isEmpty {
            completion("No certificates found. You need to import a certificate before signing apps.")
            return
        }

        let currentCert = CoreDataManager.shared.getCurrentCertificate()

        let certList = certificates.enumerated().map { index, cert -> String in
            let name = cert.certData?.name ?? "Unnamed Certificate"
            let expiryDate = cert.certData?.expirationDate ?? Date()
            let isSelected = (cert == currentCert) ? " (Selected)" : ""

            let dateFormatter = DateFormatter()
            dateFormatter.dateStyle = .medium

            return "\(index + 1). \(name)\(isSelected) - Expires: \(dateFormatter.string(from: expiryDate))"
        }.joined(separator: "\n")

        completion("Certificates:\n\(certList)")
    }

    /// Select a certificate for signing
    func selectCertificate(_ certName: String, completion: @escaping (String) -> Void) {
        let certificates = CoreDataManager.shared.getDatedCertificate()

        guard !certificates.isEmpty else {
            completion("No certificates found. You need to import a certificate first.")
            return
        }

        // Find certificate by name
        let matchingCerts = certificates.filter {
            ($0.certData?.name?.lowercased() ?? "").contains(certName.lowercased())
        }

        guard !matchingCerts.isEmpty else {
            completion("No certificate found with name: \(certName). Please check the certificate name and try again.")
            return
        }

        if matchingCerts.count > 1 {
            let certList = matchingCerts.map { $0.certData?.name ?? "Unnamed" }.joined(separator: ", ")
            completion("Multiple matching certificates found: \(certList). Please be more specific.")
            return
        }

        // Found exactly one certificate
        if let index = certificates.firstIndex(of: matchingCerts[0]) {
            Preferences.selectedCert = index
            completion("Selected certificate: \(matchingCerts[0].certData?.name ?? "Unnamed Certificate")")
        } else {
            completion("Error selecting certificate. Please try again.")
        }
    }

    /// Import a certificate
    func importCertificate(completion: @escaping (String) -> Void) {
        completion("Certificate import requires user interaction. Please go to Settings > Certificates > Import Certificate.")
    }

    // MARK: - Settings Management

    /// Change app theme
    func changeTheme(_ theme: String, completion: @escaping (String) -> Void) {
        let themeMode: Int
        switch theme.lowercased() {
            case "light", "day":
                themeMode = UIUserInterfaceStyle.light.rawValue
            case "dark", "night":
                themeMode = UIUserInterfaceStyle.dark.rawValue
            case "system", "auto", "automatic":
                themeMode = UIUserInterfaceStyle.unspecified.rawValue
            default:
                completion("Unknown theme: \(theme). Available options are: light, dark, system")
                return
        }

        Preferences.preferredInterfaceStyle = themeMode

        // Apply theme to current window
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first
        {
            window.overrideUserInterfaceStyle = UIUserInterfaceStyle(rawValue: themeMode) ?? .unspecified
        }

        completion("Theme changed to \(theme)")
    }

    /// Toggle app settings
    func toggleSetting(_ setting: String, completion: @escaping (String) -> Void) {
        // Simple implementation for a few common settings - would need to be expanded
        switch setting.lowercased() {
            case "app updates", "updates":
                Preferences.appUpdates.toggle()
                let newState = Preferences.appUpdates ? "enabled" : "disabled"
                completion("App updates \(newState)")

            case "install after signing", "auto install":
                var options = UserDefaults.standard.signingOptions
                options.installAfterSigned.toggle()
                UserDefaults.standard.signingOptions = options
                let newState = options.installAfterSigned ? "enabled" : "disabled"
                completion("Install after signing \(newState)")

            default:
                completion("Unknown setting: \(setting). Please specify a valid setting name.")
        }
    }

    /// Get detailed information about an app
    func getAppInfo(_ appName: String, completion: @escaping (String) -> Void) {
        // Check downloaded apps
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        let matchingDownloaded = downloadedApps.filter {
            ($0.name?.lowercased() ?? "").contains(appName.lowercased())
        }

        // Check signed apps
        let signedApps = CoreDataManager.shared.getDatedSignedApps()
        let matchingSignedApps = signedApps.filter {
            ($0.name?.lowercased() ?? "").contains(appName.lowercased())
        }

        if matchingDownloaded.isEmpty, matchingSignedApps.isEmpty {
            completion("No app found with name: \(appName)")
            return
        }

        var info = ""

        // Process downloaded apps
        if !matchingDownloaded.isEmpty {
            if matchingDownloaded.count > 1 {
                let names = matchingDownloaded.map { $0.name ?? "Unnamed" }.joined(separator: ", ")
                info += "Multiple downloaded apps match: \(names)\n\n"
            }

            // Show info for the first match
            let app = matchingDownloaded[0]
            info += "Downloaded App Info:\n"
            info += "- Name: \(app.name ?? "Unnamed")\n"
            info += "- Version: \(app.version ?? "Unknown")\n"
            info += "- Bundle ID: \(app.bundleidentifier ?? "Unknown")\n"

            if !matchingSignedApps.isEmpty {
                info += "\n"
            }
        }

        // Process signed apps
        if !matchingSignedApps.isEmpty {
            if matchingSignedApps.count > 1 {
                let names = matchingSignedApps.map { $0.name ?? "Unnamed" }.joined(separator: ", ")
                info += "Multiple signed apps match: \(names)\n\n"
            }

            // Show info for the first match
            let app = matchingSignedApps[0]
            info += "Signed App Info:\n"
            info += "- Name: \(app.name ?? "Unnamed")\n"
            info += "- Bundle ID: \(app.bundleidentifier ?? "Unknown")\n"

            if let expiryDate = app.timeToLive {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                info += "- Expires: \(dateFormatter.string(from: expiryDate))\n"
            }

            info += "- Team Name: \(app.teamName ?? "Unknown")\n"

            // Check if app has update available
            let hasUpdateAvailable = app.hasUpdate == true
            if hasUpdateAvailable {
                info += "- Update Available: YES\n"
                if let updateVersion = app.updateVersion {
                    info += "- Update Version: \(updateVersion)\n"
                }
            } else {
                info += "- Update Available: NO\n"
            }
        }

        completion(info)
    }

    // MARK: - Advanced Operations

    /// Resign an app with the current certificate
    func resignApp(_: String, completion: @escaping (String) -> Void) {
        // This is a placeholder - would need to be implemented with access to the app's UI
        completion("App re-signing requires user interaction. Please go to the Library tab, select the app, and choose 'Resign'.")
    }

    /// Add a tweak to an app
    func addTweak(_: String, completion: @escaping (String) -> Void) {
        // This is a placeholder - would need to be implemented with access to the app's UI
        completion("Adding tweaks requires user interaction. Please start the signing process and select 'Add Tweaks'.")
    }

    /// Modify an app's bundle ID
    func modifyBundleId(_: String, completion: @escaping (String) -> Void) {
        // This is a placeholder - would need to be implemented with access to the app's UI
        completion("Modifying bundle ID requires user interaction. Please start the signing process and edit the Bundle Identifier field.")
    }

    /// Get current app status
    func getAppStatus(completion: @escaping (String) -> Void) {
        let downloadedCount = CoreDataManager.shared.getDatedDownloadedApps().count
        let signedCount = CoreDataManager.shared.getDatedSignedApps().count
        let certificateCount = CoreDataManager.shared.getDatedCertificate().count
        let currentCert = CoreDataManager.shared.getCurrentCertificate()?.certData?.name ?? "None"

        var statusInfo = "Backdoor App Status:\n"
        statusInfo += "- Downloaded Apps: \(downloadedCount)\n"
        statusInfo += "- Signed Apps: \(signedCount)\n"
        statusInfo += "- Certificates: \(certificateCount)\n"
        statusInfo += "- Current Certificate: \(currentCert)\n"
        statusInfo += "- Auto-Install After Signing: \(UserDefaults.standard.signingOptions.installAfterSigned ? "Enabled" : "Disabled")\n"
        statusInfo += "- App Updates: \(Preferences.appUpdates ? "Enabled" : "Disabled")\n"

        // Add current screen info
        let currentScreen = self.currentState?.currentScreen ?? "Unknown"
        statusInfo += "- Current Screen: \(currentScreen)\n"

        completion(statusInfo)
    }

    /// Open an installed app
    func openApp(_ appName: String, completion: @escaping (String) -> Void) {
        let signedApps = CoreDataManager.shared.getDatedSignedApps()

        // Find the app by name
        let matchingApps = signedApps.filter {
            ($0.name?.lowercased() ?? "").contains(appName.lowercased())
        }

        guard !matchingApps.isEmpty else {
            completion("No app found with name: \(appName)")
            return
        }

        if matchingApps.count > 1 {
            let appList = matchingApps.map { $0.name ?? "Unnamed" }.joined(separator: ", ")
            completion("Multiple matching apps found: \(appList). Please be more specific.")
            return
        }

        // Found exactly one app
        guard let bundleId = matchingApps[0].bundleidentifier, !bundleId.isEmpty else {
            completion("Cannot open app: Missing bundle identifier")
            return
        }

        if let workspace = LSApplicationWorkspace.default() {
            let success = workspace.openApplication(withBundleID: bundleId)
            if success {
                completion("Opening \(matchingApps[0].name ?? "the app")...")
            } else {
                completion("Unable to open \(matchingApps[0].name ?? "the app"). It may not be installed or might have a different bundle identifier.")
            }
        } else {
            completion("Failed to access application workspace. Cannot open app.")
        }
    }

    // MARK: - Help Functions

    /// Provide help information
    func provideHelp(_ topic: String, completion: @escaping (String) -> Void) {
        switch topic.lowercased() {
            case "":
                let generalHelp = """
                Backdoor AI Assistant Help:

                I can help you with app signing, management, and settings. Here are some common topics:

                - signing: Get help with app signing
                - certificates: Information about certificate management
                - sources: Help with adding and managing sources
                - commands: List all available commands
                - apps: Help with managing apps

                Type [help:topic] for specific help on any topic.
                """
                completion(generalHelp)

            case "signing", "sign":
                let signingHelp = """
                App Signing Help:

                To sign an app:
                1. First, download the app from a source
                2. Make sure you have a valid certificate
                3. Use [sign app:app_name] to start the signing process
                4. Follow on-screen instructions to complete signing

                You can customize signing options during the process, including:
                - Bundle ID
                - App name
                - Adding tweaks
                - Modifying dylibs

                After signing, you can install the app using the installation option.
                """
                completion(signingHelp)

            case "certificates", "certificate", "cert":
                let certHelp = """
                Certificate Management Help:

                Certificates are required for signing apps. You can:

                - View certificates with [list certificates]
                - Select a certificate with [select certificate:name]
                - Import a new certificate by going to Settings > Certificates

                A valid certificate is required before you can sign or install apps.
                """
                completion(certHelp)

            case "sources", "source", "repo", "repos":
                let sourceHelp = """
                Sources Help:

                Sources provide apps that you can download and sign. You can:

                - View sources with [list sources]
                - Add a new source with [add source:url]
                - Refresh sources with [refresh sources]

                To browse apps from sources, go to the Sources tab.
                """
                completion(sourceHelp)

            case "commands", "command":
                let commandList = availableCommands().joined(separator: "\n")
                let commandHelp = """
                Available Commands:

                \(commandList)

                Use commands in the format [command:parameter]
                """
                completion(commandHelp)

            case "apps", "app":
                let appHelp = """
                App Management Help:

                You can manage your apps with these functions:

                - View downloaded apps with [list downloaded apps]
                - View signed apps with [list signed apps]
                - Sign an app with [sign app:app_name]
                - Get app details with [get app info:app_name]
                - Open an installed app with [open app:app_name]

                For more app management, visit the Library tab.
                """
                completion(appHelp)

            default:
                completion("No help available for '\(topic)'. Try [help] for general assistance.")
        }
    }
}
