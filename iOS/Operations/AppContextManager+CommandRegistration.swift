// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import SwiftUI
import UIKit

// MARK: - Command Registration Extension for AppContextManager

extension AppContextManager {
    /// Registers all available commands with the AI assistant
    func registerAllCommands() {
        // Navigation commands
        registerCommand("navigate to") { [weak self] screen, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.navigateToScreen(screen, completion: completion)
        }

        // Source management commands
        registerCommand("add source") { [weak self] sourceURL, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.addSource(sourceURL, completion: completion)
        }

        registerCommand("list sources") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.listSources(completion: completion)
        }

        registerCommand("refresh sources") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.refreshSources(completion: completion)
        }

        // App and library management
        registerCommand("download app") { [weak self] appInfo, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.downloadApp(appInfo, completion: completion)
        }

        registerCommand("sign app") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.signApp(appName, completion: completion)
        }

        registerCommand("install app") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.installApp(appName, completion: completion)
        }

        registerCommand("list downloaded apps") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.listDownloadedApps(completion: completion)
        }

        registerCommand("list signed apps") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.listSignedApps(completion: completion)
        }

        registerCommand("delete app") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.deleteApp(appName, completion: completion)
        }

        // Certificate management
        registerCommand("list certificates") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.listCertificates(completion: completion)
        }

        registerCommand("select certificate") { [weak self] certName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.selectCertificate(certName, completion: completion)
        }

        registerCommand("import certificate") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.importCertificate(completion: completion)
        }

        // Settings management
        registerCommand("change theme") { [weak self] theme, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.changeTheme(theme, completion: completion)
        }

        registerCommand("toggle setting") { [weak self] setting, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.toggleSetting(setting, completion: completion)
        }

        registerCommand("get app info") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.getAppInfo(appName, completion: completion)
        }

        // Advanced operations
        registerCommand("resign app") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.resignApp(appName, completion: completion)
        }

        registerCommand("add tweak") { [weak self] tweakInfo, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.addTweak(tweakInfo, completion: completion)
        }

        registerCommand("modify bundle id") { [weak self] bundleIdInfo, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.modifyBundleId(bundleIdInfo, completion: completion)
        }

        registerCommand("get app status") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.getAppStatus(completion: completion)
        }

        registerCommand("open app") { [weak self] appName, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.openApp(appName, completion: completion)
        }

        // Help commands
        registerCommand("help") { [weak self] topic, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.provideHelp(topic, completion: completion)
        }

        registerCommand("list commands") { [weak self] _, completion in
            guard let self = self else { completion("System error")
                return
            }
            let commands = self.availableCommands().joined(separator: "\n")
            completion("Available commands:\n\(commands)")
        }
        
        // AI assistant commands
        registerCommand("search") { [weak self] query, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.searchApp(query, completion: completion)
        }
        
        registerCommand("explain") { [weak self] topic, completion in
            guard let self = self else { completion("System error")
                return
            }
            self.explainFeature(topic, completion: completion)
        }
    }
    
    // MARK: - New Command Implementations
    
    /// Search for apps or content in the app
    private func searchApp(_ query: String, completion: @escaping (String) -> Void) {
        // Get downloaded apps
        let downloadedApps = CoreDataManager.shared.getDatedDownloadedApps()
        
        // Filter apps based on query
        let matchingApps = downloadedApps.filter { app in
            guard let name = app.name else { return false }
            return name.lowercased().contains(query.lowercased())
        }
        
        if matchingApps.isEmpty {
            completion("I couldn't find any apps matching '\(query)'. Would you like to search in sources instead?")
        } else {
            let appNames = matchingApps.compactMap { $0.name }.joined(separator: ", ")
            completion("I found the following apps matching '\(query)': \(appNames). Would you like to sign or install any of these?")
        }
    }
    
    /// Explain a feature of the app
    private func explainFeature(_ topic: String, completion: @escaping (String) -> Void) {
        let lowercasedTopic = topic.lowercased()
        
        if lowercasedTopic.contains("sign") || lowercasedTopic.contains("signing") {
            completion("""
            App signing is the process of adding a digital signature to an app so it can be installed on your device.
            
            In Backdoor, you can sign apps by:
            1. Importing a certificate (.p12 file) and provisioning profile (.mobileprovision)
            2. Selecting the app you want to sign from your library
            3. Choosing the certificate to use for signing
            4. Tapping the "Sign" button
            
            The app will then be signed and available in your "Signed Apps" section.
            """)
        } else if lowercasedTopic.contains("certificate") || lowercasedTopic.contains("cert") {
            completion("""
            Certificates in Backdoor are used to sign apps. They consist of:
            
            1. A .p12 file - Contains your private key and certificate
            2. A .mobileprovision file - Contains provisioning information
            
            You can manage your certificates in the Settings tab. To add a new certificate:
            1. Go to Settings > Certificates
            2. Tap the + button
            3. Import your .p12 file and enter its password
            4. Import the matching .mobileprovision file
            
            Once added, you can select this certificate when signing apps.
            """)
        } else if lowercasedTopic.contains("source") || lowercasedTopic.contains("repo") {
            completion("""
            Sources in Backdoor are repositories where you can find apps to download.
            
            To add a new source:
            1. Go to the Sources tab
            2. Tap the + button
            3. Enter the URL of the source
            4. Tap Add
            
            Once added, you can browse and download apps from that source.
            """)
        } else {
            completion("I don't have specific information about '\(topic)'. Would you like to know about app signing, certificates, or sources?")
        }
    }
}
