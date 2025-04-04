// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import Nuke

class ResetDataClass {
    static let shared = ResetDataClass()

    init() {}
    deinit {}

    public func clearNetworkCache() {
        URLCache.shared.removeAllCachedResponses()
        HTTPCookieStorage.shared.removeCookies(since: Date.distantPast)

        if let dataCache = ImagePipeline.shared.configuration.dataCache as? DataCache {
            dataCache.removeAll()
        }

        if let imageCache = ImagePipeline.shared.configuration.imageCache as? Nuke.ImageCache {
            imageCache.removeAll()
        }
    }

    public func deleteSignedApps() {
        do {
            try CoreDataManager.shared.clearSignedApps()
            self.deleteDirectory(named: "Apps", additionalComponents: ["Signed"])
        } catch {
            Debug.shared.log(message: "Error clearing signed apps: \(error)", type: .error)
        }
    }

    public func deleteDownloadedApps() {
        do {
            try CoreDataManager.shared.clearDownloadedApps()
            self.deleteDirectory(named: "Apps", additionalComponents: ["Unsigned"])
        } catch {
            Debug.shared.log(message: "Error clearing downloaded apps: \(error)", type: .error)
        }
    }

    public func resetCertificates(resetAll: Bool) {
        if !resetAll { Preferences.selectedCert = 0 }
        do {
            try CoreDataManager.shared.clearCertificate()
            self.deleteDirectory(named: "Certificates")
        } catch {
            Debug.shared.log(message: "Error clearing certificates: \(error)", type: .error)
        }
    }

    public func resetSources(resetAll: Bool) {
        if !resetAll { Preferences.defaultRepos = false }
        do {
            try CoreDataManager.shared.clearSources()
        } catch {
            Debug.shared.log(message: "Error clearing sources: \(error)", type: .error)
        }
    }

    private func resetAllUserDefaults() {
        if let bundleID = Bundle.main.bundleIdentifier {
            UserDefaults.standard.removePersistentDomain(forName: bundleID)
        }
    }

    public func resetAll() {
        self.deleteSignedApps()
        self.deleteDownloadedApps()
        self.resetCertificates(resetAll: true)
        self.resetSources(resetAll: true)
        self.resetAllUserDefaults()
        self.clearNetworkCache()
    }

    private func deleteDirectory(named directoryName: String, additionalComponents: [String]? = nil) {
        var directoryURL = getDocumentsDirectory().appendingPathComponent(directoryName)

        if let components = additionalComponents {
            for component in components {
                directoryURL.appendPathComponent(component)
            }
        }

        let fileManager = FileManager.default
        do {
            try fileManager.removeItem(at: directoryURL)
        } catch {
            Debug.shared.log(message: "Couldn't delete this, but thats ok!: \(error)", type: .debug)
        }
    }
}
