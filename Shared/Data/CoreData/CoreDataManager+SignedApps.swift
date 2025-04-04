// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import Foundation

extension CoreDataManager {
    /// Clear all signedapps from Core Data and delete files
    func clearSignedApps(context: NSManagedObjectContext? = nil) throws {
        let ctx = try context ?? self.context
        try clear(request: SignedApps.fetchRequest(), context: ctx)
    }

    /// Fetch all sources sorted alphabetically by name
    func getDatedSignedApps(context: NSManagedObjectContext? = nil) -> [SignedApps] {
        let request: NSFetchRequest<SignedApps> = SignedApps.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "dateAdded", ascending: false)]
        do {
            let ctx = try context ?? self.context
            return try ctx.fetch(request)
        } catch {
            Debug.shared.log(message: "Error in getDatedSignedApps: \(error)", type: .error)
            return []
        }
    }

    /// Add application to downloaded apps
    func addToSignedApps(
        context: NSManagedObjectContext? = nil,
        version: String,
        name: String,
        bundleidentifier: String,
        iconURL: String?,
        dateAdded: Date? = Date(),
        uuid: String,
        appPath: String?,
        timeToLive: Date,
        teamName: String,
        originalSourceURL: URL?,
        completion: @escaping (Result<SignedApps, Error>) -> Void
    ) {
        do {
            let ctx = try context ?? self.context
            let newApp = SignedApps(context: ctx)

            newApp.version = version
            newApp.name = name
            newApp.bundleidentifier = bundleidentifier
            newApp.iconURL = iconURL
            newApp.dateAdded = dateAdded
            newApp.uuid = uuid
            newApp.appPath = appPath
            newApp.timeToLive = timeToLive
            newApp.teamName = teamName
            newApp.originalSourceURL = originalSourceURL

            try ctx.save()
            NotificationCenter.default.post(name: Notification.Name("lfetch"), object: nil)
            completion(.success(newApp))
        } catch {
            Debug.shared.log(message: "Error saving data: \(error)", type: .error)
            completion(.failure(error))
        }
    }

    /// Get application file path (non-throwing version for compatibility)
    @available(*, deprecated, message: "Use the throwing version getFilesForSignedApps(for:getuuidonly:) instead")
    func getFilesForSignedApps(for app: SignedApps, getuuidonly: Bool = false) -> URL {
        do {
            // Directly implement the path construction to avoid circular references
            guard let uuid = app.uuid, let appPath = app.appPath, let dir = app.directory else {
                Debug.shared.log(message: "Missing required app properties (uuid, appPath, or directory)", type: .error)
                return URL(fileURLWithPath: "")
            }

            guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                Debug.shared.log(message: "Could not access documents directory", type: .error)
                return URL(fileURLWithPath: "")
            }

            var path = documentsDirectory
                .appendingPathComponent("Apps")
                .appendingPathComponent(dir)
                .appendingPathComponent(uuid)

            if !getuuidonly {
                path = path.appendingPathComponent(appPath)

                // Ensure directory exists
                let directoryPath = path.deletingLastPathComponent()
                if !FileManager.default.fileExists(atPath: directoryPath.path) {
                    try FileManager.default.createDirectory(at: directoryPath, withIntermediateDirectories: true)
                }
            }

            return path
        } catch {
            Debug.shared.log(message: "Error in getFilesForSignedApps: \(error)", type: .error)
            return URL(fileURLWithPath: "")
        }
    }

    /// Get application file path with proper error handling
    func getFilesForSignedAppsWithThrow(for app: SignedApps, getuuidonly: Bool = false) throws -> URL {
        guard let uuid = app.uuid, let appPath = app.appPath, let dir = app.directory else {
            throw FileProcessingError.missingFile("Required app properties (uuid, appPath, or directory)")
        }

        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw FileProcessingError.missingFile("Documents directory")
        }

        var path = documentsDirectory
            .appendingPathComponent("Apps")
            .appendingPathComponent(dir)
            .appendingPathComponent(uuid)

        if !getuuidonly {
            path = path.appendingPathComponent(appPath)

            // Ensure directory exists
            let directoryPath = path.deletingLastPathComponent()
            if !FileManager.default.fileExists(atPath: directoryPath.path) {
                do {
                    try FileManager.default.createDirectory(at: directoryPath, withIntermediateDirectories: true)
                } catch {
                    throw FileProcessingError.fileIOError(error)
                }
            }
        }

        return path
    }

    /// Delete a signed app (non-throwing version for compatibility)
    func deleteAllSignedAppContent(for app: SignedApps) {
        do {
            try deleteAllSignedAppContentWithThrow(for: app)
        } catch {
            Debug.shared.log(message: "CoreDataManager.deleteAllSignedAppContent: \(error)", type: .error)
        }
    }

    /// Delete a signed app with proper error handling
    func deleteAllSignedAppContentWithThrow(for app: SignedApps) throws {
        let ctx = try context
        ctx.delete(app)
        let fileURL = try getFilesForSignedAppsWithThrow(for: app, getuuidonly: true)
        try FileManager.default.removeItem(at: fileURL)
        try ctx.save()
    }

    func updateSignedApp(
        app: SignedApps,
        newTimeToLive: Date,
        newTeamName: String,
        completion: @escaping (Error?) -> Void
    ) {
        do {
            // Properly handle the optional and throwing parts separately
            let context: NSManagedObjectContext
            if let appContext = app.managedObjectContext {
                context = appContext
            } else {
                context = try self.context
            }

            app.timeToLive = newTimeToLive
            app.teamName = newTeamName

            try context.save()
            completion(nil)
        } catch {
            Debug.shared.log(message: "Error updating SignedApps: \(error)", type: .error)
            completion(error)
        }
    }

    func setUpdateAvailable(for app: SignedApps, newVersion: String) throws {
        app.hasUpdate = true
        app.updateVersion = newVersion
        try saveContext()
    }

    func clearUpdateState(for app: SignedApps) throws {
        app.hasUpdate = false
        app.updateVersion = nil
        try saveContext()
    }

    // Non-throwing versions for backward compatibility
    func setUpdateAvailableCompat(for app: SignedApps, newVersion: String) {
        do {
            try setUpdateAvailable(for: app, newVersion: newVersion)
        } catch {
            Debug.shared.log(message: "Error in setUpdateAvailable: \(error)", type: .error)
        }
    }

    func clearUpdateStateCompat(for app: SignedApps) {
        do {
            try clearUpdateState(for: app)
        } catch {
            Debug.shared.log(message: "Error in clearUpdateState: \(error)", type: .error)
        }
    }
}
