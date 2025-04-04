// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

final class CoreDataManager {
    static let shared = CoreDataManager()
    private var _context: NSManagedObjectContext?
    private var initializationError: Error?

    private init() {
        setupCoreData()
    }

    deinit {}

    private func setupCoreData() {
        do {
            try initializePersistentContainer()
            Debug.shared.log(message: "Core Data initialized successfully", type: .info)
        } catch {
            Debug.shared.log(message: "Failed to initialize Core Data: \(error.localizedDescription)", type: .error)
            initializationError = error
            // We don't crash here - we'll handle errors gracefully when context is accessed
        }
    }

    private func initializePersistentContainer() throws {
        // First try to find the model at the standard location
        let container = NSPersistentContainer(name: "Backdoor")

        // Use a semaphore to make this synchronous but not deadlock
        let semaphore = DispatchSemaphore(value: 0)
        var loadError: Error?

        container.loadPersistentStores { _, error in
            loadError = error
            semaphore.signal()
        }

        // Wait for the stores to load with a timeout
        let timeoutResult = semaphore.wait(timeout: .now() + 5)

        if timeoutResult == .timedOut {
            throw NSError(domain: "CoreDataManager", code: 1001, userInfo: [NSLocalizedDescriptionKey: "Timed out loading persistent stores"])
        }

        if let error = loadError {
            throw error
        }

        // Set the context if everything succeeded
        _context = container.viewContext
        _context?.automaticallyMergesChangesFromParent = true
    }

    var context: NSManagedObjectContext {
        get throws {
            if let context = _context {
                return context
            }

            if let error = initializationError {
                throw error
            }

            // If we get here, something unexpected happened
            let error = NSError(domain: "CoreDataManager", code: 1000, userInfo: [NSLocalizedDescriptionKey: "Core Data context unavailable"])
            Debug.shared.log(message: "Core Data context requested but unavailable: \(error.localizedDescription)", type: .error)
            throw error
        }
    }

    func saveContext() throws {
        do {
            let ctx = try context
            guard ctx.hasChanges else { return }
            try ctx.save()
        } catch {
            Debug.shared.log(message: "CoreDataManager.saveContext error: \(error.localizedDescription)", type: .error)
            throw error
        }
    }

    /// Clear all objects from fetch request.
    func clear<T: NSManagedObject>(request: NSFetchRequest<T>, context: NSManagedObjectContext? = nil) throws {
        do {
            let ctx = try context ?? self.context
            let deleteRequest = NSBatchDeleteRequest(fetchRequest: (request as? NSFetchRequest<NSFetchRequestResult>)!)
            _ = try ctx.execute(deleteRequest)
        } catch {
            Debug.shared.log(message: "CoreDataManager.clear error: \(error.localizedDescription)", type: .error)
            throw error
        }
    }

    func loadImage(from iconUrl: URL?) -> UIImage? {
        guard let iconUrl = iconUrl else { return nil }
        return UIImage(contentsOfFile: iconUrl.path)
    }

    // MARK: - Chat Session Management

    func createChatSession(title: String) throws -> ChatSession {
        do {
            let ctx = try context
            let session = ChatSession(context: ctx)
            session.sessionID = UUID().uuidString
            session.title = title
            session.creationDate = Date()
            try saveContext()
            return session
        } catch {
            Debug.shared.log(message: "CoreDataManager.createChatSession: Failed to save session - \(error.localizedDescription)", type: .error)
            throw error
        }
    }

    func addMessage(to session: ChatSession, sender: String, content: String) throws -> ChatMessage {
        do {
            let ctx = try context
            let message = ChatMessage(context: ctx)
            message.messageID = UUID().uuidString
            message.sender = sender
            message.content = content
            message.timestamp = Date()
            message.session = session
            try saveContext()
            return message
        } catch {
            Debug.shared.log(message: "CoreDataManager.addMessage: Failed to save message - \(error.localizedDescription)", type: .error)
            throw error
        }
    }

    func getMessages(for session: ChatSession) -> [ChatMessage] {
        do {
            let ctx = try context
            let request: NSFetchRequest<ChatMessage> = ChatMessage.fetchRequest()
            request.predicate = NSPredicate(format: "session == %@", session)
            request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: true)]
            return try ctx.fetch(request)
        } catch {
            Debug.shared.log(message: "CoreDataManager.getMessages: \(error.localizedDescription)", type: .error)
            return []
        }
    }

    func getChatSessions() -> [ChatSession] {
        do {
            let ctx = try context
            let request: NSFetchRequest<ChatSession> = ChatSession.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            return try ctx.fetch(request)
        } catch {
            Debug.shared.log(message: "CoreDataManager.getChatSessions: \(error.localizedDescription)", type: .error)
            return []
        }
    }

    func fetchChatHistory(for session: ChatSession) -> [ChatMessage] {
        do {
            let ctx = try context
            let request: NSFetchRequest<ChatMessage> = ChatMessage.fetchRequest()
            request.predicate = NSPredicate(format: "session == %@", session)
            request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: true)]
            let messages = try ctx.fetch(request)
            Debug.shared.log(message: "Fetched chat history for session: \(session.title ?? "Unnamed") with \(messages.count) messages", type: .info)
            return messages
        } catch {
            Debug.shared.log(message: "CoreDataManager.fetchChatHistory: \(error.localizedDescription)", type: .error)
            return []
        }
    }

    func getDatedCertificate() -> [Certificate] {
        do {
            let ctx = try context
            let request: NSFetchRequest<Certificate> = Certificate.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            return try ctx.fetch(request)
        } catch {
            Debug.shared.log(message: "CoreDataManager.getDatedCertificate: \(error.localizedDescription)", type: .error)
            return []
        }
    }

    func getCurrentCertificate() -> Certificate? {
        let certificates = getDatedCertificate()
        let selectedIndex = Preferences.selectedCert // This is already a non-optional Int with default value 0
        guard selectedIndex >= 0, selectedIndex < certificates.count else { return nil }
        return certificates[selectedIndex]
    }

    func getFilesForDownloadedApps(for app: DownloadedApps, getuuidonly: Bool) throws -> URL {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let uuid = app.uuid ?? UUID().uuidString
        let url = getuuidonly ? documentsDirectory.appendingPathComponent(uuid) : documentsDirectory.appendingPathComponent("files/\(uuid)")

        // Ensure the directory exists if not getting UUID only
        if !getuuidonly {
            let fileManager = FileManager.default
            if !fileManager.fileExists(atPath: url.path) {
                do {
                    try fileManager.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
                } catch {
                    Debug.shared.log(message: "CoreDataManager.getFilesForDownloadedApps: Failed to create directory - \(error.localizedDescription)", type: .error)
                    throw error
                }
            }
        }
        return url
    }
}

// Error type for background task operations
struct BackgroundTaskError: Error {
    let underlyingError: Error
}

extension NSPersistentContainer {
    // Use regular throws instead of rethrows and explicitly specify error handling
    func performBackgroundTask<T>(_ block: @escaping (NSManagedObjectContext) throws -> T) async throws -> T {
        do {
            return try await withCheckedThrowingContinuation { continuation in
                self.performBackgroundTask { context in
                    do {
                        let result = try block(context)
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: BackgroundTaskError(underlyingError: error))
                    }
                }
            }
        } catch {
            // Ensure we throw a properly typed error
            throw error
        }
    }
}
