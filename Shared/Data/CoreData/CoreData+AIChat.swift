// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import Foundation

// MARK: - Deprecated Core Data Stack

// Use CoreDataManager.shared instead
@available(*, deprecated, message: "Use CoreDataManager.shared instead")
class AIChatCoreDataManager {
    static let shared = CoreDataManager.shared

    private init() {}

    // MARK: - Save Context (Deprecated)

    @available(*, deprecated, message: "Use CoreDataManager.shared.saveContext() instead")
    func saveContext() {
        do {
            try CoreDataManager.shared.saveContext()
        } catch {
            Debug.shared.log(message: "AIChatCoreDataManager.saveContext: \(error)", type: .error)
        }
    }
}

// MARK: - Chat functionality moved to CoreDataManager

extension CoreDataManager {
    // Create new chat session
    func createAIChatSession(title: String? = nil) throws -> ChatSession {
        let ctx = try context
        let chatSession = ChatSession(context: ctx)

        chatSession.sessionID = UUID().uuidString
        chatSession.title = title
        chatSession.creationDate = Date()

        try saveContext()
        return chatSession
    }

    // Fetch all chat sessions
    func fetchChatSessions() -> [ChatSession] {
        let fetchRequest: NSFetchRequest<ChatSession> = ChatSession.fetchRequest()
        fetchRequest.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]

        do {
            let ctx = try context
            return try ctx.fetch(fetchRequest)
        } catch {
            Debug.shared.log(message: "Failed to fetch chat sessions: \(error)", type: .error)
            return []
        }
    }

    // Fetch specific chat session by ID
    func fetchChatSession(sessionID: String) -> ChatSession? {
        let fetchRequest: NSFetchRequest<ChatSession> = ChatSession.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "sessionID == %@", sessionID)
        fetchRequest.fetchLimit = 1

        do {
            let ctx = try context
            return try ctx.fetch(fetchRequest).first
        } catch {
            Debug.shared.log(message: "Failed to fetch chat session: \(error)", type: .error)
            return nil
        }
    }

    // Delete chat session
    func deleteChatSession(_ session: ChatSession) throws {
        let ctx = try context
        ctx.delete(session)
        try saveContext()
    }

    // Add AI message (legacy method)
    func addAIMessage(to session: ChatSession, content: String, isUser: Bool) throws -> ChatMessage {
        return try addMessage(to: session, sender: isUser ? "user" : "ai", content: content)
    }

    // Delete message by ID
    func deleteMessage(withID messageID: String) {
        let fetchRequest: NSFetchRequest<ChatMessage> = ChatMessage.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "messageID == %@", messageID)
        fetchRequest.fetchLimit = 1

        do {
            let ctx = try context
            if let message = try ctx.fetch(fetchRequest).first {
                ctx.delete(message)
                try saveContext()
            }
        } catch {
            Debug.shared.log(message: "Failed to delete message: \(error)", type: .error)
        }
    }
}
