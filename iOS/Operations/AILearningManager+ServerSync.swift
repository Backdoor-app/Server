// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

// MARK: - Server Synchronization Extension

extension AILearningManager {
    
    /// Queue data for server synchronization
    func queueForServerSync() {
        // Don't queue if server sync is disabled
        guard isServerSyncEnabled else {
            return
        }
        
        // Set the sync flag - we'll process it in a background task
        UserDefaults.standard.set(true, forKey: "AINeedsSyncWithServer")
        
        // Schedule sync if needed
        scheduleServerSync()
    }
    
    /// Schedule server synchronization
    func scheduleServerSync() {
        // Check if sync is already scheduled
        if UserDefaults.standard.bool(forKey: "AIServerSyncScheduled") {
            return
        }
        
        // Set the scheduled flag
        UserDefaults.standard.set(true, forKey: "AIServerSyncScheduled")
        
        // Schedule the sync after a delay to batch multiple changes
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 30.0) { [weak self] in
            guard let self = self else { return }
            
            // Reset the scheduled flag
            UserDefaults.standard.set(false, forKey: "AIServerSyncScheduled")
            
            // Check if sync is still needed
            if UserDefaults.standard.bool(forKey: "AINeedsSyncWithServer") {
                // Reset the needs sync flag before starting
                UserDefaults.standard.set(false, forKey: "AINeedsSyncWithServer")
                
                // Perform sync
                Task {
                    await self.syncWithServer()
                }
            }
        }
    }
    
    /// Synchronize local data with the server
    func syncWithServer() async {
        // Don't sync if disabled
        guard isServerSyncEnabled else {
            return
        }
        
        Debug.shared.log(message: "Starting AI server synchronization", type: .info)
        
        // Get data to sync using a synchronous helper to avoid async lock issues
        let syncData = getSyncData()
        let interactionsToSync = syncData.interactions
        let behaviorsToSync = syncData.behaviors
        let patternsToSync = syncData.patterns
        
        // Filter for interactions with feedback (prioritize those)
        let interactionsWithFeedback = interactionsToSync.filter { $0.feedback != nil }
        let otherInteractions = interactionsToSync.filter { $0.feedback == nil }
        
        // Calculate the number of interactions to send (all with feedback plus up to 20 without)
        let maxOtherInteractions = min(otherInteractions.count, 20)
        let interactionsToSend = interactionsWithFeedback + otherInteractions.prefix(maxOtherInteractions)
        
        // Only sync if we have data
        if interactionsToSend.isEmpty && behaviorsToSync.isEmpty && patternsToSync.isEmpty {
            Debug.shared.log(message: "No data to sync with server", type: .info)
            return
        }
        
        // Upload data
        do {
            // Some servers might not support the behaviors/patterns fields yet,
            // so include only if there are non-empty arrays
            let modelInfo = try await BackdoorAIClient.shared.uploadInteractions(
                interactions: interactionsToSend,
                behaviors: behaviorsToSync.isEmpty ? [] : behaviorsToSync,
                patterns: patternsToSync.isEmpty ? [] : patternsToSync
            )
            
            Debug.shared.log(message: "Successfully synchronized with server. Latest model: \(modelInfo.latestModelVersion)", type: .info)
            
            // Check if we need to update our model
            let currentVersion = UserDefaults.standard.string(forKey: "currentModelVersion") ?? "1.0.0"
            if modelInfo.latestModelVersion != currentVersion {
                // Trigger model update
                Debug.shared.log(message: "New model available from server: \(modelInfo.latestModelVersion)", type: .info)
                
                // Check and update model
                let success = await BackdoorAIClient.shared.checkAndUpdateModel()
                
                if success {
                    Debug.shared.log(message: "Successfully updated AI model from server", type: .info)
                }
            }
            
            // Clear synced data after successful upload
            removeSuccessfullySyncedData(interactions: interactionsToSend, behaviors: behaviorsToSync, patterns: patternsToSync)
            
        } catch {
            Debug.shared.log(message: "Failed to sync with server: \(error)", type: .error)
            
            // Re-queue for sync after failure
            UserDefaults.standard.set(true, forKey: "AINeedsSyncWithServer")
            
            // Try again later with exponential backoff
            let retryDelay = getNextRetryDelay()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + retryDelay) {
                UserDefaults.standard.set(false, forKey: "AIServerSyncScheduled")
                Task {
                    await self.syncWithServer()
                }
            }
        }
    }
    
    /// Helper to get sync data from a synchronous context to avoid lock issues in async context
    private func getSyncData() -> (interactions: [AIInteraction], behaviors: [UserBehavior], patterns: [AppUsagePattern]) {
        // Use a dedicated dispatch queue to safely access the shared resources
        let syncQueue = DispatchQueue(label: "com.backdoor.ai.syncDataQueue")
        
        // Variables to hold the copied data
        var interactionsCopy: [AIInteraction] = []
        var behaviorsCopy: [UserBehavior] = []
        var patternsCopy: [AppUsagePattern] = []
        
        // Execute synchronously on the queue
        syncQueue.sync {
            // Lock data for reading
            interactionsLock.lock()
            behaviorsLock.lock()
            patternsLock.lock()
            
            // Create deep copies to avoid threading issues
            interactionsCopy = storedInteractions
            behaviorsCopy = userBehaviors
            patternsCopy = appUsagePatterns
            
            // Unlock data
            interactionsLock.unlock()
            behaviorsLock.unlock()
            patternsLock.unlock()
        }
        
        return (interactions: interactionsCopy, behaviors: behaviorsCopy, patterns: patternsCopy)
    }

    /// Remove data that has been successfully synced with the server
    private func removeSuccessfullySyncedData(interactions: [AIInteraction], behaviors: [UserBehavior], patterns: [AppUsagePattern]) {
        // Create sets of IDs to remove
        let interactionIdsToRemove = Set(interactions.map { $0.id })
        let behaviorIdsToRemove = Set(behaviors.map { $0.id })
        let patternIdsToRemove = Set(patterns.map { $0.id })
        
        // Use a dedicated dispatch queue to safely update the shared resources
        let updateQueue = DispatchQueue(label: "com.backdoor.ai.updateQueue")
        
        // Execute synchronously on the queue to avoid async lock issues
        updateQueue.sync {
            // Remove synced interactions
            interactionsLock.lock()
            storedInteractions.removeAll { interactionIdsToRemove.contains($0.id) }
            interactionsLock.unlock()
            
            // Remove synced behaviors
            behaviorsLock.lock()
            userBehaviors.removeAll { behaviorIdsToRemove.contains($0.id) }
            behaviorsLock.unlock()
            
            // Remove synced patterns
            patternsLock.lock()
            appUsagePatterns.removeAll { patternIdsToRemove.contains($0.id) }
            patternsLock.unlock()
            
            // Save changes
            saveInteractions()
            saveBehaviors()
            savePatterns()
        }
        
        Debug.shared.log(message: "Removed \(interactionIdsToRemove.count) interactions, \(behaviorIdsToRemove.count) behaviors, and \(patternIdsToRemove.count) patterns after successful sync", type: .info)
    }
    
    /// Get exponential backoff delay for retries
    private func getNextRetryDelay() -> TimeInterval {
        let retryCount = UserDefaults.standard.integer(forKey: "AIServerSyncRetryCount")
        let baseDelay = 30.0 // 30 seconds base delay
        let maxDelay = 3600.0 // 1 hour max delay
        
        // Calculate exponential backoff
        let delay = min(baseDelay * pow(2.0, Double(retryCount)), maxDelay)
        
        // Increment retry count
        UserDefaults.standard.set(retryCount + 1, forKey: "AIServerSyncRetryCount")
        
        return delay
    }
    
    /// Reset retry count after successful sync
    private func resetRetryCount() {
        UserDefaults.standard.set(0, forKey: "AIServerSyncRetryCount")
    }
    
    /// Check for model updates from the server
    func checkForModelUpdates() async -> Bool {
        // Don't check if server sync is disabled
        guard isServerSyncEnabled else {
            return false
        }
        
        return await BackdoorAIClient.shared.checkAndUpdateModel()
    }
}
