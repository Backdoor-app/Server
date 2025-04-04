// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Combine
import CoreData
import Foundation
import UIKit

/// A comprehensive manager for app lifecycle and state preservation
///
/// This class improves app state management during lifecycle events:
/// 1. Robust state preservation across app backgrounding/foregrounding
/// 2. Enhanced background task management
/// 3. Crash recovery and session restoration
/// 4. Coordinated view state preservation
final class AppLifecycleManager {
    // MARK: - Singleton
    
    /// Shared instance
    static let shared = AppLifecycleManager()
    
    // MARK: - Properties
    
    /// Current app state
    private(set) var appState: AppState = .inactive {
        didSet {
            stateDidChange(from: oldValue, to: appState)
        }
    }
    
    /// State for view controllers
    private var viewStates: [String: Any] = [:]
    
    /// Background tasks
    private var backgroundTasks: [UIBackgroundTaskIdentifier: BackgroundTaskInfo] = [:]
    
    /// Cancellables for Combine subscriptions
    private var cancellables = Set<AnyCancellable>()
    
    /// Processing queue for background operations
    private let processingQueue = DispatchQueue(label: "com.backdoor.AppLifecycleManager.Processing", qos: .utility)
    
    /// Crash recovery timestamp
    private var lastSaveTimestamp: Date?
    
    // MARK: - Initialization
    
    private init() {
        Debug.shared.log(message: "AppLifecycleManager initializing", type: .info)
        setupObservers()
        loadPersistedState()
    }
    
    // MARK: - Setup
    
    /// Sets up observation of app lifecycle events
    private func setupObservers() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationWillTerminate),
            name: UIApplication.willTerminateNotification,
            object: nil
        )
        
        // Register for memory warnings to proactively save state
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(didReceiveMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
        
        // Tab change notifications for preserving tab-specific state
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(tabDidChange),
            name: Notification.Name("tabDidChange"),
            object: nil
        )
        
        Debug.shared.log(message: "AppLifecycleManager observers set up", type: .debug)
    }
    
    // MARK: - App Lifecycle Handlers
    
    @objc private func applicationWillResignActive() {
        appState = .inactive
        Debug.shared.log(message: "App will resign active", type: .info)
        
        // Save state immediately when app becomes inactive
        saveApplicationState()
        
        // Notify components to prepare for inactivity
        NotificationCenter.default.post(name: .appWillBecomeInactive, object: nil)
    }
    
    @objc private func applicationDidEnterBackground() {
        appState = .background
        Debug.shared.log(message: "App did enter background", type: .info)
        
        // Additional state saving for background mode
        saveApplicationState()
        
        // Start background tasks that need to continue
        startBackgroundTasks()
        
        // Notify components about background state
        NotificationCenter.default.post(name: .appDidEnterBackground, object: nil)
    }
    
    @objc private func applicationWillEnterForeground() {
        appState = .foreground
        Debug.shared.log(message: "App will enter foreground", type: .info)
        
        // Prepare to restore state
        prepareForForeground()
        
        // Notify components about foreground state
        NotificationCenter.default.post(name: .appWillEnterForeground, object: nil)
    }
    
    @objc private func applicationDidBecomeActive() {
        appState = .active
        Debug.shared.log(message: "App did become active", type: .info)
        
        // Complete state restoration
        completeStateRestoration()
        
        // Verify Core Data state is clean
        verifyDataIntegrity()
        
        // Notify components about active state
        NotificationCenter.default.post(name: .appDidBecomeActive, object: nil)
    }
    
    @objc private func applicationWillTerminate() {
        Debug.shared.log(message: "App will terminate", type: .info)
        
        // Final save before termination
        saveApplicationState(isTerminating: true)
        
        // Cancel any background tasks
        cancelAllBackgroundTasks()
        
        // Notify components about termination
        NotificationCenter.default.post(name: .appWillTerminate, object: nil)
    }
    
    @objc private func didReceiveMemoryWarning() {
        Debug.shared.log(message: "Memory warning received", type: .warning)
        
        // Save critical state
        saveApplicationState()
        
        // Cancel non-essential background tasks
        cancelNonEssentialBackgroundTasks()
        
        // Notify components to reduce memory usage
        NotificationCenter.default.post(name: .appDidReceiveMemoryWarning, object: nil)
    }
    
    @objc private func tabDidChange(_ notification: Notification) {
        if let tab = notification.userInfo?["tab"] as? String {
            Debug.shared.log(message: "Tab changed to: \(tab)", type: .debug)
            
            // Save state of previous tab before switching
            saveTabState()
            
            // Restore state for the new tab
            restoreTabState(for: tab)
        }
    }
    
    // MARK: - State Change Handling
    
    /// Called when the app state changes
    private func stateDidChange(from oldState: AppState, to newState: AppState) {
        Debug.shared.log(message: "App state changed from \(oldState) to \(newState)", type: .debug)
        
        switch (oldState, newState) {
        case (.inactive, .background):
            // Transition from inactive to background
            prepareBackgroundTasks()
            
        case (.background, .foreground):
            // Transition from background to foreground
            cancelDeferrableBackgroundTasks()
            
        case (.foreground, .active):
            // Transition from foreground to active
            completeUIRestoration()
            
        case (.active, .inactive):
            // Transition from active to inactive
            prepareForInactive()
            
        default:
            // Other state transitions
            break
        }
    }
    
    // MARK: - State Persistence
    
    /// Saves the current application state
    func saveApplicationState(isTerminating: Bool = false) {
        Debug.shared.log(message: "Saving application state (terminating: \(isTerminating))", type: .debug)
        
        // Capture timestamp of save
        let timestamp = Date()
        self.lastSaveTimestamp = timestamp
        
        // Create state container
        var state: [String: Any] = [
            "timestamp": timestamp,
            "appVersion": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown",
            "buildNumber": Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown",
            "selectedTab": UserDefaults.standard.string(forKey: "selectedTab") ?? "home"
        ]
        
        // Save view controller states
        state["viewStates"] = viewStates
        
        // Save user preferences
        state["preferences"] = [
            "interfaceStyle": Preferences.preferredInterfaceStyle,
            "language": Preferences.preferredLanguageCode ?? "system"
        ]
        
        // Get active operations
        let activeOperations = getActiveOperations()
        if !activeOperations.isEmpty {
            state["activeOperations"] = activeOperations
        }
        
        // Encode and save state
        if let encodedState = try? JSONSerialization.data(withJSONObject: state, options: []) {
            UserDefaults.standard.set(encodedState, forKey: "AppStateData")
            UserDefaults.standard.synchronize()
        }
        
        // Save Core Data context if needed
        saveDataContext(isTerminating: isTerminating)
        
        // Notify about state save completion
        NotificationCenter.default.post(name: .appStateDidSave, object: nil)
    }
    
    /// Loads the persisted application state
    private func loadPersistedState() {
        Debug.shared.log(message: "Loading persisted application state", type: .debug)
        
        // Check for crash recovery
        checkForCrashRecovery()
        
        // Load saved state
        if let stateData = UserDefaults.standard.data(forKey: "AppStateData"),
           let state = try? JSONSerialization.jsonObject(with: stateData, options: []) as? [String: Any] {
            
            // Extract timestamp
            if let timestamp = state["timestamp"] as? Date {
                lastSaveTimestamp = timestamp
                Debug.shared.log(message: "Last state saved at: \(timestamp)", type: .debug)
            }
            
            // Recover view states
            if let savedViewStates = state["viewStates"] as? [String: Any] {
                viewStates = savedViewStates
            }
            
            // Check for active operations to resume
            if let operations = state["activeOperations"] as? [[String: Any]] {
                processingQueue.async {
                    self.resumeOperations(operations)
                }
            }
        }
    }
    
    /// Saves the Core Data context
    private func saveDataContext(isTerminating: Bool) {
        do {
            try CoreDataManager.shared.saveContext()
            Debug.shared.log(message: "Core Data context saved successfully", type: .debug)
        } catch {
            Debug.shared.log(message: "Failed to save Core Data context: \(error.localizedDescription)", type: .error)
        }
    }
    
    // MARK: - State Restoration
    
    /// Prepares for app returning to foreground
    private func prepareForForeground() {
        Debug.shared.log(message: "Preparing for foreground", type: .debug)
        
        // Load any updated state
        loadPersistedState()
        
        // Prepare UI for restoration
        prepareUIForRestoration()
        
        // Notify components to prepare for foreground
        NotificationCenter.default.post(name: .appPreparingForForeground, object: nil)
    }
    
    /// Completes state restoration after app becomes active
    private func completeStateRestoration() {
        Debug.shared.log(message: "Completing state restoration", type: .debug)
        
        // Complete restoration of active tab
        if let selectedTab = UserDefaults.standard.string(forKey: "selectedTab") {
            restoreTabState(for: selectedTab)
        }
        
        // Notify components about completed restoration
        NotificationCenter.default.post(name: .appStateRestorationCompleted, object: nil)
    }
    
    /// Prepares UI for restoration
    private func prepareUIForRestoration() {
        DispatchQueue.main.async {
            // Ensure root view controller is ready
            if let rootVC = UIApplication.shared.windows.first?.rootViewController {
                rootVC.view.isUserInteractionEnabled = true
                
                // Apply theme
                self.applyAppTheme(to: rootVC)
            }
        }
    }
    
    /// Completes UI restoration
    private func completeUIRestoration() {
        DispatchQueue.main.async {
            // Find the root view controller
            if let rootVC = UIApplication.shared.windows.first?.rootViewController {
                // Refresh the entire view hierarchy
                self.refreshViewHierarchy(rootVC)
            }
        }
    }
    
    // MARK: - View State Management
    
    /// Saves the state of the current tab
    private func saveTabState() {
        if let selectedTab = UserDefaults.standard.string(forKey: "selectedTab") {
            // Find the active view controller for this tab
            if let rootVC = UIApplication.shared.windows.first?.rootViewController,
               let topVC = UIApplication.shared.topMostViewController() {
                
                // Check if view controller supports state saving
                if let stateSavable = topVC as? StateSavable {
                    let state = stateSavable.saveState()
                    viewStates["\(selectedTab)_viewState"] = state
                    Debug.shared.log(message: "Saved state for tab: \(selectedTab)", type: .debug)
                }
            }
        }
    }
    
    /// Restores the state for a specific tab
    private func restoreTabState(for tab: String) {
        // Check if we have saved state for this tab
        if let tabState = viewStates["\(tab)_viewState"] {
            // Find the active view controller for this tab
            if let rootVC = UIApplication.shared.windows.first?.rootViewController {
                // Post notification with the state to restore
                NotificationCenter.default.post(
                    name: .restoreTabState,
                    object: nil,
                    userInfo: ["tab": tab, "state": tabState]
                )
                Debug.shared.log(message: "Restored state for tab: \(tab)", type: .debug)
            }
        }
    }
    
    /// Helper method to refresh the entire view hierarchy
    private func refreshViewHierarchy(_ viewController: UIViewController) {
        // Make the view controller interactive
        viewController.view.isUserInteractionEnabled = true
        
        // Force layout update
        viewController.view.setNeedsLayout()
        viewController.view.layoutIfNeeded()
        
        // If the view controller supports refresh, refresh it
        if let refreshable = viewController as? ViewControllerRefreshable {
            refreshable.refreshContent()
        }
        
        // Recursively refresh child view controllers
        for child in viewController.children {
            refreshViewHierarchy(child)
        }
        
        // Refresh presented view controller if any
        if let presented = viewController.presentedViewController {
            refreshViewHierarchy(presented)
        }
    }
    
    /// Applies theme to view controller hierarchy
    private func applyAppTheme(to viewController: UIViewController) {
        // Apply tint color
        viewController.view.tintColor = Preferences.appTintColor.uiColor
        
        // Apply interface style
        viewController.overrideUserInterfaceStyle = UIUserInterfaceStyle(rawValue: Preferences.preferredInterfaceStyle) ?? .unspecified
        
        // Apply to child view controllers
        for child in viewController.children {
            applyAppTheme(to: child)
        }
        
        // Apply to presented view controller if any
        if let presented = viewController.presentedViewController {
            applyAppTheme(to: presented)
        }
    }
    
    // MARK: - Background Tasks
    
    /// Prepares background tasks when app is about to enter background
    private func prepareBackgroundTasks() {
        Debug.shared.log(message: "Preparing background tasks", type: .debug)
        
        // Make sure Core Data is saved
        saveDataContext(isTerminating: false)
    }
    
    /// Starts background tasks that need to continue in background
    private func startBackgroundTasks() {
        // Start monitoring task for app state
        startStateMonitoringTask()
        
        // Start tasks for any in-progress operations
        startTasksForActiveOperations()
    }
    
    /// Starts a background task for monitoring app state
    private func startStateMonitoringTask() {
        var taskId: UIBackgroundTaskIdentifier = .invalid
        
        // Use captured local variable for task ID to avoid capture-reference cycle
        taskId = UIApplication.shared.beginBackgroundTask(withName: "StateMonitoring") { [weak self, taskId] in
            // Use weak self to avoid reference cycle
            guard let self = self else { return }
            
            // Save state before expiration
            self.saveApplicationState()
            
            // End the task using the captured taskId
            if taskId != .invalid {
                UIApplication.shared.endBackgroundTask(taskId)
            }
        }
        
        if taskId != .invalid {
            let taskInfo = BackgroundTaskInfo(
                id: taskId,
                name: "StateMonitoring",
                startTime: Date(),
                isPriority: true
            )
            backgroundTasks[taskId] = taskInfo
            Debug.shared.log(message: "Started state monitoring background task", type: .debug)
        }
    }
    
    /// Starts background tasks for active operations
    private func startTasksForActiveOperations() {
        let activeOps = getActiveOperations()
        
        for operation in activeOps {
            if let name = operation["name"] as? String {
                var taskId: UIBackgroundTaskIdentifier = .invalid
                
                taskId = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self, taskId] in
                    guard let self = self else { return }
                    
                    // Save operation state before expiration
                    self.saveOperationState(for: name)
                    
                    // End the task using local captured taskId
                    if taskId != .invalid {
                        UIApplication.shared.endBackgroundTask(taskId)
                        self.backgroundTasks.removeValue(forKey: taskId)
                    }
                }
                
                if taskId != .invalid {
                    let isPriority = operation["isPriority"] as? Bool ?? false
                    let taskInfo = BackgroundTaskInfo(
                        id: taskId,
                        name: name,
                        startTime: Date(),
                        isPriority: isPriority
                    )
                    backgroundTasks[taskId] = taskInfo
                    Debug.shared.log(message: "Started background task for operation: \(name)", type: .debug)
                }
            }
        }
    }
    
    /// Cancels background tasks that can be deferred
    private func cancelDeferrableBackgroundTasks() {
        for (taskId, taskInfo) in backgroundTasks {
            if !taskInfo.isPriority {
                UIApplication.shared.endBackgroundTask(taskId)
                backgroundTasks.removeValue(forKey: taskId)
                Debug.shared.log(message: "Canceled deferrable background task: \(taskInfo.name)", type: .debug)
            }
        }
    }
    
    /// Cancels non-essential background tasks (during memory pressure)
    private func cancelNonEssentialBackgroundTasks() {
        for (taskId, taskInfo) in backgroundTasks {
            if !taskInfo.isPriority {
                UIApplication.shared.endBackgroundTask(taskId)
                backgroundTasks.removeValue(forKey: taskId)
                Debug.shared.log(message: "Canceled non-essential background task due to memory pressure: \(taskInfo.name)", type: .debug)
            }
        }
    }
    
    /// Cancels all background tasks
    private func cancelAllBackgroundTasks() {
        for (taskId, taskInfo) in backgroundTasks {
            UIApplication.shared.endBackgroundTask(taskId)
            Debug.shared.log(message: "Canceled background task: \(taskInfo.name)", type: .debug)
        }
        backgroundTasks.removeAll()
    }
    
    // MARK: - Operations Management
    
    /// Gets information about active operations
    private func getActiveOperations() -> [[String: Any]] {
        // This would be populated with real data from various managers
        // For now, we'll return an empty array
        return []
    }
    
    /// Saves state for a specific operation
    private func saveOperationState(for operationName: String) {
        // Implementation would save specific operation state
        Debug.shared.log(message: "Saving state for operation: \(operationName)", type: .debug)
    }
    
    /// Resumes operations that were active when app went to background
    private func resumeOperations(_ operations: [[String: Any]]) {
        for operation in operations {
            if let name = operation["name"] as? String {
                // Implementation would resume specific operations
                Debug.shared.log(message: "Resuming operation: \(name)", type: .debug)
                
                // Post notification to resume specific operation
                NotificationCenter.default.post(
                    name: .resumeOperation,
                    object: nil,
                    userInfo: ["operation": operation]
                )
            }
        }
    }
    
    // MARK: - Crash Recovery
    
    /// Checks for signs of a previous crash and attempts recovery
    private func checkForCrashRecovery() {
        let lastSessionEndedCleanly = UserDefaults.standard.bool(forKey: "SessionEndedCleanly")
        UserDefaults.standard.set(false, forKey: "SessionEndedCleanly")
        
        if !lastSessionEndedCleanly {
            Debug.shared.log(message: "Detected possible crash in previous session, attempting recovery", type: .warning)
            attemptCrashRecovery()
        } else {
            Debug.shared.log(message: "Previous session ended cleanly", type: .debug)
        }
    }
    
    /// Attempts to recover from a crash
    private func attemptCrashRecovery() {
        // Check Core Data integrity
        do {
            try CoreDataManager.shared.verifyStoreConsistency()
            Debug.shared.log(message: "Core Data store is consistent after crash", type: .info)
        } catch {
            Debug.shared.log(message: "Core Data inconsistency detected, attempting repair: \(error.localizedDescription)", type: .error)
            CoreDataManager.shared.attemptStoreRecovery()
        }
        
        // Notify about crash recovery
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            NotificationCenter.default.post(name: .appRecoveredFromCrash, object: nil)
        }
    }
    
    /// Verifies data integrity when app becomes active
    private func verifyDataIntegrity() {
        processingQueue.async {
            do {
                try CoreDataManager.shared.verifyStoreConsistency()
                Debug.shared.log(message: "Data integrity verified", type: .debug)
            } catch {
                Debug.shared.log(message: "Data integrity issue detected: \(error.localizedDescription)", type: .error)
            }
        }
    }
    
    /// Prepares for transitioning to inactive state
    private func prepareForInactive() {
        // Mark that the session is ending cleanly
        UserDefaults.standard.set(true, forKey: "SessionEndedCleanly")
        UserDefaults.standard.synchronize()
    }
}

// MARK: - Supporting Types

/// App lifecycle states
enum AppState: String {
    case inactive = "Inactive"
    case background = "Background"
    case foreground = "Foreground"
    case active = "Active"
}

/// Information about a background task
struct BackgroundTaskInfo {
    let id: UIBackgroundTaskIdentifier
    let name: String
    let startTime: Date
    let isPriority: Bool
}

/// Protocol for view controllers that can save and restore state
protocol StateSavable {
    func saveState() -> Any
    func restoreState(_ state: Any)
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let appWillBecomeInactive = Notification.Name("appWillBecomeInactive")
    static let appDidEnterBackground = Notification.Name("appDidEnterBackground")
    static let appWillEnterForeground = Notification.Name("appWillEnterForeground")
    static let appDidBecomeActive = Notification.Name("appDidBecomeActive")
    static let appWillTerminate = Notification.Name("appWillTerminate")
    static let appDidReceiveMemoryWarning = Notification.Name("appDidReceiveMemoryWarning")
    static let appStateDidSave = Notification.Name("appStateDidSave")
    static let appStateRestorationCompleted = Notification.Name("appStateRestorationCompleted")
    static let appPreparingForForeground = Notification.Name("appPreparingForForeground")
    static let restoreTabState = Notification.Name("restoreTabState")
    static let resumeOperation = Notification.Name("resumeOperation")
    static let appRecoveredFromCrash = Notification.Name("appRecoveredFromCrash")
}

// MARK: - Core Data Manager Extensions

extension CoreDataManager {
    /// Verifies Core Data store consistency
    func verifyStoreConsistency() throws {
        // Implementation would check Core Data store integrity
        // This is a placeholder for the actual implementation
    }
    
    /// Attempts to recover a damaged Core Data store
    func attemptStoreRecovery() {
        // Implementation would attempt to recover from Core Data issues
        // This is a placeholder for the actual implementation
    }
}
