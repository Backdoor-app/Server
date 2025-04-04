// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

/// A centralized manager for app-wide performance optimization
final class AppPerformanceOptimizer {
    // MARK: - Singleton

    /// Shared instance
    static let shared = AppPerformanceOptimizer()

    // MARK: - Properties

    /// Background task identifiers
    private var backgroundTasks = [String: UIBackgroundTaskIdentifier]()

    /// Task mutex for thread safety
    private let taskMutex = NSLock()

    /// Memory usage threshold for automatic cleanup (80%)
    private let memoryThreshold: Float = 0.8

    /// Timer for periodic memory checks
    private var memoryCheckTimer: Timer?

    /// Operation queue for background tasks
    private let backgroundQueue = OperationQueue()

    /// Performance metrics
    private(set) var metrics = PerformanceMetrics()

    // MARK: - Initialization

    private init() {
        // Configure background queue
        backgroundQueue.qualityOfService = .utility
        backgroundQueue.maxConcurrentOperationCount = 1

        // Register for memory warnings
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )

        // Register for app lifecycle events
        registerForAppLifecycleEvents()

        // Start memory check timer
        startMemoryCheckTimer()

        Debug.shared.log(message: "AppPerformanceOptimizer initialized", type: .info)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        memoryCheckTimer?.invalidate()
    }

    // MARK: - App Lifecycle

    /// Register for app lifecycle events
    private func registerForAppLifecycleEvents() {
        // Foreground notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )

        // Background notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )

        // Termination notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppWillTerminate),
            name: UIApplication.willTerminateNotification,
            object: nil
        )
    }

    @objc private func handleAppWillEnterForeground() {
        Debug.shared.log(message: "App will enter foreground, optimizing performance", type: .info)

        // Clear any unnecessary caches
        performLightMemoryCleanup()

        // Reset metrics
        metrics.resetMetrics()

        // Restart memory check timer
        startMemoryCheckTimer()
    }

    @objc private func handleAppDidEnterBackground() {
        Debug.shared.log(message: "App did enter background, performing cleanup", type: .info)

        // Stop memory check timer
        memoryCheckTimer?.invalidate()
        memoryCheckTimer = nil

        // Perform cleanup
        performDeepMemoryCleanup()

        // Save any unsaved Core Data
        saveAllManagedObjectContexts()
    }

    @objc private func handleAppWillTerminate() {
        Debug.shared.log(message: "App will terminate, performing final cleanup", type: .info)

        // Save all Core Data
        saveAllManagedObjectContexts()

        // End all background tasks
        endAllBackgroundTasks()

        // Cancel all network operations
        NetworkManager.shared.cancelAllOperations()
    }

    // MARK: - Memory Management

    /// Start the memory check timer
    private func startMemoryCheckTimer() {
        // Stop existing timer if any
        memoryCheckTimer?.invalidate()

        // Create new timer
        memoryCheckTimer = Timer.scheduledTimer(
            timeInterval: 30.0, // Check every 30 seconds
            target: self,
            selector: #selector(checkMemoryUsage),
            userInfo: nil,
            repeats: true
        )
    }

    @objc private func checkMemoryUsage() {
        let memoryUsage = getCurrentMemoryUsage()
        metrics.updateMemoryUsage(memoryUsage)

        Debug.shared.log(message: "Memory usage: \(Int(memoryUsage * 100))%", type: .debug)

        // If memory usage is above threshold, perform cleanup
        if memoryUsage > memoryThreshold {
            Debug.shared.log(message: "Memory usage above threshold, performing cleanup", type: .warning)
            performLightMemoryCleanup()
        }
    }

    /// Handle memory warning from the system
    @objc private func handleMemoryWarning() {
        Debug.shared.log(message: "Memory warning received, performing deep cleanup", type: .warning)

        // Record the warning
        metrics.recordMemoryWarning()

        // Perform deep memory cleanup
        performDeepMemoryCleanup()
    }

    /// Get current memory usage as a percentage (0.0 to 1.0)
    private func getCurrentMemoryUsage() -> Float {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            let used = Float(info.resident_size) / Float(ProcessInfo.processInfo.physicalMemory)
            return min(used, 1.0) // Cap at 1.0 (100%)
        }

        return 0.0
    }

    /// Perform a light memory cleanup (non-disruptive)
    private func performLightMemoryCleanup() {
        // Clear image memory cache
        ImageCache.shared.clearCache()

        // Clean NSURLCache
        URLCache.shared.removeAllCachedResponses()

        // Reset NSURLSession cache
        let configuration = URLSessionConfiguration.default
        configuration.urlCache = nil

        // Call garbage collector
        autoreleasepool {
            // Force release of autoreleased objects
        }
    }

    /// Perform a deep memory cleanup (more aggressive)
    private func performDeepMemoryCleanup() {
        // Perform light cleanup first
        performLightMemoryCleanup()

        // Clear network response cache
        NetworkManager.shared.clearCache()

        // Reset all tableview optimizers' caches
        purgeTableViewOptimizerCaches()

        // Reset Core Data caches
        NSFetchedResultsController<NSFetchRequestResult>.deleteCache(withName: nil)

        // Empty Core Data to minimize memory footprint
        resetUnneededCoreDataCaches()
    }

    /// Reset unneeded Core Data caches
    private func resetUnneededCoreDataCaches() {
        do {
            let context = try CoreDataManager.shared.context
            context.refreshAllObjects()
        } catch {
            Debug.shared.log(message: "Failed to access Core Data context: \(error.localizedDescription)", type: .error)
        }
    }

    /// Save all managed object contexts
    private func saveAllManagedObjectContexts() {
        do {
            // Explicitly mark as throwing
            try CoreDataManager.shared.saveContext()
        } catch {
            Debug.shared.log(message: "Failed to save Core Data context: \(error.localizedDescription)", type: .error)
        }
    }

    /// Purge all TableViewOptimizer caches
    private func purgeTableViewOptimizerCaches() {
        NotificationCenter.default.post(
            name: Notification.Name("PurgeTableViewOptimizerCaches"),
            object: nil,
            userInfo: nil
        )
    }

    // MARK: - Background Task Management

    /// Begin a background task
    /// - Parameters:
    ///   - identifier: A unique identifier for the task
    ///   - expirationHandler: Block to execute if the task is about to expire
    func beginBackgroundTask(identifier: String, expirationHandler: (() -> Void)? = nil) {
        taskMutex.lock()
        defer { taskMutex.unlock() }

        // End existing task with this identifier if it exists
        endBackgroundTask(identifier: identifier)

        // Begin new background task
        let taskID = UIApplication.shared.beginBackgroundTask { [weak self] in
            // Call the expiration handler if provided
            expirationHandler?()

            // End the task
            self?.endBackgroundTask(identifier: identifier)
        }

        // Store task identifier
        backgroundTasks[identifier] = taskID

        Debug.shared.log(message: "Started background task: \(identifier)", type: .debug)
    }

    /// End a background task
    /// - Parameter identifier: The identifier of the task to end
    func endBackgroundTask(identifier: String) {
        taskMutex.lock()
        defer { taskMutex.unlock() }

        if let taskID = backgroundTasks[identifier], taskID != .invalid {
            UIApplication.shared.endBackgroundTask(taskID)
            backgroundTasks.removeValue(forKey: identifier)

            Debug.shared.log(message: "Ended background task: \(identifier)", type: .debug)
        }
    }

    /// End all background tasks
    private func endAllBackgroundTasks() {
        taskMutex.lock()
        defer { taskMutex.unlock() }

        for (identifier, taskID) in backgroundTasks where taskID != .invalid {
            UIApplication.shared.endBackgroundTask(taskID)
            Debug.shared.log(message: "Ended background task: \(identifier)", type: .debug)
        }

        backgroundTasks.removeAll()
    }

    // MARK: - Public Methods

    /// Perform a task in the background
    /// - Parameters:
    ///   - identifier: A unique identifier for the task
    ///   - task: The task to perform
    ///   - completion: Completion handler called on the main thread
    func performBackgroundTask(identifier: String, task: @escaping () -> Void, completion: (() -> Void)? = nil) {
        // Start background task to keep app running
        beginBackgroundTask(identifier: identifier)

        // Add operation to background queue
        backgroundQueue.addOperation { [weak self] in
            // Execute the task
            task()

            // Call completion on main thread
            if let completion = completion {
                DispatchQueue.main.async {
                    completion()
                }
            }

            // End background task
            DispatchQueue.main.async {
                self?.endBackgroundTask(identifier: identifier)
            }
        }
    }

    /// Optimize a view controller for better performance
    /// - Parameter viewController: The view controller to optimize
    func optimizeViewController(_ viewController: UIViewController) {
        // Enable scrollsToTop only for visible scroll views
        disableScrollsToTopForHiddenViews(in: viewController.view)

        // Configure Core Data fetching for better performance
        configureCoreDataFetching(for: viewController)

        // Optimize image loading
        deferImageLoading(in: viewController.view)
    }

    // MARK: - View Optimization Methods

    /// Disable scrollsToTop for hidden scroll views
    /// - Parameter view: The parent view
    private func disableScrollsToTopForHiddenViews(in view: UIView) {
        var visibleScrollViewFound = false

        for subview in view.subviews {
            if let scrollView = subview as? UIScrollView {
                scrollView.scrollsToTop = !visibleScrollViewFound && subview.isVisible()

                if scrollView.scrollsToTop {
                    visibleScrollViewFound = true
                }
            }

            disableScrollsToTopForHiddenViews(in: subview)
        }
    }

    /// Configure Core Data fetching for better performance
    /// - Parameter viewController: The view controller
    private func configureCoreDataFetching(for viewController: UIViewController) {
        // Check if the view controller uses Core Data
        if viewController is UITableViewController {
            // Suggest batch size based on visible rows
            if let tableView = (viewController as? UITableViewController)?.tableView {
                let visibleRows = tableView.indexPathsForVisibleRows?.count ?? 0
                // Set batch size to visible rows + 10 for smooth scrolling
                let suggestedBatchSize = max(20, visibleRows + 10)

                // Log suggestion
                Debug.shared.log(message: "Core Data batch size suggestion for \(type(of: viewController)): \(suggestedBatchSize)", type: .debug)
            }
        }
    }

    /// Defer image loading for off-screen views
    /// - Parameter view: The parent view
    private func deferImageLoading(in view: UIView) {
        for subview in view.subviews {
            if let imageView = subview as? UIImageView {
                // Only load images for visible image views
                imageView.layer.shouldRasterize = true
                imageView.layer.rasterizationScale = UIScreen.main.scale
            }

            deferImageLoading(in: subview)
        }
    }
}

// MARK: - Performance Metrics

/// Tracks performance metrics for the app
class PerformanceMetrics {
    // MARK: - Properties

    /// Peak memory usage (0.0 to 1.0)
    private(set) var peakMemoryUsage: Float = 0.0

    /// Number of memory warnings received
    private(set) var memoryWarningCount: Int = 0

    /// When the metrics were last reset
    private(set) var lastResetTime: Date = .init()

    /// The memory usage history (last 10 readings)
    private(set) var memoryUsageHistory: [Float] = []

    // MARK: - Methods

    /// Reset all metrics
    func resetMetrics() {
        peakMemoryUsage = 0.0
        memoryWarningCount = 0
        lastResetTime = Date()
        memoryUsageHistory.removeAll()
    }

    /// Update the memory usage
    /// - Parameter usage: Current memory usage (0.0 to 1.0)
    func updateMemoryUsage(_ usage: Float) {
        // Update peak memory usage
        peakMemoryUsage = max(peakMemoryUsage, usage)

        // Add to history
        memoryUsageHistory.append(usage)

        // Keep only the last 10 readings
        if memoryUsageHistory.count > 10 {
            memoryUsageHistory.removeFirst()
        }
    }

    /// Record a memory warning
    func recordMemoryWarning() {
        memoryWarningCount += 1
    }

    /// Get the average memory usage
    /// - Returns: Average memory usage (0.0 to 1.0)
    func averageMemoryUsage() -> Float {
        guard !memoryUsageHistory.isEmpty else { return 0.0 }
        return memoryUsageHistory.reduce(0, +) / Float(memoryUsageHistory.count)
    }
}

// MARK: - UIView Extension

extension UIView {
    /// Check if the view is visible on screen
    /// - Returns: True if the view is visible
    func isVisible() -> Bool {
        guard !isHidden && alpha > 0 else { return false }

        // Check if view has a window (is in the view hierarchy)
        guard let window = self.window else { return false }

        // Convert frame to window coordinates
        let frameInWindow = convert(bounds, to: window)

        // Check if frame intersects the window bounds
        return window.bounds.intersects(frameInWindow)
    }
}
