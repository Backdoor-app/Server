// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

/// Integrates all performance optimizations into the app
final class OptimizationIntegrator {
    // MARK: - Singleton

    /// Shared instance
    static let shared = OptimizationIntegrator()

    // MARK: - Properties

    /// Flag to track if the optimizations have been integrated
    private var didIntegrateOptimizations = false

    // MARK: - Initialization

    private init() {}

    // MARK: - Integration

    /// Integrate all optimizations into the app
    /// - Parameter application: The UIApplication instance
    func integrateOptimizations(in _: UIApplication) {
        guard !didIntegrateOptimizations else {
            Debug.shared.log(message: "Optimizations already integrated", type: .warning)
            return
        }

        Debug.shared.log(message: "Integrating performance optimizations", type: .info)

        // Initialize optimizer components
        initializeOptimizers()

        // Register for notifications
        registerForNotifications()

        // Optimize table and collection views
        optimizeListViews()

        // Configure networking optimizations
        configureNetworkOptimizations()

        // Swizzle methods for global performance improvements
        applyMethodSwizzling()

        didIntegrateOptimizations = true
        Debug.shared.log(message: "Performance optimizations integrated successfully", type: .info)
    }

    // MARK: - Setup Methods

    /// Initialize all optimizer components
    private func initializeOptimizers() {
        // The act of accessing these shared instances initializes them
        _ = AppPerformanceOptimizer.shared
        _ = ImageCache.shared
        _ = NetworkManager.shared

        Debug.shared.log(message: "Performance optimizers initialized", type: .debug)
    }

    /// Register for needed notifications
    private func registerForNotifications() {
        // Register for view controller lifecycle notifications
        let notificationCenter = NotificationCenter.default

        // Register for app lifecycle notifications
        notificationCenter.addObserver(
            self,
            selector: #selector(handleAppDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )

        // Register for low memory warnings
        notificationCenter.addObserver(
            self,
            selector: #selector(handleLowMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )

        Debug.shared.log(message: "Registered for system notifications", type: .debug)
    }

    /// Configure network optimizations
    private func configureNetworkOptimizations() {
        // Configure URLCache with appropriate sizes
        let memoryCapacity = 10 * 1024 * 1024 // 10 MB
        let diskCapacity = 50 * 1024 * 1024 // 50 MB
        let cache = URLCache(memoryCapacity: memoryCapacity, diskCapacity: diskCapacity, diskPath: "URLCache")
        URLCache.shared = cache

        // Configure URLSession
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .useProtocolCachePolicy
        config.urlCache = cache
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60

        Debug.shared.log(message: "Network optimizations configured", type: .debug)
    }

    /// Apply global optimizations to table and collection views
    private func optimizeListViews() {
        // Register for UITableView.shouldPrefetchDataSource
        // We'll optimize table views as they're created

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleViewControllerDidLoad),
            name: NSNotification.Name("ViewControllerDidLoad"),
            object: nil
        )

        // Enable cell reuse and prefetching by default for all cells
        UITableView.appearance().isPrefetchingEnabled = true

        Debug.shared.log(message: "List view prefetching enabled", type: .debug)
    }

    /// Apply method swizzling for performance optimizations
    private func applyMethodSwizzling() {
        // Swizzle view controller lifecycle methods to add optimization
        UIViewController.optimizeLifecycleMethods()

        Debug.shared.log(message: "Method swizzling applied", type: .debug)
    }

    // MARK: - Notification Handlers

    @objc private func handleViewControllerDidLoad(_ notification: Notification) {
        if let viewController = notification.object as? UIViewController {
            optimizeViewController(viewController)
        }
    }

    @objc private func handleAppDidBecomeActive() {
        Debug.shared.log(message: "App became active, applying dynamic optimizations", type: .debug)

        // Optimize currently visible view controller
        if let topVC = UIApplication.shared.topMostViewController() {
            optimizeViewController(topVC)
        }
    }

    @objc private func handleLowMemoryWarning() {
        Debug.shared.log(message: "Handling low memory warning", type: .warning)

        // Clear image caches
        ImageCache.shared.clearCache()

        // Clear network caches
        NetworkManager.shared.clearCache()

        // Clear any NSCache instances
        NotificationCenter.default.post(name: NSNotification.Name("ClearAllCaches"), object: nil)
    }

    // MARK: - View Controller Optimization

    /// Optimize a view controller for better performance
    /// - Parameter viewController: The view controller to optimize
    func optimizeViewController(_ viewController: UIViewController) {
        // Use AppPerformanceOptimizer to optimize the view controller
        AppPerformanceOptimizer.shared.optimizeViewController(viewController)

        // Optimize table views
        if let tableVC = viewController as? UITableViewController {
            optimizeTableViewController(tableVC)
        }

        // Optimize child view controllers
        for childVC in viewController.children {
            optimizeViewController(childVC)
        }
    }

    /// Optimize a table view controller
    /// - Parameter tableViewController: The table view controller to optimize
    private func optimizeTableViewController(_ tableViewController: UITableViewController) {
        let tableView = tableViewController.tableView

        // Enable prefetching if not already enabled
        tableView?.prefetchDataSource = tableViewController as? UITableViewDataSourcePrefetching

        // Set estimatedRowHeight for better performance
        if tableView?.rowHeight == UITableView.automaticDimension {
            tableView?.estimatedRowHeight = 100
        }

        // Enable cell reuse identifier tracker
        // This helps identify cells that aren't being reused properly
        trackCellReuseIdentifiers(in: tableView)
    }

    /// Track cell reuse identifiers to find inefficient cell usage
    /// - Parameter tableView: The table view to track
    private func trackCellReuseIdentifiers(in tableView: UITableView?) {
        guard let tableView = tableView else { return }

        // We'll add an associated object to the table view to track reuse identifiers
        let tracker = NSMutableSet()
        objc_setAssociatedObject(
            tableView,
            "com.backdoor.cellReuseIdentifierTracker",
            tracker,
            .OBJC_ASSOCIATION_RETAIN_NONATOMIC
        )

        // Hook into the dequeueReusableCell method
        // This is done via method swizzling in UITableView extension
    }
}

// MARK: - UIViewController Extension

extension UIViewController {
    /// Apply optimization-related method swizzling
    static func optimizeLifecycleMethods() {
        let originalMethod = class_getInstanceMethod(UIViewController.self, #selector(viewDidLoad))
        let swizzledMethod = class_getInstanceMethod(UIViewController.self, #selector(optimized_viewDidLoad))

        if let originalMethod = originalMethod, let swizzledMethod = swizzledMethod {
            method_exchangeImplementations(originalMethod, swizzledMethod)
        }
    }

    @objc private func optimized_viewDidLoad() {
        // Call original implementation
        optimized_viewDidLoad()

        // Post notification for optimizer
        NotificationCenter.default.post(name: NSNotification.Name("ViewControllerDidLoad"), object: self)

        // Apply view-specific optimizations
        optimizeViewHierarchy()
    }

    /// Apply optimizations to the view hierarchy
    private func optimizeViewHierarchy() {
        // Ensure rasterization for complex views
        for subview in view.subviews {
            if subview.layer.shadowOpacity > 0 ||
                subview.layer.cornerRadius > 0 ||
                subview.layer.borderWidth > 0
            {
                // Rasterize layers with shadows or rounded corners
                subview.layer.shouldRasterize = true
                subview.layer.rasterizationScale = UIScreen.main.scale
            }
        }

        // Optimize scroll view content insets
        for case let scrollView as UIScrollView in view.subviews {
            scrollView.contentInsetAdjustmentBehavior = .automatic
        }
    }
}

// MARK: - Integration with AppDelegate

extension AppDelegate {
    /// Call this method from application(_:didFinishLaunchingWithOptions:)
    func integratePerformanceOptimizations() {
        OptimizationIntegrator.shared.integrateOptimizations(in: UIApplication.shared)
    }
}
