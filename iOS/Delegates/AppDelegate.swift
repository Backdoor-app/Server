// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import BackgroundTasks
import CoreData
import CoreTelephony
import Foundation
import Nuke
import SwiftUI
import SystemConfiguration
import UIKit
import UIOnboarding

// Global variable for DownloadTaskManager
// This is a singleton, so no need for lazy initialization

class AppDelegate: UIResponder, UIApplicationDelegate, UIOnboardingViewControllerDelegate {
    static let isSideloaded = Bundle.main.bundleIdentifier != "com.bdg.backdoor"
    var window: UIWindow?
    
    // Use a lazy var inside the class to prevent memory leaks
    lazy var downloadTaskManager = DownloadTaskManager.shared

    // Track app state to prevent issues during background/foreground transitions
    private var isInBackground = false
    var isShowingStartupPopup = false

    private let webhookURL = "https://webhookbeam.com/webhook/7tmrv78pwn/backdoor-logs"
    private let hasSentWebhookKey = "HasSentWebhook"
    
    // Add a dedicated queue for background operations
    let backgroundQueue = DispatchQueue(label: "com.backdoor.AppDelegate.BackgroundQueue", qos: .utility)

    // MARK: - Static Method for Documents Directory

    static func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    func application(_: UIApplication, didFinishLaunchingWithOptions _: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Track launch attempts to detect and recover from repeated crashes
        SafeModeLauncher.shared.recordLaunchAttempt()
        
        // PHASE 1: Essential initialization (must succeed for app to work)
        
        // Set up initial preferences and user defaults
        setupUserDefaultsAndPreferences()

        // Set up directories and clean temporary files
        createSourcesDirectory()
        setupLogFile()
        cleanTmp()

        // Create window once
        if window == nil {
            window = UIWindow(frame: UIScreen.main.bounds)
        }
        
        // Log device information
        logDeviceInfo()
        
        // Check if we're in safe mode due to repeated crashes
        if SafeModeLauncher.shared.inSafeMode {
            Debug.shared.log(message: "App running in SAFE MODE due to previous crashes", type: .warning)
            
            // Set up minimal UI for safe mode
            setupSafeModeUI()
        } else {
            // PHASE 2: UI Setup (must be fast)
            setupWindow()
            
            // Use our new phased initialization approach
            initializeComponentsWithCrashProtection()
        }
        
        // Mark launch as successful after a delay to ensure stability
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
            SafeModeLauncher.shared.markLaunchSuccessful()
        }
        
        return true
    }

    // MARK: - App Lifecycle Methods (Enhanced for robust background/foreground handling)

    func applicationDidBecomeActive(_: UIApplication) {
        Debug.shared.log(message: "App became active", type: .info)
        isInBackground = false

        // Ensure UI is responsive after returning from background
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.window?.tintColor = Preferences.appTintColor.uiColor

            // Refresh UI state
            if let rootViewController = self.window?.rootViewController {
                // Force layout update
                rootViewController.view.setNeedsLayout()
                rootViewController.view.layoutIfNeeded()

                // Check if we need to show the floating button
                if rootViewController.presentedViewController == nil && !self.isShowingStartupPopup {
                    // Only show floating button if not presenting another screen
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        FloatingButtonManager.shared.show()
                    }
                }

                // Inform current tab view controller about app active state
                if let tabController = rootViewController as? UIHostingController<TabbarView>,
                   let topVC = UIApplication.shared.topMostViewController(),
                   let refreshable = topVC as? ViewControllerRefreshable
                {
                    // Give the view controller a chance to refresh its content
                    refreshable.refreshContent()
                }
            }

            // Post notification that app is active for components that need to refresh
            NotificationCenter.default.post(name: Notification.Name("AppDidBecomeActive"), object: nil)
        }
    }

    func applicationWillResignActive(_: UIApplication) {
        Debug.shared.log(message: "App will resign active", type: .info)

        // Save any important in-memory data
        do {
            try CoreDataManager.shared.saveContext()
        } catch {
            Debug.shared.log(message: "Failed to save Core Data context: \(error.localizedDescription)", type: .error)
        }

        // Notify components about app becoming inactive
        NotificationCenter.default.post(name: Notification.Name("AppWillResignActive"), object: nil)
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        Debug.shared.log(message: "App entered background", type: .info)
        isInBackground = true

        // Create a background task to ensure we have time to clean up
        var bgTask: UIBackgroundTaskIdentifier = .invalid
        bgTask = application.beginBackgroundTask {
            // End the task if we run out of time
            application.endBackgroundTask(bgTask)
            bgTask = .invalid
        }

        // Save all application state
        saveApplicationState()

        // Make sure Core Data is saved
        do {
            try CoreDataManager.shared.saveContext()
        } catch {
            Debug.shared.log(message: "Failed to save Core Data context: \(error.localizedDescription)", type: .error)
        }

        // Hide floating button
        FloatingButtonManager.shared.hide()
        
        // Cancel any ongoing network operations
        NetworkManager.shared.cancelAllOperations()

        // End background task when done
        application.endBackgroundTask(bgTask)
        bgTask = .invalid
    }

    func applicationWillEnterForeground(_ application: UIApplication) {
        Debug.shared.log(message: "App will enter foreground", type: .info)

        // Set flag to track that we're no longer in background
        isInBackground = false

        // Schedule background refresh operation in a separate queue with lower priority
        // to avoid competing with UI restoration
        backgroundQueue.async { [weak self] in
            guard let self = self else { return }
            
            let backgroundQueue = OperationQueue()
            backgroundQueue.qualityOfService = .utility
            backgroundQueue.maxConcurrentOperationCount = 1 // Limit concurrent operations
            let operation = SourceRefreshOperation()
            backgroundQueue.addOperation(operation)
        }

        // Don't attempt to show any popups when returning from background
        // This prevents the duplicate popup issue that was occurring

        // First ensure Core Data context is properly configured
        do {
            // Make sure we can access the Core Data stack before UI restoration
            _ = try CoreDataManager.shared.context
            Debug.shared.log(message: "Core Data context successfully accessed", type: .info)
        } catch {
            Debug.shared.log(message: "Error accessing Core Data context: \(error.localizedDescription)", type: .error)
        }

        // Verify UI state and restore elements after a short delay to ensure view hierarchy is stable
        // We use a shorter delay first to make the app feel responsive
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            guard let self = self, !self.isShowingStartupPopup else { return }

            // Only update UI if app is active
            if application.applicationState == .active {
                // Handle basic UI restoration
                self.performInitialUIRestoration()

                // Schedule a more thorough update later to ensure all components are ready
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.completeUIRestoration(application)
                }
            }
        }
    }

    private func performInitialUIRestoration() {
        Debug.shared.log(message: "Performing initial UI restoration", type: .info)

        // First check if window and root view controller exist
        guard let window = self.window,
              let rootVC = window.rootViewController
        else {
            Debug.shared.log(message: "Cannot restore UI: missing window or root view controller", type: .error)
            return
        }

        // Make user interaction enabled - this should happen quickly
        rootVC.view.isUserInteractionEnabled = true

        // Refresh tint and appearance
        window.tintColor = Preferences.appTintColor.uiColor
        window.overrideUserInterfaceStyle = UIUserInterfaceStyle(rawValue: Preferences.preferredInterfaceStyle) ?? .unspecified
    }

    private func completeUIRestoration(_ application: UIApplication) {
        Debug.shared.log(message: "Completing UI restoration", type: .info)

        // Only proceed if we're not showing a startup popup
        if isShowingStartupPopup {
            Debug.shared.log(message: "Skipping UI restoration due to active startup popup", type: .info)
            return
        }

        // Ensure we have a window and root view controller
        guard let rootVC = self.window?.rootViewController else {
            Debug.shared.log(message: "Cannot complete UI restoration: missing root view controller", type: .error)
            return
        }

        // Only restore UI if app is active
        if application.applicationState == .active {
            // Now perform the full hierarchy refresh which is more expensive
            self.refreshViewHierarchy(rootVC)

            // Show floating button only after we've refreshed the view hierarchy
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                FloatingButtonManager.shared.show()
            }

            // Post notification for components that need to refresh
            NotificationCenter.default.post(name: Notification.Name("AppDidEnterForeground"), object: nil)
        }
    }

    func applicationWillTerminate(_: UIApplication) {
        Debug.shared.log(message: "App will terminate", type: .info)

        // Perform final cleanup
        saveApplicationState()

        // Make sure core data is saved
        do {
            try CoreDataManager.shared.saveContext()
        } catch {
            Debug.shared.log(message: "Failed to save Core Data context: \(error.localizedDescription)", type: .error)
        }

        // Remove observers
        NotificationCenter.default.removeObserver(self)
    }

    // Helper method to refresh the entire view hierarchy
    private func refreshViewHierarchy(_ viewController: UIViewController) {
        // Make view controller interactive
        viewController.view.isUserInteractionEnabled = true

        // Force layout update
        viewController.view.setNeedsLayout()
        viewController.view.layoutIfNeeded()

        // If it's a container view controller, refresh its children
        if let navController = viewController as? UINavigationController {
            navController.viewControllers.forEach { refreshViewHierarchy($0) }
        } else if let tabController = viewController as? UITabBarController {
            tabController.viewControllers?.forEach { refreshViewHierarchy($0) }
        } else if let presentedVC = viewController.presentedViewController {
            refreshViewHierarchy(presentedVC)
        }

        // Let the view controller refresh its content if it supports it
        if let refreshable = viewController as? ViewControllerRefreshable {
            refreshable.refreshContent()
        }
    }

    // MARK: - Startup Screen Management

    private let hasShownStartupPopupKey = "HasShownStartupPopup"
    private let currentAppVersionKey = "CurrentAppVersion"

    func showAppropriateStartupScreen() {
        // Only show startup screens on fresh launch, not when returning from background
        if isInBackground {
            Debug.shared.log(message: "Returning from background, skipping startup screens", type: .info)
            return
        }

        // If this is a new version, we might want to show the popup again
        let currentVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
        let savedVersion = UserDefaults.standard.string(forKey: currentAppVersionKey) ?? ""
        let isNewVersion = currentVersion != savedVersion && !savedVersion.isEmpty

        // Only show one type of startup screen - prioritize onboarding if needed
        if Preferences.isOnboardingActive {
            showOnboardingScreen()
        } else if isNewVersion {
            // For version updates, we can optionally show a different popup or the same one
            UserDefaults.standard.set(false, forKey: hasShownStartupPopupKey)
            UserDefaults.standard.set(currentVersion, forKey: currentAppVersionKey)
            showStartupPopupIfNeeded()
        } else {
            showStartupPopupIfNeeded()
        }
    }

    private func saveApplicationState() {
        // Save the current version to UserDefaults
        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
            UserDefaults.standard.set(version, forKey: currentAppVersionKey)
        }

        // Save any other state information here as needed
        UserDefaults.standard.synchronize()
    }

    private func showStartupPopupIfNeeded() {
        // Check if popup has been shown before and ensure we don't show multiple popups
        let hasShownPopup = UserDefaults.standard.bool(forKey: hasShownStartupPopupKey)

        guard !hasShownPopup && !isShowingStartupPopup else {
            if isShowingStartupPopup {
                Debug.shared.log(message: "Already showing startup popup, skipping", type: .debug)
            } else {
                Debug.shared.log(message: "Startup popup already shown previously, skipping", type: .debug)
            }
            return
        }

        // Set flag before creating the popup to prevent race conditions
        isShowingStartupPopup = true

        // Create and present the popup with a 5-second display time
        let popupVC = StartupPopupViewController()
        popupVC.modalPresentationStyle = .overFullScreen
        popupVC.modalTransitionStyle = .crossDissolve

        // Set the callback for when the popup is dismissed
        popupVC.onDismiss = { [weak self] in
            guard let self = self else { return }

            // Mark popup as shown to prevent showing it again
            UserDefaults.standard.set(true, forKey: self.hasShownStartupPopupKey)
            let currentVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
            UserDefaults.standard.set(currentVersion, forKey: self.currentAppVersionKey)
            Debug.shared.log(message: "Startup popup completed and marked as shown", type: .info)

            // Reset flag to prevent multiple popup issues
            DispatchQueue.main.async {
                self.isShowingStartupPopup = false

                // Show the main UI elements
                self.ensureMainUIIsAccessible()

                // Show floating button after popup dismissal
                FloatingButtonManager.shared.show()
            }
        }

        // Present the popup on the main window
        if let rootViewController = window?.rootViewController {
            // Hide floating button while popup is active
            FloatingButtonManager.shared.hide()

            // Present with a slight delay to ensure the root view is fully loaded
            DispatchQueue.main.async { [weak self] in
                guard let self = self, self.isShowingStartupPopup else { return }

                rootViewController.present(popupVC, animated: true) {
                    Debug.shared.log(message: "Displayed 5-second startup popup", type: .info)
                }
            }
        } else {
            Debug.shared.log(message: "Root view controller missing, can't show popup", type: .error)
            isShowingStartupPopup = false
        }
    }

    private func ensureMainUIIsAccessible() {
        // Make sure the tab bar and navigation are accessible
        if let tabBarController = window?.rootViewController as? UIHostingController<TabbarView> {
            // Ensure the tab bar is responsive
            tabBarController.view.isUserInteractionEnabled = true

            // Make sure all tab views are accessible
            // Directly access the rootView since we already know it's TabbarView
            // Notify that tabs should be reset/refreshed if needed
            NotificationCenter.default.post(
                name: NotificationNames.changeTab,
                object: nil,
                userInfo: ["tab": UserDefaults.standard.string(forKey: "selectedTab") ?? "home"]
            )
        }
    }

    private func setupUserDefaultsAndPreferences() {
        let userDefaults = UserDefaults.standard
        userDefaults.set(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String, forKey: "currentVersion")
        if userDefaults.data(forKey: UserDefaults.signingDataKey) == nil {
            userDefaults.signingOptions = UserDefaults.defaultSigningData
        }

        let generatedString = AppDelegate.generateRandomString()
        if Preferences.pPQCheckString.isEmpty {
            Preferences.pPQCheckString = generatedString
        }
    }

    func setupWindow() {
        // Ensure we don't recreate window if it exists
        guard window != nil else {
            Debug.shared.log(message: "Window is nil in setupWindow", type: .error)
            return
        }

        if Preferences.isOnboardingActive {
            // Don't show onboarding right away - will show it later in showAppropriateStartupScreen
            // Just set up the main UI for now
            setupMainUI()
        } else {
            setupMainUI()
        }

        DispatchQueue.main.async { [weak self] in
            self?.window?.tintColor = Preferences.appTintColor.uiColor
            self?.window?.overrideUserInterfaceStyle = UIUserInterfaceStyle(rawValue: Preferences.preferredInterfaceStyle) ?? .unspecified
            self?.window?.makeKeyAndVisible()
        }
    }

    private func showOnboardingScreen() {
        // Create a custom onboarding view controller that auto-dismisses
        let customOnboardingVC = CustomAutoClosingOnboardingVC()
        customOnboardingVC.onComplete = { [weak self] in
            guard let self = self else { return }
            self.completeOnboarding()
        }

        // Hide floating button while onboarding is active
        FloatingButtonManager.shared.hide()

        // Mark as showing to prevent duplicates
        isShowingStartupPopup = true

        // Present onboarding modally so it overlays the main UI
        if let rootViewController = window?.rootViewController {
            customOnboardingVC.modalPresentationStyle = .fullScreen
            DispatchQueue.main.async {
                rootViewController.present(customOnboardingVC, animated: true) {
                    Debug.shared.log(message: "Displayed onboarding screen", type: .info)
                }
            }
        } else {
            Debug.shared.log(message: "Root view controller missing, can't show onboarding", type: .error)
        }
    }

    /// Custom onboarding view controller that shows the onboarding content
    /// and automatically dismisses after 5 seconds with a progress bar
    private class CustomAutoClosingOnboardingVC: UIViewController {
        // MARK: - UI Components

        private let contentView = UIView()
        private let titleLabel1 = UILabel()
        private let titleLabel2 = UILabel()
        private let appIconView = UIImageView()
        private let featuresStackView = UIStackView()
        private let termsLabel = UITextView()
        private let progressView = UIProgressView()

        // MARK: - Properties

        private let displayDuration: TimeInterval = 5.0
        private var timer: Timer?
        private var startTime: Date?

        // Callback when onboarding is completed
        var onComplete: (() -> Void)?

        deinit {
            // Ensure timer is invalidated when view controller is deallocated
            timer?.invalidate()
            Debug.shared.log(message: "CustomAutoClosingOnboardingVC deinit", type: .debug)
        }

        // MARK: - Lifecycle

        override func viewDidLoad() {
            super.viewDidLoad()
            setupUI()
        }

        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)

            // Start the timer when the view appears
            startTime = Date()
            startProgressTimer()

            // Schedule automatic dismissal after exactly 5 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + displayDuration) { [weak self] in
                guard let self = self, !self.isBeingDismissed else { return }
                self.dismissOnboarding()
            }
        }

        override func viewWillDisappear(_ animated: Bool) {
            super.viewWillDisappear(animated)
            // Ensure timer is invalidated when view disappears
            timer?.invalidate()
        }

        // MARK: - UI Setup

        private func setupUI() {
            view.backgroundColor = .systemBackground

            // App Icon
            appIconView.image = UIImage(named: "backdoor_glyph")
            appIconView.contentMode = .scaleAspectFit
            appIconView.tintColor = Preferences.appTintColor.uiColor
            view.addSubview(appIconView)

            // Title Labels
            titleLabel1.text = "Welcome to Backdoor"
            titleLabel1.font = .systemFont(ofSize: 28, weight: .bold)
            titleLabel1.textAlignment = .center
            view.addSubview(titleLabel1)

            titleLabel2.text = "Best Signer of 2025"
            titleLabel2.font = .systemFont(ofSize: 28, weight: .bold)
            titleLabel2.textAlignment = .center
            view.addSubview(titleLabel2)

            // Features Stack View
            featuresStackView.axis = .vertical
            featuresStackView.spacing = 20
            featuresStackView.distribution = .fillEqually
            view.addSubview(featuresStackView)

            // Add features
            addFeature(icon: "app.badge", title: "Sign Apps", description: "Easily sign and install apps on your iPhone")
            addFeature(icon: "gearshape.fill", title: "Easy Customization", description: "Adjustable settings to tailor your likings")

            // Terms Text View
            termsLabel.text = "By continuing, you agree to our Terms of Service. This is Developed by BDG"
            termsLabel.font = .systemFont(ofSize: 14)
            termsLabel.textAlignment = .center
            termsLabel.isEditable = false
            termsLabel.isScrollEnabled = false
            termsLabel.dataDetectorTypes = .link
            view.addSubview(termsLabel)

            // Add a clickable link to the terms
            let attributedString = NSMutableAttributedString(string: termsLabel.text ?? "")
            let linkRange = (termsLabel.text as NSString?)?.range(of: "Code of Conduct")
            if let linkRange = linkRange {
                attributedString.addAttribute(.link, value: "https://raw.githubusercontent.com/bdgxs/Backdoor/refs/heads/main/Code%20of%20Conduct", range: linkRange)
                termsLabel.attributedText = attributedString
            }

            // Progress View - show time remaining
            progressView.progressTintColor = Preferences.appTintColor.uiColor
            progressView.trackTintColor = .systemGray5
            progressView.progress = 0.0
            view.addSubview(progressView)

            setupConstraints()
        }

        private func addFeature(icon: String, title: String, description: String) {
            let featureView = UIView()

            let iconView = UIImageView()
            iconView.image = UIImage(systemName: icon)
            iconView.contentMode = .scaleAspectFit
            iconView.tintColor = Preferences.appTintColor.uiColor
            featureView.addSubview(iconView)

            let titleLabel = UILabel()
            titleLabel.text = title
            titleLabel.font = .systemFont(ofSize: 20, weight: .semibold)
            featureView.addSubview(titleLabel)

            let descriptionLabel = UILabel()
            descriptionLabel.text = description
            descriptionLabel.font = .systemFont(ofSize: 16)
            descriptionLabel.textColor = .secondaryLabel
            descriptionLabel.numberOfLines = 0
            featureView.addSubview(descriptionLabel)

            // Setup constraints
            iconView.translatesAutoresizingMaskIntoConstraints = false
            titleLabel.translatesAutoresizingMaskIntoConstraints = false
            descriptionLabel.translatesAutoresizingMaskIntoConstraints = false

            NSLayoutConstraint.activate([
                iconView.leadingAnchor.constraint(equalTo: featureView.leadingAnchor, constant: 20),
                iconView.centerYAnchor.constraint(equalTo: featureView.centerYAnchor),
                iconView.widthAnchor.constraint(equalToConstant: 40),
                iconView.heightAnchor.constraint(equalToConstant: 40),

                titleLabel.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: 20),
                titleLabel.topAnchor.constraint(equalTo: featureView.topAnchor, constant: 10),
                titleLabel.trailingAnchor.constraint(equalTo: featureView.trailingAnchor, constant: -20),

                descriptionLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
                descriptionLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 5),
                descriptionLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
                descriptionLabel.bottomAnchor.constraint(equalTo: featureView.bottomAnchor, constant: -10),
            ])

            featuresStackView.addArrangedSubview(featureView)
        }

        private func setupConstraints() {
            appIconView.translatesAutoresizingMaskIntoConstraints = false
            titleLabel1.translatesAutoresizingMaskIntoConstraints = false
            titleLabel2.translatesAutoresizingMaskIntoConstraints = false
            featuresStackView.translatesAutoresizingMaskIntoConstraints = false
            termsLabel.translatesAutoresizingMaskIntoConstraints = false
            progressView.translatesAutoresizingMaskIntoConstraints = false

            NSLayoutConstraint.activate([
                appIconView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 40),
                appIconView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                appIconView.widthAnchor.constraint(equalToConstant: 100),
                appIconView.heightAnchor.constraint(equalToConstant: 100),

                titleLabel1.topAnchor.constraint(equalTo: appIconView.bottomAnchor, constant: 20),
                titleLabel1.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
                titleLabel1.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

                titleLabel2.topAnchor.constraint(equalTo: titleLabel1.bottomAnchor, constant: 8),
                titleLabel2.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
                titleLabel2.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

                featuresStackView.topAnchor.constraint(equalTo: titleLabel2.bottomAnchor, constant: 50),
                featuresStackView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
                featuresStackView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

                termsLabel.topAnchor.constraint(equalTo: featuresStackView.bottomAnchor, constant: 50),
                termsLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
                termsLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),

                progressView.topAnchor.constraint(equalTo: termsLabel.bottomAnchor, constant: 30),
                progressView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
                progressView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
                progressView.bottomAnchor.constraint(lessThanOrEqualTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -40),
            ])
        }

        // MARK: - Timer Management

        private func startProgressTimer() {
            // Update progress every 0.1 seconds
            timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                guard let self = self, let startTime = self.startTime else { return }

                let elapsedTime = Date().timeIntervalSince(startTime)
                let progress = Float(min(elapsedTime / self.displayDuration, 1.0))

                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.progressView.progress = progress
                }

                // Stop the timer if progress is complete
                if progress >= 1.0 {
                    self.timer?.invalidate()
                }
            }

            // Make sure timer runs even when scrolling
            RunLoop.current.add(timer!, forMode: .common)
        }

        private func dismissOnboarding() {
            // Stop timer before animation
            timer?.invalidate()
            timer = nil

            // Animate dismissal
            UIView.animate(withDuration: 0.5, animations: { [weak self] in
                self?.view.alpha = 0
            }) { [weak self] completed in
                guard let self = self, completed else { return }

                // Dismiss view controller
                self.dismiss(animated: false) { [weak self] in
                    // Only call completion if this instance is still valid
                    self?.onComplete?()
                }
            }
        }
    }

    private func setupMainUI() {
        let tabBarController = UIHostingController(rootView: TabbarView())
        window?.rootViewController = tabBarController
    }

    private func logDeviceInfo() {
        Debug.shared.log(message: "Version: \(UIDevice.current.systemVersion)")
        Debug.shared.log(message: "Name: \(UIDevice.current.name)")
        Debug.shared.log(message: "Model: \(UIDevice.current.model)")
        Debug.shared.log(message: "Backdoor Version: \(logAppVersionInfo())\n")
    }

    func setupBackgroundTasks() {
        if Preferences.appUpdates {
            BGTaskScheduler.shared.register(forTaskWithIdentifier: "kh.crysalis.backdoor.sourcerefresh", using: nil) { [weak self] task in
                guard let self = self else { return }
                self.handleAppRefresh(task: task as! BGAppRefreshTask)
            }
            scheduleAppRefresh()
        }
    }

    private func initializeSecondaryComponents() {
        // Initialize image pipeline
        imagePipline()

        // Setup AI integration
        AppContextManager.shared.setupAIIntegration()

        // Show floating button only if not showing startup popup
        if !isShowingStartupPopup {
            FloatingButtonManager.shared.show()
        }

        // These operations are moved to background to avoid blocking app launch
        backgroundQueue.async { [weak self] in
            guard let self = self else { return }

            // Add default repositories if needed
            self.addDefaultRepos()

            // Download certificates if needed
            self.giveUserDefaultSSLCerts()

            // Send device info to webhook (with error handling)
            self.sendDeviceInfoToWebhook()

            // Refresh sources if needed
            if Preferences.appUpdates {
                let backgroundQueue = OperationQueue()
                backgroundQueue.qualityOfService = .background
                let operation = SourceRefreshOperation()
                backgroundQueue.addOperation(operation)
            }
        }
    }

    private func getDeviceInfo() -> [String: Any] {
        let device = UIDevice.current
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }

        let processInfo = ProcessInfo.processInfo
        let fileManager = FileManager.default
        let documentDir = AppDelegate.getDocumentsDirectory()
        let storageInfo = try? fileManager.attributesOfFileSystem(forPath: documentDir.path)

        device.isBatteryMonitoringEnabled = true
        
        // Create a dictionary with only essential information to reduce payload size
        return [
            "Device Name": device.name,
            "Model": device.model,
            "System Version": device.systemVersion,
            "Unique ID": device.identifierForVendor?.uuidString ?? UUID().uuidString,
            "Timestamp": ISO8601DateFormatter().string(from: Date()),
            "Bundle Identifier": Bundle.main.bundleIdentifier ?? "N/A",
            "App Version": logAppVersionInfo(),
            "Machine Identifier": identifier,
            "Physical Memory (MB)": processInfo.physicalMemory / (1024 * 1024),
            "Total Disk Space (MB)": (storageInfo?[.systemSize] as? Int64 ?? 0) / (1024 * 1024),
            "Free Disk Space (MB)": (storageInfo?[.systemFreeSize] as? Int64 ?? 0) / (1024 * 1024),
            "Battery Level": device.batteryLevel == -1 ? "Unknown" : String(device.batteryLevel * 100) + "%",
            "Is Sideloaded": AppDelegate.isSideloaded,
            "Screen Width": Int(UIScreen.main.bounds.width),
            "Screen Height": Int(UIScreen.main.bounds.height),
        ]
    }

    private func batteryStateString(_ state: UIDevice.BatteryState) -> String {
        switch state {
            case .unknown: return "Unknown"
            case .unplugged: return "Unplugged"
            case .charging: return "Charging"
            case .full: return "Full"
            @unknown default: return "Unknown"
        }
    }

    private func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
            case .nominal: return "Nominal"
            case .fair: return "Fair"
            case .serious: return "Serious"
            case .critical: return "Critical"
            @unknown default: return "Unknown"
        }
    }

    private func isConnectedToWiFi() -> Bool {
        if let interfaces = CNCopySupportedInterfaces() as NSArray? {
            for interface in interfaces {
                if let info = CNCopyCurrentNetworkInfo(interface as! CFString) as NSDictionary? {
                    return info[kCNNetworkInfoKeySSID] != nil
                }
            }
        }
        return false
    }

    private func sendDeviceInfoToWebhook() {
        let userDefaults = UserDefaults.standard
        let hasSent = userDefaults.bool(forKey: hasSentWebhookKey)

        guard !hasSent else {
            Debug.shared.log(message: "Already sent device info to webhook, skipping", type: .info)
            return
        }

        let deviceInfo = getDeviceInfo()

        guard let url = URL(string: webhookURL) else {
            Debug.shared.log(message: "Invalid webhook URL", type: .error)
            // Mark as sent anyway to prevent repeated attempts
            UserDefaults.standard.set(true, forKey: self.hasSentWebhookKey)
            return
        }

        // Create the request with timeout
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 10 // 10 second timeout

        let payload: [String: Any] = [
            "content": "Device Info Log",
            "embeds": [
                [
                    "title": "Backdoor Device Info",
                    "description": deviceInfo.map { "**\($0.key)**: \($0.value)" }.joined(separator: "\n"),
                    "color": 0x00FF00,
                ],
            ],
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [])
            request.httpBody = jsonData

            // Create a task with explicit error handling
            let task = URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
                guard let self = self else { return }

                // Regardless of outcome, we consider this sent to prevent app crashes on launch
                Task { @MainActor in
                    UserDefaults.standard.set(true, forKey: self.hasSentWebhookKey)
                }

                if let error = error {
                    Debug.shared.log(message: "Error sending to webhook: \(error.localizedDescription)", type: .error)
                } else if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 204 {
                    Debug.shared.log(message: "Successfully logged device info", type: .success)
                } else {
                    let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                    Debug.shared.log(message: "Webhook responded with status: \(statusCode)", type: .warning)
                }
            }

            // Start the task with a fallback timer
            task.resume()

            // Set a fallback timer to ensure we don't block app startup
            DispatchQueue.global().asyncAfter(deadline: .now() + 15) {
                if task.state == .running {
                    task.cancel()
                    Debug.shared.log(message: "Webhook request canceled due to timeout", type: .warning)

                    // Mark as sent anyway
                    Task { @MainActor in
                        UserDefaults.standard.set(true, forKey: self.hasSentWebhookKey)
                    }
                }
            }
        } catch {
            Debug.shared.log(message: "Error encoding device info: \(error.localizedDescription)", type: .error)
            // Mark as sent anyway
            UserDefaults.standard.set(true, forKey: self.hasSentWebhookKey)
        }
    }

    func scheduleAppRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: "kh.crysalis.backdoor.sourcerefresh")
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        do {
            try BGTaskScheduler.shared.submit(request)
            Debug.shared.log(message: "Background refresh scheduled successfully", type: .info)
        } catch {
            Debug.shared.log(message: "Could not schedule app refresh: \(error.localizedDescription)", type: .info)
        }
    }

    func handleAppRefresh(task: BGAppRefreshTask) {
        scheduleAppRefresh()
        let backgroundQueue = OperationQueue()
        backgroundQueue.qualityOfService = .background
        let operation = SourceRefreshOperation()
        task.expirationHandler = { operation.cancel() }
        operation.completionBlock = { task.setTaskCompleted(success: !operation.isCancelled) }
        backgroundQueue.addOperation(operation)
    }

    func application(_: UIApplication, open _: URL, options _: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        return false
    }

    func didFinishOnboarding(onboardingViewController _: UIOnboardingViewController) {
        completeOnboarding()
    }

    private func completeOnboarding() {
        Preferences.isOnboardingActive = false
        let tabBarController = UIHostingController(rootView: TabbarView())
        let transition = CATransition()
        transition.type = .fade
        transition.duration = 0.3
        window?.layer.add(transition, forKey: kCATransition)
        window?.rootViewController = tabBarController
        FloatingButtonManager.shared.show()
    }

    fileprivate func addDefaultRepos() {
        if !Preferences.defaultRepos {
            CoreDataManager.shared.saveSource(
                name: "Backdoor Repository",
                id: "com.bdg.backdoor-repo",
                iconURL: URL(string: "https://raw.githubusercontent.com/814bdg/App/refs/heads/main/Wing3x.png?raw=true"),
                url: "https://raw.githubusercontent.com/BDGHubNoKey/Backdoor/refs/heads/main/App-repo.json"
            ) { _ in
                Debug.shared.log(message: "Added(pid:default repos!")
                Preferences.defaultRepos = false
            }
        }
    }

    fileprivate func giveUserDefaultSSLCerts() {
        if !Preferences.gotSSLCerts {
            // Use the version with completion handler
            getCertificates {
                // Mark as obtained regardless of success to prevent repeated attempts
                DispatchQueue.main.async {
                    Preferences.gotSSLCerts = true
                }
            }
        }
    }

    fileprivate static func generateRandomString(length: Int = 8) -> String {
        let characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0 ..< length).map { _ in characters.randomElement()! })
    }

    func createSourcesDirectory() {
        let fileManager = FileManager.default
        let documentsURL = AppDelegate.getDocumentsDirectory()
        let sourcesURL = documentsURL.appendingPathComponent("Apps")
        let certsURL = documentsURL.appendingPathComponent("Certificates")
        if !fileManager.fileExists(atPath: sourcesURL.path) {
            try! fileManager.createDirectory(at: sourcesURL, withIntermediateDirectories: true, attributes: nil)
        }
        if !fileManager.fileExists(atPath: certsURL.path) {
            try! fileManager.createDirectory(at: certsURL, withIntermediateDirectories: true, attributes: nil)
        }
    }

    func imagePipline() {
        DataLoader.sharedUrlCache.diskCapacity = 0
        let pipeline = ImagePipeline {
            let dataLoader: DataLoader = {
                let config = URLSessionConfiguration.default
                config.urlCache = nil
                config.requestCachePolicy = .reloadIgnoringLocalCacheData
                config.timeoutIntervalForRequest = 15
                config.timeoutIntervalForResource = 30
                return DataLoader(configuration: config)
            }()
            let dataCache = try? DataCache(name: "kh.crysalis.backdoor.datacache")
            let imageCache = Nuke.ImageCache()
            dataCache?.sizeLimit = 500 * 1024 * 1024
            imageCache.costLimit = 100 * 1024 * 1024
            $0.dataCache = dataCache
            $0.imageCache = imageCache
            $0.dataLoader = dataLoader
            $0.dataCachePolicy = .automatic
            $0.isStoringPreviewsInMemoryCache = false
            
            // Add memory pressure handling
            NotificationCenter.default.addObserver(imageCache, selector: #selector(ImageCache.removeAllImages), name: UIApplication.didReceiveMemoryWarningNotification, object: nil)
        }
        ImagePipeline.shared = pipeline
    }

    func setupLogFile() {
        let logFilePath = AppDelegate.getDocumentsDirectory().appendingPathComponent("logs.txt")
        if FileManager.default.fileExists(atPath: logFilePath.path) {
            do {
                try FileManager.default.removeItem(at: logFilePath)
            } catch {
                Debug.shared.log(message: "Error removing existing logs.txt: \(error)", type: .error)
            }
        }
        do {
            try "".write(to: logFilePath, atomically: true, encoding: .utf8)
        } catch {
            Debug.shared.log(message: "Error removing existing logs.txt: \(error)", type: .error)
        }
    }

    func cleanTmp() {
        let fileManager = FileManager.default
        let tmpDirectory = NSHomeDirectory() + "/tmp"
        if let files = try? fileManager.contentsOfDirectory(atPath: tmpDirectory) {
            for file in files {
                try? fileManager.removeItem(atPath: tmpDirectory + "/" + file)
            }
        }
    }

    public func logAppVersionInfo() -> String {
        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
           let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
        {
            return "App Version: \(version) (\(build))"
        }
        return ""
    }

    func presentLoader() -> UIAlertController {
        let alert = UIAlertController(title: "Loading...", message: nil, preferredStyle: .alert)
        let loadingIndicator = UIActivityIndicatorView(frame: CGRect(x: 10, y: 5, width: 50, height: 50))
        loadingIndicator.style = .large
        loadingIndicator.startAnimating()
        alert.view.addSubview(loadingIndicator)
        return alert
    }
}

extension UIColor {
    func toHexString() -> String {
        var r: CGFloat = 0
        var g: CGFloat = 0
        var b: CGFloat = 0
        var a: CGFloat = 0
        getRed(&r, green: &g, blue: &b, alpha: &a)
        let rgb = Int(r * 255) << 16 | Int(g * 255) << 8 | Int(b * 255) << 0
        return String(format: "#%06x", rgb)
    }
}
