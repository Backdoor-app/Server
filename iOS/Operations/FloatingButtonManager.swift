// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import SwiftUI
import UIKit

// MARK: - Notification Names

extension Notification.Name {
    static let showAIAssistant = Notification.Name("showAIAssistant")
    // Tab change notifications are defined in TabbarView.swift
}

/// Manages the floating AI button across the app
final class FloatingButtonManager {
    // Singleton instance
    static let shared = FloatingButtonManager()

    // UI components - using a direct approach rather than separate window
    private let floatingButton = FloatingAIButton()

    // Thread-safe state tracking with a dedicated queue
    private let stateQueue = DispatchQueue(label: "com.backdoor.floatingButtonState", qos: .userInteractive)
    private var _isPresentingChat = false
    private var isPresentingChat: Bool {
        get { stateQueue.sync { _isPresentingChat } }
        set { stateQueue.sync { _isPresentingChat = newValue } }
    }

    // Thread-safe setup state
    private var _isSetUp = false
    private var isSetUp: Bool {
        get { stateQueue.sync { _isSetUp } }
        set { stateQueue.sync { _isSetUp = newValue } }
    }

    // Weak references to parent views
    private weak var parentViewController: UIViewController?
    private weak var parentView: UIView?

    // Recovery counter to prevent excessive retries
    private var _recoveryAttempts = 0
    private var recoveryAttempts: Int {
        get { stateQueue.sync { _recoveryAttempts } }
        set { stateQueue.sync { _recoveryAttempts = newValue } }
    }

    private let maxRecoveryAttempts = 3

    // Processing queue for handling asynchronous tasks
    private let processingQueue = DispatchQueue(label: "com.backdoor.floatingButtonProcessing", qos: .userInitiated)

    // Monitor whether app is in active state
    private var isAppActive = true

    private init() {
        // Log initialization
        Debug.shared.log(message: "FloatingButtonManager initialized", type: .info)

        // Configure the floating button
        configureFloatingButton()

        // Set up notification observers
        setupObservers()

        // Set up the AI interaction
        setupAIInteraction()
    }

    deinit {
        // Clean up observers to prevent memory leaks
        NotificationCenter.default.removeObserver(self)
        Debug.shared.log(message: "FloatingButtonManager deinit", type: .debug)
    }

    private func configureFloatingButton() {
        // Configure the button's appearance
        floatingButton.layer.zPosition = 999 // Ensure it's above other views
        floatingButton.isUserInteractionEnabled = true

        // Add a shadow for better visibility
        floatingButton.layer.shadowColor = UIColor.black.cgColor
        floatingButton.layer.shadowOpacity = 0.3
        floatingButton.layer.shadowOffset = CGSize(width: 0, height: 2)
        floatingButton.layer.shadowRadius = 4
    }

    private func setupObservers() {
        // Use processingQueue to ensure thread safety when setting up observers
        processingQueue.async { [weak self] in
            guard let self = self else { return }

            // Observe orientation changes
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleOrientationChange),
                name: UIDevice.orientationDidChangeNotification,
                object: nil
            )

            // Observe interface style changes
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.updateButtonAppearance),
                name: NSNotification.Name("UIInterfaceStyleChanged"),
                object: nil
            )

            // Listen for button taps
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleAIRequest),
                name: .showAIAssistant,
                object: nil
            )

            // Listen for app lifecycle events
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleAppDidBecomeActive),
                name: UIApplication.didBecomeActiveNotification,
                object: nil
            )

            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleAppWillResignActive),
                name: UIApplication.willResignActiveNotification,
                object: nil
            )

            // Listen for tab changes - observe both notification names for consistency
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleTabChange),
                name: Notification.Name("changeTab"),
                object: nil
            )

            NotificationCenter.default.addObserver(
                self,
                selector: #selector(self.handleTabChange),
                name: Notification.Name("tabDidChange"),
                object: nil
            )
        }
    }

    @objc private func handleTabChange(_: Notification) {
        // When tab changes, we need to reattach the button to the new view controller
        // First check if app is active to avoid unnecessary work
        guard isAppActive else {
            Debug.shared.log(message: "Tab change ignored - app inactive", type: .debug)
            return
        }

        // Wait a moment to ensure the tab change is complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self else { return }

            // Only proceed if we're not currently showing the chat
            if !self.isPresentingChat {
                self.processingQueue.async {
                    // Reset recovery attempts and try to reattach
                    self.recoveryAttempts = 0

                    DispatchQueue.main.async {
                        self.attachToRootView()
                    }
                }
            }
        }
    }

    private func attachToRootView() {
        // This method should only be called on the main thread
        dispatchPrecondition(condition: .onQueue(.main))

        // Check if we're already in the process of attaching
        guard !isPresentingChat else {
            Debug.shared.log(message: "Skipping button attach - chat is presenting", type: .debug)
            return
        }

        // Find the top view controller to add the button to
        guard let rootVC = UIApplication.shared.topMostViewController() else {
            Debug.shared.log(message: "No root view controller found", type: .error)
            return
        }

        // Check if the view controller is in a valid state
        guard !rootVC.isBeingDismissed, !rootVC.isBeingPresented,
              rootVC.view.window != nil, rootVC.isViewLoaded
        else {
            Debug.shared.log(message: "View controller in invalid state for button attachment", type: .warning)

            // Retry after a short delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.attachToRootView()
            }
            return
        }

        // Clean up any existing button
        floatingButton.removeFromSuperview()

        // Remember the parent for later repositioning
        parentViewController = rootVC
        parentView = rootVC.view

        // Set the frame size (but not position yet)
        floatingButton.frame = CGRect(x: 0, y: 0, width: 60, height: 60)

        // Add to view first so didMoveToSuperview can access safe areas
        rootVC.view.addSubview(floatingButton)

        // At this point, the didMoveToSuperview method in FloatingAIButton will have
        // attempted to restore the saved position. We'll give that a moment to apply.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self, weak rootVC] in
            guard let self = self, let rootVC = rootVC else { return }

            // Ensure the button is in a valid position within the safe area
            let safeArea = rootVC.view.safeAreaInsets
            let minX = 30 + safeArea.left
            let maxX = rootVC.view.bounds.width - 30 - safeArea.right
            let minY = 30 + safeArea.top
            let maxY = rootVC.view.bounds.height - 30 - safeArea.bottom

            // Adjust position if outside safe bounds
            let currentCenter = self.floatingButton.center
            let xPos = min(max(currentCenter.x, minX), maxX)
            let yPos = min(max(currentCenter.y, minY), maxY)

            // Animate to new position if needed
            if xPos != currentCenter.x || yPos != currentCenter.y {
                UIView.animate(withDuration: 0.3) {
                    self.floatingButton.center = CGPoint(x: xPos, y: yPos)
                }
            }

            Debug.shared.log(message: "Ensured floating button is within bounds at \(self.floatingButton.center)", type: .debug)
        }

        // Mark setup as complete
        isSetUp = true
        recoveryAttempts = 0

        Debug.shared.log(message: "Floating button attached to root view", type: .info)
    }

    @objc private func handleOrientationChange() {
        // Ensure we're on the main thread for UI updates
        DispatchQueue.main.async { [weak self] in
            self?.updateButtonPosition()
        }
    }

    private func updateButtonPosition() {
        // This method should only be called on the main thread
        dispatchPrecondition(condition: .onQueue(.main))

        // Skip if button is not visible or app is inactive
        guard !floatingButton.isHidden, isAppActive else { return }

        // Verify the parent view controller is still valid
        guard let parentVC = parentViewController, parentVC.view.window != nil,
              !parentVC.isBeingDismissed, !parentVC.isBeingPresented
        else {
            // View is not in window hierarchy, try to re-attach
            if recoveryAttempts < maxRecoveryAttempts {
                recoveryAttempts += 1
                Debug.shared.log(message: "Trying to recover floating button (attempt \(recoveryAttempts))", type: .warning)

                // Attempt to reattach
                attachToRootView()
            }
            return
        }

        // Reset recovery counter since we have a valid parent
        recoveryAttempts = 0

        // Update position based on current orientation and safe area
        let safeArea = parentVC.view.safeAreaInsets
        let maxX = parentVC.view.bounds.width - 40 - safeArea.right
        let maxY = parentVC.view.bounds.height - 100 - safeArea.bottom

        UIView.animate(withDuration: 0.3) {
            self.floatingButton.center = CGPoint(x: maxX, y: maxY)
        }
    }

    @objc private func handleAppDidBecomeActive() {
        // Mark app as active
        isAppActive = true

        // Ensure the button is shown when app becomes active, with a slight delay
        // to allow view hierarchy to stabilize
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self = self else { return }

            if !self.isPresentingChat {
                self.show()
            }
        }
    }

    @objc private func handleAppWillResignActive() {
        // Mark app as inactive
        isAppActive = false

        // Hide button when app goes inactive to avoid overlay issues
        hide()
    }

    @objc private func updateButtonAppearance() {
        // Update the floating button appearance when theme changes
        DispatchQueue.main.async { [weak self] in
            self?.floatingButton.updateAppearance()
        }
    }

    /// Show the floating button
    func show() {
        // Ensure we're on the main thread for UI updates
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            // Don't show if we're presenting the chat
            if self.isPresentingChat {
                return
            }

            // Make button visible
            self.floatingButton.isHidden = false

            // Attach to root view if needed
            if !self.isSetUp || self.parentView?.window == nil {
                self.attachToRootView()
            } else {
                // Otherwise just update position
                self.updateButtonPosition()
            }
        }
    }

    /// Hide the floating button
    func hide() {
        // Ensure we're on the main thread for UI updates
        DispatchQueue.main.async { [weak self] in
            self?.floatingButton.isHidden = true
        }
    }

    private func setupAIInteraction() {
        // Register app commands for the AI assistant
        // These are executed on the processing queue for thread safety

        // Command: add source
        AppContextManager.shared.registerCommand("add source") { [weak self] sourceURL, completion in
            self?.processingQueue.async {
                guard let _ = URL(string: sourceURL) else {
                    Debug.shared.log(message: "Invalid source URL: \(sourceURL)", type: .error)
                    completion("Invalid source URL")
                    return
                }
                CoreDataManager.shared.saveSource(name: "Custom Source", id: UUID().uuidString, iconURL: nil, url: sourceURL) { error in
                    if let error = error {
                        Debug.shared.log(message: "Failed to add source: \(error)", type: .error)
                        completion("Failed to add source: \(error.localizedDescription)")
                    } else {
                        Debug.shared.log(message: "Added source: \(sourceURL)", type: .success)
                        completion("Source added successfully")
                    }
                }
            }
        }

        // Command: list sources
        AppContextManager.shared.registerCommand("list sources") { [weak self] _, completion in
            self?.processingQueue.async {
                let sources = CoreDataManager.shared.getAZSources()
                let sourceNames = sources.map { $0.name ?? "Unnamed" }.joined(separator: "\n")
                completion(sourceNames.isEmpty ? "No sources available" : sourceNames)
            }
        }

        // Command: list downloaded apps
        AppContextManager.shared.registerCommand("list downloaded apps") { [weak self] _, completion in
            self?.processingQueue.async {
                let apps = CoreDataManager.shared.getDatedDownloadedApps()
                let appNames = apps.map { "\($0.name ?? "Unnamed") (\($0.version ?? "Unknown"))" }.joined(separator: "\n")
                completion(appNames.isEmpty ? "No downloaded apps" : appNames)
            }
        }

        // Command: list signed apps
        AppContextManager.shared.registerCommand("list signed apps") { [weak self] _, completion in
            self?.processingQueue.async {
                let apps = CoreDataManager.shared.getDatedSignedApps()
                let appNames = apps.map { "\($0.name ?? "Unnamed") (\($0.bundleidentifier ?? "Unknown"))" }.joined(separator: "\n")
                completion(appNames.isEmpty ? "No signed apps" : appNames)
            }
        }

        // Command: list certificates
        AppContextManager.shared.registerCommand("list certificates") { [weak self] _, completion in
            self?.processingQueue.async {
                let certificates = CoreDataManager.shared.getDatedCertificate()
                let certNames = certificates.map { $0.certData?.name ?? "Unnamed" }.joined(separator: "\n")
                completion(certNames.isEmpty ? "No certificates" : certNames)
            }
        }

        // Command: navigate to
        AppContextManager.shared.registerCommand("navigate to") { screen, completion in
            DispatchQueue.main.async {
                guard let _ = UIApplication.shared.topMostViewController() as? UIHostingController<TabbarView> else {
                    Debug.shared.log(message: "Cannot navigate: Not on main tab bar", type: .error)
                    completion("Cannot navigate: Not on main screen")
                    return
                }

                var targetTab: String
                switch screen.lowercased() {
                    case "home":
                        targetTab = "home"
                    case "sources":
                        targetTab = "sources"
                    case "library":
                        targetTab = "library"
                    case "settings":
                        targetTab = "settings"
                    case "bdg hub", "bdghub", "hub":
                        targetTab = "bdgHub"
                    default:
                        Debug.shared.log(message: "Unknown screen: \(screen)", type: .warning)
                        completion("Unknown screen: \(screen)")
                        return
                }

                UserDefaults.standard.set(targetTab, forKey: "selectedTab")

                // Post to both notification names for maximum compatibility
                NotificationCenter.default.post(name: Notification.Name("changeTab"), object: nil, userInfo: ["tab": targetTab])
                NotificationCenter.default.post(name: Notification.Name("tabDidChange"), object: nil, userInfo: ["tab": targetTab])

                completion("Navigated to \(screen)")
            }
        }

        // Command: system info
        AppContextManager.shared.registerCommand("system info") { _, completion in
            let device = UIDevice.current
            let info = """
            Device: \(device.name) (\(device.model))
            iOS: \(device.systemVersion)
            App: Backdoor \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown")
            """
            completion(info)
        }

        // Command: refresh context
        AppContextManager.shared.registerCommand("refresh context") { [weak self] _, completion in
            self?.processingQueue.async {
                if let topVC = UIApplication.shared.topMostViewController() {
                    AppContextManager.shared.updateContext(topVC)
                    CustomAIContextProvider.shared.refreshContext()
                    completion("Context refreshed successfully")
                } else {
                    completion("Failed to refresh context: Could not determine current screen")
                }
            }
        }
    }

    @objc private func handleAIRequest() {
        // We need to ensure we're always on the main thread for UI operations
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in
                self?.handleAIRequest()
            }
            return
        }

        // Prevent multiple presentations with thread-safe check
        if isPresentingChat {
            Debug.shared.log(message: "Already presenting chat, ignoring request", type: .warning)
            return
        }

        // Set flag to prevent multiple presentations - do this immediately
        isPresentingChat = true

        // Provide haptic feedback to confirm button tap
        let feedback = UIImpactFeedbackGenerator(style: .medium)
        feedback.impactOccurred()

        // Hide the floating button immediately while preparing chat
        hide()

        // Find the top view controller on which to present the chat
        guard let topVC = UIApplication.shared.topMostViewController() else {
            Debug.shared.log(message: "Could not find top view controller to present chat", type: .error)
            isPresentingChat = false
            show() // Show the button again
            return
        }

        // Verify the view controller is in a valid state to present
        if topVC.isBeingDismissed || topVC.isBeingPresented || topVC.isMovingFromParent || topVC.isMovingToParent {
            Debug.shared.log(message: "View controller is in transition, delaying chat presentation", type: .warning)

            // Delay and retry once the transition completes
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.isPresentingChat = false
                self?.handleAIRequest()
            }
            return
        }

        // Prepare chat data on background queue
        processingQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                // Update AI context
                AppContextManager.shared.updateContext(topVC)
                CustomAIContextProvider.shared.refreshContext()

                // Create a new chat session
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                dateFormatter.timeStyle = .short
                let timestamp = dateFormatter.string(from: Date())
                let title = "Chat on \(timestamp)"

                // Create the session
                let session = try CoreDataManager.shared.createAIChatSession(title: title)

                // Present the UI on the main thread
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.presentChatInterfaceSafely(with: session, from: topVC)
                }
            } catch {
                Debug.shared.log(message: "Failed to create chat session: \(error.localizedDescription)", type: .error)

                // Reset state and show UI feedback on main thread
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.isPresentingChat = false
                    self.show() // Show the button again

                    // Show error alert
                    self.showErrorAlert(message: "Chat initialization failed. Please try again later.", on: topVC)
                }
            }
        }
    }

    private func presentChatInterfaceSafely(with session: ChatSession, from presenter: UIViewController) {
        // Create chat view controller with the session
        let chatVC = ChatViewController(session: session)

        // Ensure we have a valid dismissal handler
        chatVC.dismissHandler = { [weak self] in
            DispatchQueue.main.async {
                self?.isPresentingChat = false
                self?.show() // Show the button again
            }
        }

        // Wrap in navigation controller for better presentation
        let navController = UINavigationController(rootViewController: chatVC)

        // Configure presentation style based on device
        if UIDevice.current.userInterfaceIdiom == .pad {
            // iPad-specific presentation
            navController.modalPresentationStyle = .formSheet
            navController.preferredContentSize = CGSize(width: 540, height: 620)
        } else {
            // iPhone presentation
            if #available(iOS 15.0, *) {
                if let sheet = navController.sheetPresentationController {
                    // Use sheet presentation for iOS 15+
                    sheet.detents = [.medium(), .large()]
                    sheet.prefersGrabberVisible = true
                    sheet.preferredCornerRadius = 24

                    // Add delegate to handle dismissal properly
                    sheet.delegate = chatVC as? UISheetPresentationControllerDelegate
                }
            } else {
                // Fallback for older iOS versions
                navController.modalPresentationStyle = .fullScreen
            }
        }

        // Ensure safe presentation
        self.presentViewControllerSafely(navController, from: presenter)
    }

    private func presentViewControllerSafely(_ viewController: UIViewController, from presenter: UIViewController) {
        // Check if presenter is valid - if not, reset state and return
        guard !presenter.isBeingDismissed, !presenter.isBeingPresented, presenter.view.window != nil else {
            Debug.shared.log(message: "Presenter view controller is in invalid state for presentation", type: .error)
            isPresentingChat = false
            show()
            return
        }

        // Handle pending dismissals of any currently presented view controller
        if let presentedVC = presenter.presentedViewController {
            // If there's already a presented VC, dismiss it first
            presentedVC.dismiss(animated: true) { [weak self, weak presenter, weak viewController] in
                guard let self = self,
                      let presenter = presenter,
                      let viewController = viewController,
                      !presenter.isBeingDismissed
                else {
                    self?.isPresentingChat = false
                    self?.show()
                    return
                }

                // Now present the chat interface
                self.performPresentation(viewController, from: presenter)
            }
        } else {
            // No existing presentation, present directly
            performPresentation(viewController, from: presenter)
        }
    }

    private func performPresentation(_ viewController: UIViewController, from presenter: UIViewController) {
        // Add a try-catch for presentation failures
        do {
            presenter.present(viewController, animated: true) { [weak self] in
                // Log success
                Debug.shared.log(message: "AI assistant presented successfully", type: .info)
            }
        } catch {
            // If presentation fails for any reason, reset state
            Debug.shared.log(message: "Failed to present AI assistant: \(error.localizedDescription)", type: .error)
            isPresentingChat = false
            show()
        }
    }

    private func showErrorAlert(message: String, on viewController: UIViewController) {
        let alert = UIAlertController(
            title: "Chat Error",
            message: message,
            preferredStyle: .alert
        )

        alert.addAction(UIAlertAction(title: "OK", style: .default) { [weak self] _ in
            // Ensure button is shown after alert dismissal
            self?.show()
        })

        // Present alert with a slight delay to ensure any pending transitions complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            if viewController.presentedViewController == nil, !viewController.isBeingDismissed {
                viewController.present(alert, animated: true)
            } else {
                // If we can't present, at least log the error
                Debug.shared.log(message: "Could not present error alert: \(message)", type: .error)
                self.isPresentingChat = false
                self.show()
            }
        }
    }
}
