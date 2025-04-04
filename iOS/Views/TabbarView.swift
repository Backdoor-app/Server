// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import SwiftUI

/// Extension to define notification names for tab-related events
extension Notification.Name {
    static let tabDidChange = Notification.Name("tabDidChange")
    static let changeTab = Notification.Name("changeTab")
}

/// Main TabView providing navigation between app sections
struct TabbarView: View {
    // State for the selected tab, initialized from UserDefaults
    @State private var selectedTab: Tab = .init(rawValue: UserDefaults.standard.string(forKey: "selectedTab") ?? "home") ?? .home

    // Track if a programmatic tab change is in progress to avoid notification loops
    @State private var isProgrammaticTabChange = false

    // Tab identifiers
    enum Tab: String, CaseIterable, Identifiable {
        case home
        case sources
        case library
        case settings
        case bdgHub

        var id: String { self.rawValue }

        var displayName: String {
            switch self {
                case .home: return String.localized("TAB_HOME")
                case .sources: return String.localized("TAB_SOURCES")
                case .library: return String.localized("TAB_LIBRARY")
                case .settings: return String.localized("TAB_SETTINGS")
                case .bdgHub: return "BDG HUB"
            }
        }

        var iconName: String {
            switch self {
                case .home: return "house.fill"
                case .sources:
                    if #available(iOS 16.0, *) {
                        return "globe.desk.fill"
                    } else {
                        return "books.vertical.fill"
                    }
                case .library: return "square.grid.2x2.fill"
                case .settings: return "gearshape.2.fill"
                case .bdgHub: return "star.fill"
            }
        }
    }

    // Initialize with notification observer for tab changes
    init() {
        // No direct notification observer setup here - moved to .onReceive
        Debug.shared.log(message: "TabbarView initialized", type: .debug)
    }

    // Handle tab change notification from other parts of the app
    private func handleTabChangeNotification(_ notification: Notification) {
        if let newTab = notification.userInfo?["tab"] as? String,
           let tab = Tab(rawValue: newTab)
        {
            // Set the flag to prevent duplicate notifications
            isProgrammaticTabChange = true

            // Update tab with animation on the main thread
            DispatchQueue.main.async {
                withAnimation(.easeInOut(duration: 0.3)) {
                    selectedTab = tab
                }

                // Save selection to UserDefaults
                UserDefaults.standard.set(tab.rawValue, forKey: "selectedTab")
                UserDefaults.standard.synchronize()

                // Reset the flag with a slight delay to allow animations to complete
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    isProgrammaticTabChange = false

                    // Notify that tab change is complete
                    NotificationCenter.default.post(
                        name: .tabDidChange,
                        object: nil,
                        userInfo: ["tab": tab.rawValue]
                    )
                }
            }
        }
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            ForEach(Tab.allCases, id: \.self) { tabCase in
                tabView(for: tabCase)
                    .tag(tabCase)
            }
        }
        // Handle tab change notifications
        .onReceive(NotificationCenter.default.publisher(for: .changeTab)) { notification in
            handleTabChangeNotification(notification)
        }
        // Handle user-initiated tab changes
        .onChange(of: selectedTab) { newTab in
            // Only handle if not a programmatic change to avoid loops
            if !isProgrammaticTabChange {
                // Save the selected tab to UserDefaults
                UserDefaults.standard.set(newTab.rawValue, forKey: "selectedTab")
                UserDefaults.standard.synchronize()

                // Log the tab change
                Debug.shared.log(message: "User changed tab to: \(newTab.rawValue)", type: .debug)

                // Trigger animation for tab change
                withAnimation(.spring(response: 0.4, dampingFraction: 0.7, blendDuration: 0.4)) {
                    // Provide feedback for tab change
                    let generator = UIImpactFeedbackGenerator(style: .light)
                    generator.impactOccurred()

                    // Notify that tab has changed (for other components to react)
                    NotificationCenter.default.post(
                        name: .tabDidChange,
                        object: nil,
                        userInfo: ["tab": newTab.rawValue]
                    )
                }
            }
        }
        .onAppear {
            // Ensure the app is responsive on appear
            if let topVC = UIApplication.shared.topMostViewController() {
                topVC.view.isUserInteractionEnabled = true

                // Log the initial tab
                Debug.shared.log(message: "TabbarView appeared with tab: \(selectedTab.rawValue)", type: .debug)
            }
        }
    }

    @ViewBuilder
    private func tabView(for tab: Tab) -> some View {
        switch tab {
            case .home:
                createTab(
                    viewController: HomeViewController.self,
                    title: tab.displayName,
                    imageName: tab.iconName
                )
            case .sources:
                createTab(
                    viewController: SourcesViewController.self,
                    title: tab.displayName,
                    imageName: tab.iconName
                )
            case .library:
                createTab(
                    viewController: LibraryViewController.self,
                    title: tab.displayName,
                    imageName: tab.iconName
                )
            case .settings:
                createTab(
                    viewController: SettingsViewController.self,
                    title: tab.displayName,
                    imageName: tab.iconName
                )
            case .bdgHub:
                createTab(
                    viewController: WebViewController.self,
                    title: tab.displayName,
                    imageName: tab.iconName
                )
        }
    }

    @ViewBuilder
    private func createTab<T: UIViewController>(
        viewController: T.Type,
        title: String,
        imageName: String
    ) -> some View {
        NavigationViewController(viewController, title: title)
            .edgesIgnoringSafeArea(.all)
            .tabItem {
                Label(title, systemImage: imageName)
            }
    }
}

/// SwiftUI wrapper for UIKit view controllers with improved lifecycle management
struct NavigationViewController<Content: UIViewController>: UIViewControllerRepresentable {
    let content: Content.Type
    let title: String

    // Coordinator to maintain controller references and prevent premature deallocations
    class Coordinator {
        var viewController: UIViewController?
    }

    init(_ content: Content.Type, title: String) {
        self.content = content
        self.title = title
    }

    func makeCoordinator() -> Coordinator {
        return Coordinator()
    }

    func makeUIViewController(context: Context) -> UINavigationController {
        // Create view controller
        let viewController = content.init()
        context.coordinator.viewController = viewController

        // Configure view controller
        viewController.navigationItem.title = title

        // Ensure user interaction is enabled
        viewController.view.isUserInteractionEnabled = true

        // Create navigation controller
        let navController = UINavigationController(rootViewController: viewController)

        // Ensure navigation controller is interactive
        navController.view.isUserInteractionEnabled = true

        // Ensure the controller is properly initialized
        viewController.view.setNeedsLayout()
        viewController.view.layoutIfNeeded()

        // Log successful creation
        Debug.shared.log(message: "Created navigation controller for \(String(describing: content))", type: .debug)

        return navController
    }

    func updateUIViewController(_ uiViewController: UINavigationController, context _: Context) {
        // Ensure the view controller remains responsive
        uiViewController.view.isUserInteractionEnabled = true

        // Update top view controller's properties if needed
        if let topVC = uiViewController.topViewController {
            topVC.view.isUserInteractionEnabled = true

            // Update title if changed
            if topVC.navigationItem.title != title {
                topVC.navigationItem.title = title
            }

            // If the view controller supports content refreshing, refresh it
            // Check if the view is loaded and visible first to avoid unnecessary work
            if topVC.isViewLoaded && topVC.view.window != nil {
                topVC.refreshContent() 
            }
        }
    }
}

/// Protocol for view controllers that can refresh their content during tab switches
protocol ViewControllerRefreshable {
    func refreshContent()
}

/// Default implementation for all UIViewControllers
extension UIViewController: ViewControllerRefreshable {
    @objc func refreshContent() {
        // Default implementation does nothing
        // Subclasses can override this to refresh their content when tabs switch
    }
}
