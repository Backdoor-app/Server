// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Combine
import SwiftUI
import UIKit

/// Advanced theme manager for consistent UI styling and appearance
///
/// Provides:
/// 1. Comprehensive theming system with light/dark modes
/// 2. Dynamic color adjustments based on accessibility settings
/// 3. Animation capabilities for theme transitions
/// 4. Theme customization options
final class ThemeManager {
    // MARK: - Singleton
    
    /// Shared instance
    static let shared = ThemeManager()
    
    // MARK: - Properties
    
    /// Current theme
    @Published private(set) var currentTheme: AppTheme
    
    /// Current accessibility mode
    @Published private(set) var accessibilityMode: AccessibilityMode = .standard
    
    /// Theme transition animation duration
    private let transitionDuration: TimeInterval = 0.3
    
    /// Publishers cancellables
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    private init() {
        // Set up themes before initializing currentTheme
        let savedThemeId = UserDefaults.standard.string(forKey: "selectedThemeId") ?? ThemeType.system.rawValue
        
        // Initialize with system theme as a temporary value
        let systemThemeTemp = AppTheme(
            id: ThemeType.system.rawValue,
            name: "System",
            type: .system,
            primaryColor: .systemBlue,
            secondaryColor: .systemIndigo,
            accentColor: .systemOrange,
            backgroundColor: UIColor { $0.userInterfaceStyle == .dark ? .systemBackground : .systemBackground },
            cardColor: UIColor { $0.userInterfaceStyle == .dark ? .secondarySystemBackground : .secondarySystemBackground },
            textColor: UIColor { $0.userInterfaceStyle == .dark ? .label : .label },
            secondaryTextColor: UIColor { $0.userInterfaceStyle == .dark ? .secondaryLabel : .secondaryLabel }
        )
        
        // Initialize current theme
        currentTheme = systemThemeTemp
        
        // Now update with the correct theme after themes array is initialized
        if let selectedTheme = themes.first(where: { $0.id == savedThemeId }) {
            currentTheme = selectedTheme
        }
        
        // Load accessibility settings
        loadAccessibilitySettings()
        
        // Set up observers
        setupObservers()
        
        Debug.shared.log(message: "ThemeManager initialized with theme: \(currentTheme.name)", type: .info)
    }
    
    // MARK: - Setup
    
    private func setupObservers() {
        // Observe system appearance changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(systemAppearanceChanged),
            name: NSNotification.Name("UIUserInterfaceStyleChanged"),
            object: nil
        )
        
        // Observe accessibility settings changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(accessibilitySettingsChanged),
            name: UIAccessibility.darkerSystemColorsStatusDidChangeNotification,
            object: nil
        )
        
        // Observe UIAccessibility.isReduceTransparencyEnabled changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(accessibilitySettingsChanged),
            name: UIAccessibility.reduceTransparencyStatusDidChangeNotification,
            object: nil
        )
        
        // Observe UIAccessibility.isReduceMotionEnabled changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(accessibilitySettingsChanged),
            name: UIAccessibility.reduceMotionStatusDidChangeNotification,
            object: nil
        )
        
        // Observe UIAccessibility.isHighContrastEnabled changes
        if #available(iOS 14.0, *) {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(accessibilitySettingsChanged),
                name: UIAccessibility.invertColorsStatusDidChangeNotification,
                object: nil
            )
        }
    }
    
    @objc private func systemAppearanceChanged() {
        Debug.shared.log(message: "System appearance changed", type: .debug)
        
        // If using system theme, update to match system
        if currentTheme.type == .system {
            updateSystemTheme()
            notifyThemeChanged()
        }
    }
    
    @objc private func accessibilitySettingsChanged() {
        Debug.shared.log(message: "Accessibility settings changed", type: .debug)
        
        // Update accessibility mode based on system settings
        updateAccessibilityMode()
        
        // Notify accessibility changes
        notifyAccessibilityChanged()
    }
    
    // MARK: - Theme Management
    
    /// Available themes
    private(set) var themes: [AppTheme] = [
        // System theme (adapts to device settings)
        AppTheme(
            id: ThemeType.system.rawValue,
            name: "System",
            type: .system,
            primaryColor: .systemBlue,
            secondaryColor: .systemIndigo,
            accentColor: .systemOrange,
            backgroundColor: UIColor { $0.userInterfaceStyle == .dark ? .systemBackground : .systemBackground },
            cardColor: UIColor { $0.userInterfaceStyle == .dark ? .secondarySystemBackground : .secondarySystemBackground },
            textColor: UIColor { $0.userInterfaceStyle == .dark ? .label : .label },
            secondaryTextColor: UIColor { $0.userInterfaceStyle == .dark ? .secondaryLabel : .secondaryLabel }
        ),
        
        // Dark theme
        AppTheme(
            id: ThemeType.dark.rawValue,
            name: "Dark",
            type: .dark,
            primaryColor: UIColor(hex: "#0A84FF"),
            secondaryColor: UIColor(hex: "#5E5CE6"),
            accentColor: UIColor(hex: "#FF9F0A"),
            backgroundColor: UIColor(hex: "#1C1C1E"),
            cardColor: UIColor(hex: "#2C2C2E"),
            textColor: UIColor(hex: "#FFFFFF"),
            secondaryTextColor: UIColor(hex: "#EBEBF5").withAlphaComponent(0.6)
        ),
        
        // Light theme
        AppTheme(
            id: ThemeType.light.rawValue,
            name: "Light",
            type: .light,
            primaryColor: UIColor(hex: "#007AFF"),
            secondaryColor: UIColor(hex: "#5856D6"),
            accentColor: UIColor(hex: "#FF9500"),
            backgroundColor: UIColor(hex: "#F2F2F7"),
            cardColor: UIColor(hex: "#FFFFFF"),
            textColor: UIColor(hex: "#000000"),
            secondaryTextColor: UIColor(hex: "#3C3C43").withAlphaComponent(0.6)
        ),
        
        // High Contrast theme for accessibility
        AppTheme(
            id: ThemeType.highContrast.rawValue,
            name: "High Contrast",
            type: .highContrast,
            primaryColor: UIColor(hex: "#0040FF"),
            secondaryColor: UIColor(hex: "#7A04EB"),
            accentColor: UIColor(hex: "#FF5500"),
            backgroundColor: UIColor(hex: "#000000"),
            cardColor: UIColor(hex: "#1A1A1A"),
            textColor: UIColor(hex: "#FFFFFF"),
            secondaryTextColor: UIColor(hex: "#DDDDDD")
        ),
        
        // Blue theme
        AppTheme(
            id: ThemeType.blue.rawValue,
            name: "Blue",
            type: .blue,
            primaryColor: UIColor(hex: "#0096FF"),
            secondaryColor: UIColor(hex: "#6A7BFF"),
            accentColor: UIColor(hex: "#FFD60A"),
            backgroundColor: UIColor(hex: "#F0F7FF"),
            cardColor: UIColor(hex: "#FFFFFF"),
            textColor: UIColor(hex: "#000000"),
            secondaryTextColor: UIColor(hex: "#3C3C43").withAlphaComponent(0.6),
            darkModeColors: AppThemeColors(
                primaryColor: UIColor(hex: "#0096FF"),
                secondaryColor: UIColor(hex: "#6A7BFF"),
                accentColor: UIColor(hex: "#FFD60A"),
                backgroundColor: UIColor(hex: "#0A2C4A"),
                cardColor: UIColor(hex: "#153A5F"),
                textColor: UIColor(hex: "#FFFFFF"),
                secondaryTextColor: UIColor(hex: "#EBEBF5").withAlphaComponent(0.6)
            )
        ),
        
        // Green theme
        AppTheme(
            id: ThemeType.green.rawValue,
            name: "Green",
            type: .green,
            primaryColor: UIColor(hex: "#30D158"),
            secondaryColor: UIColor(hex: "#66D4CF"),
            accentColor: UIColor(hex: "#FF6482"),
            backgroundColor: UIColor(hex: "#F2FFF5"),
            cardColor: UIColor(hex: "#FFFFFF"),
            textColor: UIColor(hex: "#000000"),
            secondaryTextColor: UIColor(hex: "#3C3C43").withAlphaComponent(0.6),
            darkModeColors: AppThemeColors(
                primaryColor: UIColor(hex: "#30D158"),
                secondaryColor: UIColor(hex: "#66D4CF"),
                accentColor: UIColor(hex: "#FF6482"),
                backgroundColor: UIColor(hex: "#0A2E17"),
                cardColor: UIColor(hex: "#15462A"),
                textColor: UIColor(hex: "#FFFFFF"),
                secondaryTextColor: UIColor(hex: "#EBEBF5").withAlphaComponent(0.6)
            )
        )
    ]
    
    /// System theme reference
    private var systemTheme: AppTheme {
        themes.first { $0.type == .system }!
    }
    
    /// Updates the system theme based on current device settings
    private func updateSystemTheme() {
        // Use current trait collection's user interface style
        let isDarkMode = UITraitCollection.current.userInterfaceStyle == .dark
        
        // Create updated system theme
        let updatedSystemTheme = AppTheme(
            id: ThemeType.system.rawValue,
            name: "System",
            type: .system,
            primaryColor: .systemBlue,
            secondaryColor: .systemIndigo,
            accentColor: .systemOrange,
            backgroundColor: isDarkMode ? .systemBackground : .systemBackground,
            cardColor: isDarkMode ? .secondarySystemBackground : .secondarySystemBackground,
            textColor: isDarkMode ? .label : .label,
            secondaryTextColor: isDarkMode ? .secondaryLabel : .secondaryLabel
        )
        
        // Update system theme in themes array
        if let index = themes.firstIndex(where: { $0.type == .system }) {
            themes[index] = updatedSystemTheme
        }
        
        // Update current theme if using system
        if currentTheme.type == .system {
            currentTheme = updatedSystemTheme
        }
    }
    
    // MARK: - Theme Selection
    
    /// Changes the app theme
    /// - Parameter themeType: The type of theme to apply
    func setTheme(_ themeType: ThemeType) {
        guard let newTheme = themes.first(where: { $0.type == themeType }),
              newTheme.id != currentTheme.id else {
            return
        }
        
        Debug.shared.log(message: "Changing theme to: \(newTheme.name)", type: .info)
        
        // Set the new theme
        UserDefaults.standard.set(newTheme.id, forKey: "selectedThemeId")
        
        // Apply theme with animation
        UIView.animate(withDuration: transitionDuration) {
            self.currentTheme = newTheme
            self.applyThemeToApplication()
        }
        
        // Notify theme change
        notifyThemeChanged()
    }
    
    /// Applies the current theme to the entire application
    func applyThemeToApplication() {
        // Get the appropriate colors based on interface style
        let isDarkMode = UITraitCollection.current.userInterfaceStyle == .dark
        let colors = getThemeColorsForCurrentInterfaceStyle()
        
        // Set application-wide tint color
        UIApplication.shared.windows.forEach { window in
            window.tintColor = colors.primaryColor
        }
        
        // Update user preferences
        Preferences.appTintColor = CodableColor(colors.primaryColor)
        
        // Set UIAppearance defaults
        UINavigationBar.appearance().tintColor = colors.primaryColor
        UITabBar.appearance().tintColor = colors.primaryColor
        
        // Set UITextField appearance
        UITextField.appearance().textColor = colors.textColor
        
        // Set UIButton appearance for system buttons
        UIButton.appearance().tintColor = colors.primaryColor
        
        // Set specific appearance for different UI elements
        applyAppearanceToUIElements(with: colors)
        
        // Force UI update
        DispatchQueue.main.async {
            UIApplication.shared.windows.forEach { window in
                for view in window.subviews {
                    view.removeFromSuperview()
                    window.addSubview(view)
                }
            }
        }
    }
    
    /// Applies appearance to specific UI elements
    private func applyAppearanceToUIElements(with colors: AppThemeColors) {
        // UINavigationBar appearance
        if #available(iOS 15.0, *) {
            let navigationBarAppearance = UINavigationBarAppearance()
            navigationBarAppearance.configureWithOpaqueBackground()
            navigationBarAppearance.backgroundColor = colors.backgroundColor
            navigationBarAppearance.titleTextAttributes = [.foregroundColor: colors.textColor]
            navigationBarAppearance.largeTitleTextAttributes = [.foregroundColor: colors.textColor]
            
            UINavigationBar.appearance().standardAppearance = navigationBarAppearance
            UINavigationBar.appearance().compactAppearance = navigationBarAppearance
            UINavigationBar.appearance().scrollEdgeAppearance = navigationBarAppearance
        } else {
            UINavigationBar.appearance().barTintColor = colors.backgroundColor
            UINavigationBar.appearance().titleTextAttributes = [.foregroundColor: colors.textColor]
            UINavigationBar.appearance().largeTitleTextAttributes = [.foregroundColor: colors.textColor]
        }
        
        // UITabBar appearance
        if #available(iOS 15.0, *) {
            let tabBarAppearance = UITabBarAppearance()
            tabBarAppearance.configureWithOpaqueBackground()
            tabBarAppearance.backgroundColor = colors.backgroundColor
            
            UITabBar.appearance().standardAppearance = tabBarAppearance
            UITabBar.appearance().scrollEdgeAppearance = tabBarAppearance
        } else {
            UITabBar.appearance().barTintColor = colors.backgroundColor
        }
        
        // UITableView appearance
        UITableView.appearance().backgroundColor = colors.backgroundColor
        UITableViewCell.appearance().backgroundColor = colors.cardColor
        
        // UICollectionView appearance
        UICollectionView.appearance().backgroundColor = colors.backgroundColor
    }
    
    // MARK: - UI Element Color Getters
    
    /// Gets the appropriate primary color for the current context
    func primaryColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.primaryColor)
    }
    
    /// Gets the appropriate secondary color for the current context
    func secondaryColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.secondaryColor)
    }
    
    /// Gets the appropriate accent color for the current context
    func accentColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.accentColor)
    }
    
    /// Gets the appropriate background color for the current context
    func backgroundColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.backgroundColor)
    }
    
    /// Gets the appropriate card color for the current context
    func cardColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.cardColor)
    }
    
    /// Gets the appropriate text color for the current context
    func textColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.textColor)
    }
    
    /// Gets the appropriate secondary text color for the current context
    func secondaryTextColor(for traitCollection: UITraitCollection? = nil) -> UIColor {
        let colors = getThemeColorsForInterfaceStyle(
            traitCollection?.userInterfaceStyle ?? UITraitCollection.current.userInterfaceStyle
        )
        return getAccessibilityAdjustedColor(colors.secondaryTextColor)
    }
    
    // MARK: - Helper Methods
    
    /// Gets colors for the current interface style
    private func getThemeColorsForCurrentInterfaceStyle() -> AppThemeColors {
        getThemeColorsForInterfaceStyle(UITraitCollection.current.userInterfaceStyle)
    }
    
    /// Gets theme colors for a specific interface style
    private func getThemeColorsForInterfaceStyle(_ style: UIUserInterfaceStyle) -> AppThemeColors {
        let isDarkMode = style == .dark
        
        // If theme has specific dark mode colors and we're in dark mode, use those
        if isDarkMode, let darkModeColors = currentTheme.darkModeColors {
            return darkModeColors
        }
        
        // Otherwise, use the theme's default colors
        return AppThemeColors(
            primaryColor: currentTheme.primaryColor,
            secondaryColor: currentTheme.secondaryColor,
            accentColor: currentTheme.accentColor,
            backgroundColor: currentTheme.backgroundColor,
            cardColor: currentTheme.cardColor,
            textColor: currentTheme.textColor,
            secondaryTextColor: currentTheme.secondaryTextColor
        )
    }
    
    // MARK: - Accessibility Support
    
    /// Loads accessibility settings from the system
    private func loadAccessibilitySettings() {
        updateAccessibilityMode()
    }
    
    /// Updates the accessibility mode based on system settings
    private func updateAccessibilityMode() {
        if UIAccessibility.isInvertColorsEnabled {
            accessibilityMode = .invertColors
        } else if UIAccessibility.isDarkerSystemColorsEnabled {
            accessibilityMode = .highContrast
        } else if UIAccessibility.isReduceTransparencyEnabled {
            accessibilityMode = .reduceTransparency
        } else {
            accessibilityMode = .standard
        }
        
        Debug.shared.log(message: "Accessibility mode updated to: \(accessibilityMode)", type: .debug)
    }
    
    /// Adjusts a color based on current accessibility settings
    private func getAccessibilityAdjustedColor(_ color: UIColor) -> UIColor {
        var adjustedColor = color
        
        switch accessibilityMode {
            case .highContrast:
                // Increase contrast
                adjustedColor = increaseContrast(for: color)
                
            case .invertColors:
                // Let the system handle color inversion
                break
                
            case .reduceTransparency:
                // Ensure colors are fully opaque
                var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
                color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
                adjustedColor = UIColor(red: red, green: green, blue: blue, alpha: 1.0)
                
            case .standard:
                // No adjustments
                break
        }
        
        return adjustedColor
    }
    
    /// Increases contrast for a color
    private func increaseContrast(for color: UIColor) -> UIColor {
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        
        // Calculate luminance (perceived brightness)
        let luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        
        // If the color is dark, make it darker; if it's light, make it lighter
        if luminance < 0.5 {
            // Darken the color
            return UIColor(
                red: max(0, red - 0.2),
                green: max(0, green - 0.2),
                blue: max(0, blue - 0.2),
                alpha: alpha
            )
        } else {
            // Lighten the color
            return UIColor(
                red: min(1, red + 0.2),
                green: min(1, green + 0.2),
                blue: min(1, blue + 0.2),
                alpha: alpha
            )
        }
    }
    
    // MARK: - Notification Methods
    
    /// Notifies observers that the theme has changed
    private func notifyThemeChanged() {
        NotificationCenter.default.post(
            name: .themeDidChange,
            object: nil,
            userInfo: ["theme": currentTheme.id]
        )
    }
    
    /// Notifies observers that accessibility settings have changed
    private func notifyAccessibilityChanged() {
        NotificationCenter.default.post(
            name: .accessibilitySettingsDidChange,
            object: nil,
            userInfo: ["mode": accessibilityMode.rawValue]
        )
    }
}

// MARK: - Supporting Types

/// Theme types
enum ThemeType: String {
    case system
    case light
    case dark
    case highContrast
    case blue
    case green
}

/// Accessibility modes
enum AccessibilityMode: String {
    case standard
    case highContrast
    case invertColors
    case reduceTransparency
}

/// Theme data structure
struct AppTheme {
    let id: String
    let name: String
    let type: ThemeType
    
    let primaryColor: UIColor
    let secondaryColor: UIColor
    let accentColor: UIColor
    let backgroundColor: UIColor
    let cardColor: UIColor
    let textColor: UIColor
    let secondaryTextColor: UIColor
    
    var darkModeColors: AppThemeColors? = nil
}

/// Colors for a theme
struct AppThemeColors {
    let primaryColor: UIColor
    let secondaryColor: UIColor
    let accentColor: UIColor
    let backgroundColor: UIColor
    let cardColor: UIColor
    let textColor: UIColor
    let secondaryTextColor: UIColor
}

// MARK: - Extensions

extension Notification.Name {
    static let themeDidChange = Notification.Name("themeDidChange")
    static let accessibilitySettingsDidChange = Notification.Name("accessibilitySettingsDidChange")
}

// MARK: - SwiftUI Support

@available(iOS 14.0, *)
struct ThemeEnvironmentKey: EnvironmentKey {
    static let defaultValue: AppTheme = ThemeManager.shared.currentTheme
}

@available(iOS 14.0, *)
extension EnvironmentValues {
    var appTheme: AppTheme {
        get { self[ThemeEnvironmentKey.self] }
        set { self[ThemeEnvironmentKey.self] = newValue }
    }
}

// MARK: - View Controller Extension for Theme Support

extension UIViewController {
    /// Applies the current theme to the view controller
    func applyTheme() {
        let theme = ThemeManager.shared
        _ = traitCollection.userInterfaceStyle
        
        // Apply theme colors
        view.backgroundColor = theme.backgroundColor(for: traitCollection)
        
        // Apply to navigation items
        navigationController?.navigationBar.tintColor = theme.primaryColor(for: traitCollection)
        
        // Apply to child view controllers
        children.forEach { $0.applyTheme() }
    }
}
