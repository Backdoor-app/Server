// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// SafeModeLauncher - Detects and recovers from repeated app crashes by providing a minimal safe mode
class SafeModeLauncher {
    static let shared = SafeModeLauncher()
    
    // Keys for UserDefaults
    private let launchAttemptsKey = "launchAttempts"
    private let safeModeFlagKey = "inSafeMode"
    private let maxLaunchAttempts = 3
    private var launchSuccessMarked = false
    
    /// Whether the app is currently in safe mode
    var inSafeMode: Bool {
        return UserDefaults.standard.bool(forKey: safeModeFlagKey)
    }
    
    /// Record a launch attempt and enter safe mode if there have been too many failures
    func recordLaunchAttempt() {
        let launchAttempts = UserDefaults.standard.integer(forKey: launchAttemptsKey) + 1
        UserDefaults.standard.set(launchAttempts, forKey: launchAttemptsKey)
        UserDefaults.standard.synchronize()
        
        if launchAttempts >= maxLaunchAttempts {
            enableSafeMode()
        }
        
        print("📱 App launch attempt #\(launchAttempts) recorded")
    }
    
    /// Mark the launch as successful, resetting the launch attempts counter
    func markLaunchSuccessful() {
        launchSuccessMarked = true
        
        // Reset counter after successful launch with a delay to ensure stability
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            guard let self = self, self.launchSuccessMarked == true else { return }
            
            // Use a constant default to avoid empty string keys if self is nil
            let key = self.launchAttemptsKey
            UserDefaults.standard.set(0, forKey: key)
            UserDefaults.standard.synchronize()
            print("✅ App launch marked as successful, counter reset")
        }
    }
    
    /// Enable safe mode - limiting functionality to ensure stability
    func enableSafeMode() {
        UserDefaults.standard.set(true, forKey: safeModeFlagKey)
        
        // Disable potentially problematic features
        UserDefaults.standard.set(false, forKey: "AILearningEnabled")
        UserDefaults.standard.set(false, forKey: "AIServerSyncEnabled")
        UserDefaults.standard.synchronize()
        
        print("⚠️ SAFE MODE ENABLED - Limited functionality")
    }
    
    /// Disable safe mode and reset launch attempts
    func disableSafeMode() {
        UserDefaults.standard.set(false, forKey: safeModeFlagKey)
        UserDefaults.standard.set(0, forKey: launchAttemptsKey)
        UserDefaults.standard.synchronize()
        
        print("🔄 Safe mode disabled, app will restart with full functionality")
    }
    
    /// Present a safe mode alert to inform the user
    func showSafeModeAlert(on viewController: UIViewController, completion: (() -> Void)? = nil) {
        let alert = UIAlertController(
            title: "Safe Mode Activated",
            message: "The app has been started in safe mode due to previous crashes. Advanced features are disabled for stability.",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "Continue", style: .default) { _ in
            completion?()
        })
        
        alert.addAction(UIAlertAction(title: "Exit Safe Mode", style: .destructive) { [weak self] _ in
            self?.promptForAppRestart(on: viewController)
        })
        
        viewController.present(alert, animated: true)
    }
    
    /// Prompt the user to restart the app after exiting safe mode
    private func promptForAppRestart(on viewController: UIViewController) {
        let alert = UIAlertController(
            title: "Restart Required",
            message: "The app needs to restart to exit safe mode. Do you want to restart now?",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "Restart Now", style: .destructive) { [weak self] _ in
            self?.disableSafeMode()
            // Properly terminate the app instead of using exit(0)
            UIApplication.shared.perform(#selector(NSXPCConnection.suspend))
        })
        
        alert.addAction(UIAlertAction(title: "Later", style: .cancel))
        
        viewController.present(alert, animated: true)
    }
}
