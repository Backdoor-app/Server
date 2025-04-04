// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import SystemConfiguration
import Network

// MARK: - Network Monitoring Extension
extension AppDelegate {
    /// Set up network monitoring to track connectivity changes
    func setupNetworkMonitoring() {
        Debug.shared.log(message: "Setting up network monitoring", type: .info)
        
        if #available(iOS 12.0, *) {
            // Use NWPathMonitor for iOS 12 and above
            setupModernNetworkMonitoring()
        } else {
            // Use older approach for earlier iOS versions
            setupLegacyNetworkMonitoring()
        }
    }
    
    @available(iOS 12.0, *)
    private func setupModernNetworkMonitoring() {
        let monitor = NWPathMonitor()
        
        monitor.pathUpdateHandler = { [weak self] path in
            let isConnected = path.status == .satisfied
            let connectionType = self?.getConnectionTypeString(path) ?? "Unknown"
            
            Debug.shared.log(message: "Network status changed - Connected: \(isConnected), Type: \(connectionType)", type: .info)
            
            DispatchQueue.main.async {
                // Post notification about network status change
                NotificationCenter.default.post(
                    name: Notification.Name("NetworkStatusChanged"),
                    object: nil,
                    userInfo: ["isConnected": isConnected, "connectionType": connectionType]
                )
            }
        }
        
        // Start monitoring on a background queue
        let queue = DispatchQueue(label: "com.backdoor.NetworkMonitor")
        monitor.start(queue: queue)
    }
    
    @available(iOS 12.0, *)
    private func getConnectionTypeString(_ path: NWPath) -> String {
        if path.usesInterfaceType(.wifi) {
            return "WiFi"
        } else if path.usesInterfaceType(.cellular) {
            return "Cellular"
        } else if path.usesInterfaceType(.wiredEthernet) {
            return "Ethernet"
        } else if path.usesInterfaceType(.loopback) {
            return "Loopback"
        } else {
            return "Other"
        }
    }
    
    private func setupLegacyNetworkMonitoring() {
        // Use Reachability or other approaches for legacy iOS versions
        // Set up timer to check connectivity status periodically
        let timer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            let isConnected = self.isConnectedToNetwork()
            Debug.shared.log(message: "Network check - Connected: \(isConnected)", type: .info)
            
            DispatchQueue.main.async {
                NotificationCenter.default.post(
                    name: Notification.Name("NetworkStatusChanged"),
                    object: nil,
                    userInfo: ["isConnected": isConnected]
                )
            }
        }
        
        // Make sure timer runs even when scrolling
        RunLoop.current.add(timer, forMode: .common)
        
        // Also check immediately on setup
        let isConnected = isConnectedToNetwork()
        Debug.shared.log(message: "Initial network status - Connected: \(isConnected)", type: .info)
    }
    
    private func isConnectedToNetwork() -> Bool {
        var zeroAddress = sockaddr_in()
        zeroAddress.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        zeroAddress.sin_family = sa_family_t(AF_INET)
        
        guard let defaultRouteReachability = withUnsafePointer(to: &zeroAddress, {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                SCNetworkReachabilityCreateWithAddress(nil, $0)
            }
        }) else {
            return false
        }
        
        var flags: SCNetworkReachabilityFlags = []
        if !SCNetworkReachabilityGetFlags(defaultRouteReachability, &flags) {
            return false
        }
        
        let isReachable = flags.contains(.reachable)
        let needsConnection = flags.contains(.connectionRequired)
        
        return isReachable && !needsConnection
    }
}
