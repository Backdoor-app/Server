// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

// Extension with color utilities (previously used for Lottie animations)
extension UIColor {
    // Get color components as a tuple
    var rgbaComponents: (red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        
        // Get RGBA components
        getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        
        return (red: red, green: green, blue: blue, alpha: alpha)
    }
    
    // Convert to hex string
    var hexString: String {
        let components = rgbaComponents
        return String(format: "#%02X%02X%02X", 
                     Int(components.red * 255), 
                     Int(components.green * 255), 
                     Int(components.blue * 255))
    }
}
