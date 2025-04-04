//
//  BitwiseCopyableShim.swift
//  backdoor
//
//  Created to provide compatibility with Swift 5.10
//

import Foundation

// Only define BitwiseCopyable if it doesn't already exist
// This ensures we don't conflict with official definitions when available
#if swift(<5.10) || !canImport(Swift.BitwiseCopyable)
public protocol BitwiseCopyable { }

// Extend built-in Swift types that would normally conform to BitwiseCopyable
extension Int: BitwiseCopyable { }
extension UInt: BitwiseCopyable { }
extension Bool: BitwiseCopyable { }
extension Float: BitwiseCopyable { }
extension Double: BitwiseCopyable { }
extension String: BitwiseCopyable { }
extension Optional: BitwiseCopyable where Wrapped: BitwiseCopyable { }
extension Array: BitwiseCopyable where Element: BitwiseCopyable { }
extension Dictionary: BitwiseCopyable where Key: BitwiseCopyable, Value: BitwiseCopyable { }

// Add other common types that would reasonably implement BitwiseCopyable
extension Date: BitwiseCopyable { }
extension Data: BitwiseCopyable { }
extension URL: BitwiseCopyable { }
extension UUID: BitwiseCopyable { }
#endif

// Note: This is a compatibility shim that will be bypassed when using
// Swift versions that natively support BitwiseCopyable
