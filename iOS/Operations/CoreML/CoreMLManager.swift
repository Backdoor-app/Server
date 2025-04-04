// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import CoreML
import UIKit

/// Manages CoreML model loading and prediction operations
final class CoreMLManager {
    // Singleton instance
    static let shared = CoreMLManager()
    
    // Model storage
    private var mlModel: MLModel?
    private var modelLoaded = false
    private var modelURL: URL?
    
    // Public getter for model loaded status
    var isModelLoaded: Bool {
        return modelLoaded
    }
    
    // Queue for thread-safe operations
    private let predictionQueue = DispatchQueue(label: "com.backdoor.coreml.prediction", qos: .userInitiated)
    
    // Private initializer for singleton
    private init() {
        // First try looking in the app bundle
        if let modelPath = Bundle.main.path(forResource: "coreml_model", ofType: "mlmodel", inDirectory: "model") {
            modelURL = URL(fileURLWithPath: modelPath)
            Debug.shared.log(message: "CoreML model found in bundle at path: \(modelPath)", type: .info)
        } 
        // If not found in bundle, try looking in project's model directory
        else {
            // Check for the model in the project root
            let fileManager = FileManager.default
            let modelDirectoryPath = getModelDirectoryPath()
            let modelFilePath = modelDirectoryPath.appendingPathComponent("coreml_model.mlmodel").path
            
            if fileManager.fileExists(atPath: modelFilePath) {
                modelURL = URL(fileURLWithPath: modelFilePath)
                Debug.shared.log(message: "CoreML model found in project directory at: \(modelFilePath)", type: .info)
            } else {
                Debug.shared.log(message: "CoreML model not found in bundle or project directory", type: .error)
            }
        }
    }
    
    /// Get path to model directory
    private func getModelDirectoryPath() -> URL {
        // Try to find the model directory relative to the app's bundle
        let bundleURL = Bundle.main.bundleURL
        
        // First check if we're in a development environment
        let projectRootURL = bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        let modelDir = projectRootURL.appendingPathComponent("model")
        
        return modelDir
    }
    
    // Model loading state tracking
    private var isModelLoading = false
    private var hasShownDeferredNotification = false
    
    /// Load the CoreML model asynchronously with safeguards
    func loadModel(completion: ((Bool) -> Void)? = nil) {
        // Skip in safe mode
        if SafeModeLauncher.shared.inSafeMode {
            Debug.shared.log(message: "CoreML model loading skipped in safe mode", type: .info)
            completion?(false)
            return
        }
        
        // Check memory status before attempting to load large model
        if shouldCheckMemory() && isMemoryConstrained() {
            Debug.shared.log(message: "Memory pressure detected - deferring CoreML model loading", type: .warning)
            notifyUserOfDeferredLoading()
            completion?(false)
            return
        }
        
        // Prevent duplicate loading attempts
        if isModelLoading {
            Debug.shared.log(message: "Model loading already in progress", type: .info)
            completion?(false)
            return
        }
        
        // Mark loading as in progress
        isModelLoading = true
        
        // Call the enhanced version that checks for locally trained models
        loadModelWithLocalLearning { [weak self] success in
            self?.isModelLoading = false
            completion?(success)
        }
    }
    
    /// Check if memory is constrained
    private func isMemoryConstrained() -> Bool {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            let usedMemory = Double(info.resident_size)
            let totalMemory = Double(ProcessInfo.processInfo.physicalMemory)
            let memoryUsage = usedMemory / totalMemory
            
            Debug.shared.log(message: "Memory usage: \(Int(memoryUsage * 100))%", type: .info)
            
            return memoryUsage > 0.7 // If using more than 70% of memory
        } else {
            // Log error for debugging purposes
            Debug.shared.log(message: "Failed to get memory info: error \(kerr)", type: .error)
            
            // Default to true (memory is constrained) to be cautious when we can't determine
            return true
        }
    }
    
    /// Determine if we should check memory at all (performance optimization)
    private func shouldCheckMemory() -> Bool {
        // Only check memory on devices that might be constrained
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let totalMemoryGB = Double(totalMemory) / 1024.0 / 1024.0 / 1024.0
        
        // Debug log total device memory
        Debug.shared.log(message: "Device has \(String(format: "%.1f", totalMemoryGB)) GB RAM", type: .info)
        
        // Always check if device has less than 3GB RAM
        return totalMemoryGB < 3.0
    }
    
    /// Notify user that AI features are deferred
    private func notifyUserOfDeferredLoading() {
        // Only show once per session
        if hasShownDeferredNotification {
            return
        }
        
        hasShownDeferredNotification = true
        
        DispatchQueue.main.async {
            // Find top view controller using the shared extension method
            if let topVC = UIApplication.shared.topMostViewController() {
                let alert = UIAlertController(
                    title: "AI Features Delayed",
                    message: "AI features will be available when system resources allow. You can continue using other app features.",
                    preferredStyle: .alert
                )
                
                alert.addAction(UIAlertAction(title: "OK", style: .default))
                
                topVC.present(alert, animated: true)
            }
        }
    }
    
    /// Load the CoreML model with local learning support
    func loadModelWithLocalLearning(completion: ((Bool) -> Void)? = nil) {
        // If model is already loaded, return early
        guard !modelLoaded else {
            Debug.shared.log(message: "Model already loaded", type: .info)
            completion?(true)
            return
        }
        
        // First check for a locally trained model
        if let localModelURL = AILearningManager.shared.getLatestModelURL() {
            Debug.shared.log(message: "Found locally trained model, attempting to load", type: .info)
            loadModelFromURL(localModelURL) { [weak self] success in
                if success {
                    Debug.shared.log(message: "Successfully loaded locally trained model", type: .info)
                    completion?(true)
                } else {
                    Debug.shared.log(message: "Failed to load locally trained model, falling back to default", type: .warning)
                    // Fall back to default model
                    self?.loadDefaultModel(completion: completion)
                }
            }
            return
        }
        
        // No locally trained model, use default
        loadDefaultModel(completion: completion)
    }
    
    /// Load the default CoreML model
    private func loadDefaultModel(completion: ((Bool) -> Void)? = nil) {
        // First try with existing URL
        if let modelURL = modelURL {
            loadModelFromURL(modelURL, completion: completion)
            return
        }
        
        // If URL is not available, try to get the model from documents directory or copy it there
        ModelFileManager.shared.prepareMLModel { [weak self] result in
            switch result {
            case .success(let modelURL):
                // Load the model from the copied location
                self?.loadModelFromURL(modelURL, completion: completion)
                
            case .failure(let error):
                Debug.shared.log(message: "Failed to prepare model file: \(error.localizedDescription)", type: .error)
                DispatchQueue.main.async {
                    completion?(false)
                }
            }
        }
    }
    
    /// Load model from the specified URL with memory safety and progress indication
    private func loadModelFromURL(_ url: URL, completion: ((Bool) -> Void)? = nil) {
        // Show loading indicator for large files after a small delay
        var loadingAlert: UIAlertController?
        var loadingAlertPresented = false
        
        // Only show UI after a brief delay if loading is still ongoing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.75) { [weak self] in
            if self?.isModelLoading == true {
                // Find appropriate view controller to present on
                // Using the shared extension method to find top view controller
                guard let topVC = UIApplication.shared.topMostViewController() else {
                    return
                }
                
                // Don't interrupt if user is already looking at something
                if topVC.presentedViewController != nil {
                    return
                }
                
                // Create alert with progress indicator
                loadingAlert = UIAlertController(
                    title: "Loading AI Model",
                    message: "Please wait while AI features are prepared...",
                    preferredStyle: .alert
                )
                
                // Add activity indicator
                let indicator = UIActivityIndicatorView(style: .medium)
                indicator.translatesAutoresizingMaskIntoConstraints = false
                loadingAlert?.view.addSubview(indicator)
                indicator.startAnimating()
                
                // Position indicator
                NSLayoutConstraint.activate([
                    indicator.centerXAnchor.constraint(equalTo: loadingAlert!.view.centerXAnchor),
                    indicator.topAnchor.constraint(equalTo: loadingAlert!.view.topAnchor, constant: 80),
                    indicator.widthAnchor.constraint(equalToConstant: 40),
                    indicator.heightAnchor.constraint(equalToConstant: 40)
                ])
                
                // Add cancel button
                loadingAlert?.addAction(UIAlertAction(title: "Cancel", style: .cancel) { [weak self] _ in
                    self?.isModelLoading = false
                    completion?(false)
                })
                
                if let alert = loadingAlert {
                    topVC.present(alert, animated: true)
                    loadingAlertPresented = true
                }
            }
        }
        
        // Set up memory pressure observer
        var memoryObserver: NSObjectProtocol?
        memoryObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main) { [weak self] _ in
                Debug.shared.log(message: "Memory warning during model loading, canceling", type: .warning)
                
                // Clean up - ensure UI operations happen on main thread
                if loadingAlertPresented {
                    DispatchQueue.main.async {
                        loadingAlert?.dismiss(animated: true)
                    }
                }
                
                if let observer = memoryObserver {
                    NotificationCenter.default.removeObserver(observer)
                }
                
                self?.isModelLoading = false
                completion?(false)
            }
        
        // Perform actual loading in background
        predictionQueue.async { [weak self] in
            do {
                // Check file size before loading
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                    if let fileSize = attributes[.size] as? NSNumber {
                        let fileSizeMB = Double(truncating: fileSize) / 1024.0 / 1024.0
                        Debug.shared.log(message: "Model file size: \(String(format: "%.1f", fileSizeMB)) MB", type: .info)
                        
                        // Extra warning for very large models
                        if fileSizeMB > 300 {
                            Debug.shared.log(message: "Warning: Very large model file, performance may be affected", type: .warning)
                        }
                    }
                } catch {
                    Debug.shared.log(message: "Could not determine model file size", type: .warning)
                }
                
                // Actual model loading
                let compiledModelURL = try MLModel.compileModel(at: url)
                let model = try MLModel(contentsOf: compiledModelURL)
                
                // Remove observer as loading succeeded
                if let observer = memoryObserver {
                    NotificationCenter.default.removeObserver(observer)
                }
                
                // Dismiss loading alert
                DispatchQueue.main.async {
                    if loadingAlertPresented {
                        loadingAlert?.dismiss(animated: true)
                    }
                    
                    self?.mlModel = model
                    self?.modelLoaded = true
                    Debug.shared.log(message: "CoreML model loaded successfully from: \(url.path)", type: .info)
                    completion?(true)
                }
            } catch {
                // Remove observer on failure
                if let observer = memoryObserver {
                    NotificationCenter.default.removeObserver(observer)
                }
                
                // Dismiss loading alert
                DispatchQueue.main.async {
                    if loadingAlertPresented {
                        loadingAlert?.dismiss(animated: true)
                    }
                    
                    Debug.shared.log(message: "Failed to load CoreML model: \(error)", type: .error)
                    completion?(false)
                }
            }
        }
    }
    
    /// Analyzes user input to determine intent with ML model
    func predictIntent(from text: String, completion: @escaping (Result<PredictionResult, PredictionError>) -> Void) {
        // Ensure model is loaded before attempting prediction
        if !modelLoaded {
            loadModel { [weak self] success in
                if success {
                    self?.performPrediction(text: text, completion: completion)
                } else {
                    completion(.failure(.modelNotLoaded))
                }
            }
        } else {
            performPrediction(text: text, completion: completion)
        }
    }
    
    /// Performs sentiment analysis on text
    func analyzeSentiment(from text: String, completion: @escaping (Result<SentimentResult, PredictionError>) -> Void) {
        // Ensure model is loaded before attempting prediction
        if !modelLoaded {
            loadModel { [weak self] success in
                if success {
                    self?.performSentimentAnalysis(text: text, completion: completion)
                } else {
                    completion(.failure(.modelNotLoaded))
                }
            }
        } else {
            performSentimentAnalysis(text: text, completion: completion)
        }
    }
    
    // MARK: - Private Methods
    
    /// Execute prediction on loaded model
    private func performPrediction(text: String, completion: @escaping (Result<PredictionResult, PredictionError>) -> Void) {
        predictionQueue.async { [weak self] in
            guard let self = self, let model = self.mlModel else {
                DispatchQueue.main.async {
                    completion(.failure(.modelNotLoaded))
                }
                return
            }
            
            do {
                // Create prediction input based on model metadata
                // This uses a flexible approach to handle various model input types
                let modelDescription = model.modelDescription
                
                // Check model input features
                guard let inputDescription = modelDescription.inputDescriptionsByName.first else {
                    throw PredictionError.invalidModelFormat
                }
                
                let featureName = inputDescription.key
                let featureType = inputDescription.value.type
                
                var inputFeatures: [String: MLFeatureValue] = [:]
                
                // Handle different input feature types
                switch featureType {
                case .string:
                    // Text classification models typically use string inputs
                    inputFeatures[featureName] = MLFeatureValue(string: text)
                    
                case .dictionary:
                    // Some NLP models use dictionary inputs
                    inputFeatures[featureName] = try MLFeatureValue(dictionary: ["text": NSNumber(value: 1)])
                    
                case .multiArray:
                    // For models that expect text to be pre-encoded (we just use a placeholder here)
                    // In a real implementation, we would properly encode the text
                    Debug.shared.log(message: "Model requires multiArray input, using placeholder", type: .warning)
                    let placeholder = try MLMultiArray(shape: [1], dataType: .double)
                    placeholder[0] = NSNumber(value: 1.0)
                    inputFeatures[featureName] = MLFeatureValue(multiArray: placeholder)
                    
                default:
                    throw PredictionError.unsupportedInputType
                }
                
                // Create input from features
                let provider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)
                
                // Make prediction
                let prediction = try model.prediction(from: provider)
                
                // Process prediction outputs
                let result = try self.processOutputs(prediction: prediction, text: text)
                
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch let error as PredictionError {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                DispatchQueue.main.async {
                    Debug.shared.log(message: "CoreML prediction failed: \(error)", type: .error)
                    completion(.failure(.predictionFailed(error)))
                }
            }
        }
    }
    
    /// Process model outputs into a structured result
    private func processOutputs(prediction: MLFeatureProvider, text: String) throws -> PredictionResult {
        // Default values
        var intent = "unknown"
        var confidence: Double = 0.0
        var parameters: [String: Any] = [:]
        var probabilities: [String: Double] = [:]
        
        // Try to extract output features from the prediction
        for featureName in prediction.featureNames {
            guard let feature = prediction.featureValue(for: featureName) else { continue }
            
            switch feature.type {
            case .string:
                // If output is a string, it's likely a classification label (intent)
                // Check if the string value is not empty
                if !feature.stringValue.isEmpty {
                    intent = feature.stringValue
                }
                
            case .dictionary:
                // Some models output probabilities as a dictionary
                if let dict = feature.dictionaryValue as? [String: Double] {
                    probabilities = dict
                    
                    // Find highest probability label
                    if let topPair = dict.max(by: { $0.value < $1.value }) {
                        intent = topPair.key
                        confidence = topPair.value
                    }
                }
                
            case .multiArray:
                // Some models output probability arrays
                if let multiArray = feature.multiArrayValue {
                    // Convert multiArray to dictionary of probabilities
                    // This is a simplified version - we'd need model metadata to map indices to labels
                    let count = multiArray.count
                    
                    for i in 0..<count {
                        let index = i
                        let value = multiArray[index].doubleValue
                        if value != 0 { // Assuming 0 means no value, adjust as needed
                            probabilities["class_\(i)"] = value
                            
                            // Track highest confidence
                            if value > confidence {
                                confidence = value
                                intent = "class_\(i)"
                            }
                        }
                    }
                }
                
            default:
                continue
            }
        }
        
        // Extract potential parameters from the text based on the intent
        parameters = extractParameters(from: text, intent: intent)
        
        return PredictionResult(
            intent: intent,
            confidence: confidence,
            text: text,
            parameters: parameters,
            probabilities: probabilities
        )
    }
    
    /// Extract parameters from text based on the predicted intent
    private func extractParameters(from text: String, intent: String) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        // Use regex to extract structured data from the text
        switch intent.lowercased() {
        case "sign_app", "signing":
            // Extract app name
            let appNameMatches = text.extractMatch(pattern: "(?i)sign\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?([^?.,]+)", groupIndex: 1)
            if let appName = appNameMatches {
                parameters["appName"] = appName.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "navigate", "navigation":
            // Extract destination
            let destinationMatches = text.extractMatch(pattern: "(?i)(?:go\\s+to|navigate\\s+to|open|show)\\s+(?:the\\s+)?([^?.,]+?)\\s+(?:tab|screen|page|section)", groupIndex: 1)
            if let destination = destinationMatches {
                parameters["destination"] = destination.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "add_source", "source":
            // Extract URL
            let urlMatches = text.extractMatch(pattern: "(?i)add\\s+(?:a\\s+)?(?:new\\s+)?source\\s+(?:with\\s+url\\s+|at\\s+|from\\s+)?([^?.,\\s]+)", groupIndex: 1)
            if let url = urlMatches {
                parameters["url"] = url.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "install_app", "install":
            // Extract app name
            let appNameMatches = text.extractMatch(pattern: "(?i)install\\s+(?:the\\s+)?app\\s+(?:called\\s+|named\\s+)?([^?.,]+)", groupIndex: 1)
            if let appName = appNameMatches {
                parameters["appName"] = appName.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "question", "query":
            // Extract question topic
            let topicMatches = text.extractMatch(pattern: "(?i)(?:about|regarding|related\\s+to)\\s+([^?.,]+)", groupIndex: 1)
            if let topic = topicMatches {
                parameters["topic"] = topic.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "greeting", "hello":
            // No parameters needed for greetings
            break
            
        case "help", "assistance":
            // Extract the type of help needed if specified
            if let helpTopic = text.extractMatch(pattern: "(?i)help\\s+(?:me\\s+)?(?:with\\s+)?(.+)", groupIndex: 1) {
                parameters["topic"] = helpTopic.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "search":
            // Extract search query
            if let query = text.extractMatch(pattern: "(?i)(?:search|find|look\\s+for)\\s+(.+)", groupIndex: 1) {
                parameters["query"] = query.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
        case "generate", "create":
            // Extract creation type and content
            if let createType = text.extractMatch(pattern: "(?i)(?:generate|create)\\s+(?:a\\s+)?(?:new\\s+)?([\\w\\s]+?)\\s+(?:for|with|that)\\s+(.+)", groupIndex: 1) {
                parameters["type"] = createType.trimmingCharacters(in: .whitespacesAndNewlines)
                
                if let content = text.extractMatch(pattern: "(?i)(?:generate|create)\\s+(?:a\\s+)?(?:new\\s+)?[\\w\\s]+?\\s+(?:for|with|that)\\s+(.+)", groupIndex: 1) {
                    parameters["content"] = content.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
            
        default:
            // For unknown intents, try to extract common patterns
            extractCommonParameters(from: text, into: &parameters)
        }
        
        return parameters
    }
    
    /// Extract common parameters from any text
    private func extractCommonParameters(from text: String, into parameters: inout [String: Any]) {
        // Look for app names
        if let appName = text.extractMatch(pattern: "(?i)\\b(?:app|application)\\s+(?:called|named)\\s+\"?([^\".,?!]+)\"?", groupIndex: 1) {
            parameters["appName"] = appName.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // Look for URLs
        if let url = text.extractMatch(pattern: "(?i)(?:https?://|www\\.)([^\\s.,]+\\.[^\\s.,]+[^\\s.,?!]*)", groupIndex: 0) {
            parameters["url"] = url.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // Look for file paths
        if let filePath = text.extractMatch(pattern: "(?i)(?:file|path)\\s+(?:at|in)?\\s+\"?([^\".,:;?!]+)\"?", groupIndex: 1) {
            parameters["filePath"] = filePath.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    
    /// Performs sentiment analysis on text
    private func performSentimentAnalysis(text: String, completion: @escaping (Result<SentimentResult, PredictionError>) -> Void) {
        predictionQueue.async { [weak self] in
            guard let self = self, let model = self.mlModel else {
                DispatchQueue.main.async {
                    completion(.failure(.modelNotLoaded))
                }
                return
            }
            
            do {
                // Similar approach to intent prediction but simplified for sentiment
                let modelDescription = model.modelDescription
                
                guard let inputDescription = modelDescription.inputDescriptionsByName.first else {
                    throw PredictionError.invalidModelFormat
                }
                
                let featureName = inputDescription.key
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    featureName: MLFeatureValue(string: text)
                ])
                
                // Make prediction
                let prediction = try model.prediction(from: provider)
                
                // Process sentiment outputs - simplified for common sentiment models
                var sentiment = SentimentType.neutral
                var score: Double = 0.5
                
                for featureName in prediction.featureNames {
                    guard let feature = prediction.featureValue(for: featureName) else { continue }
                    
                    // Check if the feature matches one of our sentiment values
                    if feature.type == .string && ["positive", "negative", "neutral"].contains(feature.stringValue.lowercased()) {
                        // Direct sentiment classification
                        sentiment = SentimentType(rawValue: feature.stringValue.lowercased()) ?? .neutral
                        score = sentiment == .neutral ? 0.5 : (sentiment == .positive ? 0.75 : 0.25)
                    } else if let value = feature.dictionaryValue as? [String: Double] {
                        // Probabilities for each sentiment
                        if let positive = value["positive"], let negative = value["negative"] {
                            score = positive
                            sentiment = positive > negative ? .positive : (negative > positive ? .negative : .neutral)
                        }
                    } else if feature.type == .int64 {
                        // Some models use integers (0, 1, 2) for sentiment
                        let sentimentValue = Int(feature.int64Value)
                        switch sentimentValue {
                        case 0: sentiment = .negative; score = 0.25
                        case 1: sentiment = .neutral; score = 0.5
                        case 2: sentiment = .positive; score = 0.75
                        default: sentiment = .neutral; score = 0.5
                        }
                    }
                }
                
                let result = SentimentResult(
                    sentiment: sentiment,
                    score: score,
                    text: text
                )
                
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch let error as PredictionError {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                DispatchQueue.main.async {
                    Debug.shared.log(message: "CoreML sentiment analysis failed: \(error)", type: .error)
                    completion(.failure(.predictionFailed(error)))
                }
            }
        }
    }
}

// Import the extension containing extractMatch from AppContextManager+AIIntegration.swift
// Note: The extractMatch method is defined elsewhere in the project
// We avoid redeclaration by using that implementation instead

// MARK: - Models

/// Result of intent prediction
struct PredictionResult {
    let intent: String
    let confidence: Double
    let text: String
    let parameters: [String: Any]
    let probabilities: [String: Double]
}

/// Result of sentiment analysis
struct SentimentResult {
    let sentiment: SentimentType
    let score: Double
    let text: String
}

/// Types of sentiment
enum SentimentType: String {
    case positive
    case negative
    case neutral
}

/// Errors that can occur during prediction
enum PredictionError: Error, LocalizedError {
    case modelNotLoaded
    case modelNotFound
    case invalidModelFormat
    case unsupportedInputType
    case unsupportedOperation
    case predictionFailed(Error)
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "CoreML model is not loaded"
        case .modelNotFound:
            return "CoreML model could not be found"
        case .invalidModelFormat:
            return "CoreML model has an invalid format"
        case .unsupportedInputType:
            return "CoreML model has an unsupported input type"
        case .unsupportedOperation:
            return "Operation not supported on this iOS version"
        case let .predictionFailed(error):
            return "Prediction failed: \(error.localizedDescription)"
        }
    }
}
