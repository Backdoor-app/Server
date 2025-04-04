// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

/// A comprehensive networking system with caching, retries, and background processing
final class NetworkManager {
    // MARK: - Singleton

    /// Shared instance of the network manager
    static let shared = NetworkManager()

    // MARK: - Configuration

    /// Configuration for network requests
    struct Configuration {
        /// Default timeout interval for requests (seconds)
        var timeoutInterval: TimeInterval = 30.0

        /// Maximum number of retry attempts
        var maxRetryAttempts: Int = 3

        /// Base delay for exponential backoff (seconds)
        var baseRetryDelay: TimeInterval = 1.0

        /// Whether to use caching for GET requests
        var useCache: Bool = true

        /// Cache lifetime in seconds (1 hour)
        var cacheLifetime: TimeInterval = 3600

        /// Maximum concurrent operations
        var maxConcurrentOperations: Int = 4
    }

    // MARK: - Properties

    /// The configuration for this manager
    internal let _configuration: Configuration

    /// Public accessor for the configuration
    var configuration: Configuration {
        return _configuration
    }

    /// URL session for making network requests
    internal let session: URLSession

    /// Operation queue for network operations
    private let operationQueue = OperationQueue()

    /// Dictionary to keep track of active operations
    internal var activeOperations = [URLRequest: URLSessionTask]()

    /// Queue for synchronizing access to active operations
    internal let operationQueueAccessQueue = DispatchQueue(label: "com.backdoor.NetworkManager.OperationQueue")

    /// In-memory cache for responses
    internal let responseCache = NSCache<NSString, CachedResponse>()

    /// File manager for disk operations
    private let fileManager = FileManager.default

    /// Directory for disk cache
    private let cacheDirectory: URL
    
    /// Queue for cache cleanup operations
    private let cleanupQueue = DispatchQueue(label: "com.backdoor.NetworkManager.CleanupQueue", qos: .background)

    // MARK: - Initialization

    private init(configuration: Configuration = Configuration()) {
        self._configuration = configuration

        // Configure URL session
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = configuration.timeoutInterval
        sessionConfig.timeoutIntervalForResource = configuration.timeoutInterval * 2
        sessionConfig.waitsForConnectivity = true
        sessionConfig.urlCache = nil // Disable system URL cache to use our custom cache
        sessionConfig.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: sessionConfig)

        // Configure operation queue
        operationQueue.maxConcurrentOperationCount = configuration.maxConcurrentOperations
        operationQueue.qualityOfService = .userInitiated

        // Configure cache
        responseCache.name = "com.backdoor.NetworkManager.ResponseCache"
        responseCache.countLimit = 100 // Set a reasonable limit for in-memory cache
        
        // Set total cost limit to 50MB (approximate)
        responseCache.totalCostLimit = 50 * 1024 * 1024

        // Configure disk cache directory
        let cachesDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first!
        cacheDirectory = cachesDirectory.appendingPathComponent("NetworkCache", isDirectory: true)

        // Create cache directory if it doesn't exist
        do {
            try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        } catch {
            Debug.shared.log(message: "Failed to create network cache directory: \(error)", type: .error)
        }

        // Clean expired caches
        cleanExpiredCaches()
        
        // Register for memory warning notifications
        NotificationCenter.default.addObserver(self, 
                                              selector: #selector(handleMemoryWarning), 
                                              name: UIApplication.didReceiveMemoryWarningNotification, 
                                              object: nil)
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    @objc private func handleMemoryWarning() {
        // Clear memory cache on memory warning
        responseCache.removeAllObjects()
        Debug.shared.log(message: "Cleared network response cache due to memory warning", type: .warning)
    }

    // MARK: - Public Interface

    /// Perform a network request
    /// - Parameters:
    ///   - request: The URL request to perform
    ///   - caching: Whether to use caching (default is based on configuration)
    ///   - completion: Completion handler with the result
    /// - Returns: A cancellable task identifier
    @discardableResult
    func performRequest<T: Decodable>(
        _ request: URLRequest,
        caching: Bool? = nil,
        completion: @escaping (Result<T, Error>) -> Void
    ) -> URLSessionTask? {
        // Determine whether to use caching
        let useCache = caching ?? (_configuration.useCache && request.httpMethod?.uppercased() == "GET")

        // Check if request is already in progress
        let existingTask = operationQueueAccessQueue.sync { activeOperations[request] }
        if let existingTask = existingTask {
            Debug.shared.log(message: "Request already in progress: \(request.url?.absoluteString ?? "Unknown URL")", type: .debug)
            return existingTask
        }

        // Check cache if caching is enabled
        if useCache {
            if let cachedResponse = getCachedResponse(for: request) {
                Debug.shared.log(message: "Cache hit: \(request.url?.absoluteString ?? "Unknown URL")", type: .debug)

                do {
                    let decodedObject = try JSONDecoder().decode(T.self, from: cachedResponse.data)
                    DispatchQueue.main.async {
                        completion(.success(decodedObject))
                    }
                    return nil
                } catch {
                    Debug.shared.log(message: "Failed to decode cached response: \(error)", type: .error)
                    // Continue with network request if decoding fails
                }
            }
        }

        // Create network task
        let task = createNetworkTask(request: request, retryCount: 0, useCache: useCache) { (result: Result<T, Error>) in
            completion(result)
        }

        // Add to active operations
        operationQueueAccessQueue.sync {
            activeOperations[request] = task
        }

        // Start the task
        task.resume()

        return task
    }

    /// Cancel all active operations
    func cancelAllOperations() {
        operationQueueAccessQueue.sync {
            for (_, task) in activeOperations {
                task.cancel()
            }
            activeOperations.removeAll()
        }
    }

    /// Cancel a specific operation
    /// - Parameter request: The request to cancel
    func cancelOperation(for request: URLRequest) {
        operationQueueAccessQueue.sync {
            if let task = activeOperations[request] {
                task.cancel()
                activeOperations.removeValue(forKey: request)
            }
        }
    }

    /// Clear all caches (memory and disk)
    func clearCache() {
        // Clear memory cache
        responseCache.removeAllObjects()

        // Clear disk cache
        cleanupQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                let contents = try self.fileManager.contentsOfDirectory(at: self.cacheDirectory, includingPropertiesForKeys: nil)
                for url in contents {
                    try self.fileManager.removeItem(at: url)
                }
                Debug.shared.log(message: "Network cache cleared", type: .info)
            } catch {
                Debug.shared.log(message: "Failed to clear network cache: \(error)", type: .error)
            }
        }
    }

    // MARK: - Private Methods

    /// Create a network task with retry logic
    /// - Parameters:
    ///   - request: The URL request
    ///   - retryCount: Current retry count
    ///   - useCache: Whether to cache the response
    ///   - completion: Completion handler with the result
    /// - Returns: The URLSessionTask
    private func createNetworkTask<T: Decodable>(
        request: URLRequest,
        retryCount: Int,
        useCache: Bool,
        completion: @escaping (Result<T, Error>) -> Void
    ) -> URLSessionTask {
        let task = session.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }

            // Remove from active operations
            self.operationQueueAccessQueue.sync {
                self.activeOperations.removeValue(forKey: request)
            }

            // Handle network error
            if let error = error {
                // Check if this is a connectivity issue that should be retried
                if (error as NSError).code == NSURLErrorNotConnectedToInternet ||
                    (error as NSError).code == NSURLErrorTimedOut ||
                    (error as NSError).code == NSURLErrorNetworkConnectionLost
                {
                    // Check if we should retry
                    if retryCount < self.configuration.maxRetryAttempts {
                        // Calculate delay with exponential backoff
                        let delay = self.configuration.baseRetryDelay * pow(2.0, Double(retryCount))

                        Debug.shared.log(message: "Network error, retrying in \(delay) seconds (attempt \(retryCount + 1)): \(request.url?.absoluteString ?? "Unknown URL")", type: .warning)

                        // Retry after delay
                        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
                            guard let self = self else { return }

                            let retryTask = self.createNetworkTask(
                                request: request,
                                retryCount: retryCount + 1,
                                useCache: useCache,
                                completion: completion
                            )

                            // Add to active operations
                            self.operationQueueAccessQueue.sync {
                                self.activeOperations[request] = retryTask
                            }

                            // Start the retry task
                            retryTask.resume()
                        }

                        return
                    }
                }

                Debug.shared.log(message: "Network request failed: \(error.localizedDescription)", type: .error)
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }

            // Check for valid HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                DispatchQueue.main.async {
                    completion(.failure(NetworkError.invalidResponse))
                }
                return
            }

            // Check status code
            guard (200 ... 299).contains(httpResponse.statusCode) else {
                let error = NetworkError.httpError(statusCode: httpResponse.statusCode)

                // Check if we should retry server errors
                if (500 ... 599).contains(httpResponse.statusCode) && retryCount < self.configuration.maxRetryAttempts {
                    // Calculate delay with exponential backoff
                    let delay = self.configuration.baseRetryDelay * pow(2.0, Double(retryCount))

                    Debug.shared.log(message: "Server error (\(httpResponse.statusCode)), retrying in \(delay) seconds (attempt \(retryCount + 1)): \(request.url?.absoluteString ?? "Unknown URL")", type: .warning)

                    // Retry after delay
                    DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
                        guard let self = self else { return }

                        let retryTask = self.createNetworkTask(
                            request: request,
                            retryCount: retryCount + 1,
                            useCache: useCache,
                            completion: completion
                        )

                        // Add to active operations
                        self.operationQueueAccessQueue.sync {
                            self.activeOperations[request] = retryTask
                        }

                        // Start the retry task
                        retryTask.resume()
                    }

                    return
                }

                Debug.shared.log(message: "HTTP error: \(httpResponse.statusCode)", type: .error)
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }

            // Ensure we have data
            guard let data = data else {
                DispatchQueue.main.async {
                    completion(.failure(NetworkError.noData))
                }
                return
            }

            // Cache the response if needed
            if useCache {
                self.cacheResponse(data: data, for: request)
            }

            // Decode the response on a background queue to avoid blocking the main thread
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let decodedObject = try JSONDecoder().decode(T.self, from: data)
                    DispatchQueue.main.async {
                        completion(.success(decodedObject))
                    }
                } catch {
                    Debug.shared.log(message: "Failed to decode response: \(error.localizedDescription)", type: .error)
                    DispatchQueue.main.async {
                        completion(.failure(NetworkError.decodingError(error)))
                    }
                }
            }
        }

        return task
    }

    // MARK: - Caching

    /// Cache a response
    /// - Parameters:
    ///   - data: The response data
    ///   - request: The URL request
    internal func cacheResponse(data: Data, for request: URLRequest) {
        guard let url = request.url else { return }

        // Create cached response
        let cachedResponse = CachedResponse(data: data, timestamp: Date())

        // Store in memory cache with cost based on data size
        let cacheKey = NSString(string: url.absoluteString)
        responseCache.setObject(cachedResponse, forKey: cacheKey, cost: data.count)

        // Store in disk cache
        cleanupQueue.async { [weak self] in
            guard let self = self else { return }
            
            let fileURL = self.cacheFileURL(for: url)
            
            do {
                let cacheData = try NSKeyedArchiver.archivedData(withRootObject: cachedResponse, requiringSecureCoding: true)
                try cacheData.write(to: fileURL)
                Debug.shared.log(message: "Response cached: \(url.absoluteString)", type: .debug)
            } catch {
                Debug.shared.log(message: "Failed to cache response: \(error)", type: .error)
            }
        }
    }

    /// Get a cached response
    /// - Parameter request: The URL request
    /// - Returns: The cached response, if available and not expired
    private func getCachedResponse(for request: URLRequest) -> CachedResponse? {
        guard let url = request.url else { return nil }

        // Check memory cache first
        let cacheKey = NSString(string: url.absoluteString)
        if let cachedResponse = responseCache.object(forKey: cacheKey) {
            // Check if cache is expired
            if !isCacheExpired(cachedResponse) {
                return cachedResponse
            }
        }

        // Check disk cache if not in memory
        let fileURL = cacheFileURL(for: url)

        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                if let cachedResponse = try NSKeyedUnarchiver.unarchivedObject(ofClass: CachedResponse.self, from: data) {
                    // Check if cache is expired
                    if !isCacheExpired(cachedResponse) {
                        // Store in memory cache for future use with cost based on data size
                        responseCache.setObject(cachedResponse, forKey: cacheKey, cost: cachedResponse.data.count)
                        return cachedResponse
                    } else {
                        // Remove expired cache from disk
                        cleanupQueue.async { [weak self] in
                            try? self?.fileManager.removeItem(at: fileURL)
                        }
                    }
                }
            } catch {
                Debug.shared.log(message: "Failed to load cached response: \(error)", type: .error)
            }
        }

        return nil
    }

    /// Check if a cached response is expired
    /// - Parameter cachedResponse: The cached response
    /// - Returns: True if the cache is expired
    internal func isCacheExpired(_ cachedResponse: CachedResponse) -> Bool {
        let now = Date()
        let expirationTime = cachedResponse.timestamp.addingTimeInterval(_configuration.cacheLifetime)
        return now > expirationTime
    }

    /// Get the file URL for caching a response
    /// - Parameter url: The URL of the request
    /// - Returns: The file URL for caching
    private func cacheFileURL(for url: URL) -> URL {
        // Use URL's absoluteString hashed as the filename
        let urlString = url.absoluteString
        let filename = urlString.hash.magnitude.description
        return cacheDirectory.appendingPathComponent(filename)
    }

    /// Clean expired caches
    private func cleanExpiredCaches() {
        cleanupQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                let contents = try self.fileManager.contentsOfDirectory(
                    at: self.cacheDirectory,
                    includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]
                )

                let now = Date()
                var totalSize: UInt64 = 0
                var filesToDelete: [URL] = []

                // First pass: identify expired files and calculate total size
                for url in contents {
                    do {
                        let data = try Data(contentsOf: url)
                        if let cachedResponse = try NSKeyedUnarchiver.unarchivedObject(ofClass: CachedResponse.self, from: data) {
                            let expirationTime = cachedResponse.timestamp.addingTimeInterval(self._configuration.cacheLifetime)
                            
                            // Add to delete list if expired
                            if now > expirationTime {
                                filesToDelete.append(url)
                            } else {
                                // Add to total size if not expired
                                let attributes = try self.fileManager.attributesOfItem(atPath: url.path)
                                if let fileSize = attributes[.size] as? UInt64 {
                                    totalSize += fileSize
                                }
                            }
                        } else {
                            // Invalid cache file, add to delete list
                            filesToDelete.append(url)
                        }
                    } catch {
                        // If we can't read the file, add to delete list
                        filesToDelete.append(url)
                    }
                }
                
                // Delete expired files
                for url in filesToDelete {
                    try? self.fileManager.removeItem(at: url)
                    Debug.shared.log(message: "Removed expired network cache: \(url.lastPathComponent)", type: .debug)
                }
                
                // If total size is still too large, delete oldest files
                if totalSize > 100 * 1024 * 1024 { // 100 MB limit
                    let remainingFiles = try self.fileManager.contentsOfDirectory(
                        at: self.cacheDirectory,
                        includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]
                    )
                    
                    // Sort by modification date (oldest first)
                    let sortedFiles = remainingFiles.sorted { url1, url2 -> Bool in
                        do {
                            let attr1 = try self.fileManager.attributesOfItem(atPath: url1.path)
                            let attr2 = try self.fileManager.attributesOfItem(atPath: url2.path)
                            let date1 = attr1[.modificationDate] as? Date ?? Date.distantFuture
                            let date2 = attr2[.modificationDate] as? Date ?? Date.distantFuture
                            return date1 < date2
                        } catch {
                            return false
                        }
                    }
                    
                    // Delete oldest files until we're under the limit
                    var currentSize = totalSize
                    let targetSize: UInt64 = 80 * 1024 * 1024 // Target 80 MB after cleanup
                    
                    for url in sortedFiles {
                        if currentSize <= targetSize {
                            break
                        }
                        
                        do {
                            let attributes = try self.fileManager.attributesOfItem(atPath: url.path)
                            if let fileSize = attributes[.size] as? UInt64 {
                                try self.fileManager.removeItem(at: url)
                                currentSize -= fileSize
                                Debug.shared.log(message: "Removed old network cache to reduce size: \(url.lastPathComponent)", type: .debug)
                            }
                        } catch {
                            Debug.shared.log(message: "Failed to remove cache file: \(error)", type: .error)
                        }
                    }
                }
            } catch {
                Debug.shared.log(message: "Failed to clean expired caches: \(error)", type: .error)
            }
        }
    }
}

// MARK: - NetworkError

/// Errors that can occur during network operations
enum NetworkError: Error, LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(statusCode: Int)
    case noData
    case decodingError(Error)
    case cancelled

    var errorDescription: String? {
        switch self {
            case .invalidURL:
                return "Invalid URL"
            case .invalidResponse:
                return "Invalid response from server"
            case let .httpError(statusCode):
                return "HTTP error: \(statusCode)"
            case .noData:
                return "No data received"
            case let .decodingError(error):
                return "Failed to decode response: \(error.localizedDescription)"
            case .cancelled:
                return "Request was cancelled"
        }
    }
}

// MARK: - CachedResponse

/// A cached network response
final class CachedResponse: NSObject, NSSecureCoding {
    static var supportsSecureCoding: Bool = true

    /// The response data
    let data: Data

    /// When the response was cached
    let timestamp: Date

    /// Initialize with data and timestamp
    /// - Parameters:
    ///   - data: The response data
    ///   - timestamp: When the response was cached
    init(data: Data, timestamp: Date) {
        self.data = data
        self.timestamp = timestamp
        super.init()
    }

    // MARK: - NSSecureCoding

    /// Encode with coder
    /// - Parameter coder: The coder
    func encode(with coder: NSCoder) {
        coder.encode(data, forKey: "data")
        coder.encode(timestamp, forKey: "timestamp")
    }

    /// Initialize with coder
    /// - Parameter coder: The coder
    required init?(coder: NSCoder) {
        guard let data = coder.decodeObject(of: NSData.self, forKey: "data") as? Data,
              let timestamp = coder.decodeObject(of: NSDate.self, forKey: "timestamp") as? Date
        else {
            return nil
        }

        self.data = data
        self.timestamp = timestamp
        super.init()
    }
}

// MARK: - URLRequest Extension

extension URLRequest {
    /// Create a request with optional caching and timeout configuration
    /// - Parameters:
    ///   - url: The URL for the request
    ///   - httpMethod: HTTP method (default is GET)
    ///   - timeoutInterval: Timeout interval (default is from NetworkManager.Configuration)
    /// - Returns: The configured URL request
    static func build(
        url: URL,
        httpMethod: String = "GET",
        timeoutInterval: TimeInterval = NetworkManager.shared.configuration.timeoutInterval
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeoutInterval)
        request.httpMethod = httpMethod
        return request
    }
}

// MARK: - NetworkManager Configuration Extension

// We access configuration directly from the instance property
// No extension needed here

// MARK: - Batch Request

/// A batch of network requests that can be executed together
final class BatchRequest {
    // MARK: - Properties

    /// Requests in the batch
    private var requests: [URLRequest] = []

    /// Completion handler for the batch
    private var completion: (([Any], [Error]) -> Void)?

    /// Results of completed requests
    private var results: [Any] = []

    /// Errors from failed requests
    private var errors: [Error] = []

    /// Count of completed requests
    private var completedCount = 0

    /// Lock for thread-safe access to state
    private let lock = NSLock()

    // MARK: - Initialization

    /// Initialize a new batch request
    init() {}

    // MARK: - Public Interface

    /// Add a request to the batch
    /// - Parameter request: The URL request to add
    /// - Returns: Self for chaining
    @discardableResult
    func add(_ request: URLRequest) -> Self {
        lock.lock()
        requests.append(request)
        lock.unlock()
        return self
    }

    /// Execute all requests in the batch
    /// - Parameter completion: Completion handler with results and errors
    func execute(completion: @escaping ([Any], [Error]) -> Void) {
        self.completion = completion

        lock.lock()
        let requestsCopy = requests
        lock.unlock()

        // Initialize results and errors arrays
        results = Array(repeating: NSNull(), count: requestsCopy.count)
        errors = Array(repeating: NetworkError.invalidResponse, count: requestsCopy.count)

        // Execute each request
        for (index, request) in requestsCopy.enumerated() {
            // Use a special non-generic method for batch requests
            NetworkManager.shared.performRequestWithoutDecoding(request) { [weak self] result in
                guard let self = self else { return }

                self.lock.lock()
                defer { self.lock.unlock() }

                switch result {
                    case let .success(value):
                        self.results[index] = value
                    case let .failure(error):
                        self.errors[index] = error
                }

                self.completedCount += 1

                // Check if all requests are complete
                if self.completedCount == requestsCopy.count {
                    DispatchQueue.main.async {
                        self.completion?(self.results, self.errors)
                    }
                }
            }
        }
    }

    /// Cancel all requests in the batch
    func cancel() {
        lock.lock()
        let requestsCopy = requests
        lock.unlock()

        for request in requestsCopy {
            NetworkManager.shared.cancelOperation(for: request)
        }
    }
}
