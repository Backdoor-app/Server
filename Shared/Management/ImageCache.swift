// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// A comprehensive image caching system that handles both memory and disk caching
final class ImageCache {
    // MARK: - Singleton

    /// Shared instance of the image cache
    static let shared = ImageCache()

    // MARK: - Cache Storage

    /// Memory cache for quick access to recently used images
    private let memoryCache = NSCache<NSString, UIImage>()

    /// File manager for disk operations
    private let fileManager = FileManager.default

    /// Queue for disk operations
    private let diskQueue = DispatchQueue(label: "com.backdoor.ImageCache.DiskQueue", qos: .utility)

    /// Queue for image processing operations
    private let processingQueue = DispatchQueue(label: "com.backdoor.ImageCache.ProcessingQueue", qos: .userInitiated, attributes: .concurrent)

    /// Directory for disk cache
    private let cacheDirectory: URL

    /// Maximum memory cache size (in number of items)
    private let maxMemoryCacheSize = 100

    /// Maximum disk cache size (in bytes)
    private let maxDiskCacheSize: UInt = 100 * 1024 * 1024 // 100 MB

    // MARK: - Active Operations

    /// Dictionary to keep track of active download operations
    private var downloadOperations = [URL: Operation]()

    /// Queue for synchronizing access to download operations
    private let operationQueue = DispatchQueue(label: "com.backdoor.ImageCache.OperationQueue", attributes: .concurrent)

    /// Operation queue for downloads
    private let downloadQueue = OperationQueue()
    
    /// Cache for image URLs that failed to load
    private var failedURLs = Set<URL>()
    
    /// Queue for synchronizing access to failed URLs
    private let failedURLsQueue = DispatchQueue(label: "com.backdoor.ImageCache.FailedURLsQueue")

    // MARK: - Initialization

    private init() {
        // Configure memory cache
        memoryCache.name = "com.backdoor.ImageCache"
        memoryCache.countLimit = maxMemoryCacheSize
        memoryCache.totalCostLimit = 50 * 1024 * 1024 // 50 MB limit

        // Configure disk cache directory
        let cachesDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first!
        cacheDirectory = cachesDirectory.appendingPathComponent("ImageCache", isDirectory: true)

        // Create cache directory if it doesn't exist
        do {
            try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        } catch {
            Debug.shared.log(message: "Failed to create image cache directory: \(error)", type: .error)
        }

        // Configure download queue
        downloadQueue.maxConcurrentOperationCount = 4
        downloadQueue.qualityOfService = .utility

        // Subscribe to memory warning notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )

        // Subscribe to app background notification
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )

        // Perform initial cleanup
        performCleanup()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Public Interface

    /// Load an image from the given URL, with caching
    /// - Parameters:
    ///   - url: The URL to load the image from
    ///   - placeholder: Optional placeholder image to use while loading
    ///   - downsampling: Whether to downsample the image to target size
    ///   - targetSize: Target size for downsampling (if enabled)
    ///   - completion: Completion handler with the loaded image
    func loadImage(from url: URL?,
                   placeholder: UIImage? = nil,
                   downsampling: Bool = true,
                   targetSize: CGSize = CGSize(width: 80, height: 80),
                   completion: @escaping (UIImage?) -> Void)
    {
        // Return placeholder immediately if URL is nil
        guard let url = url else {
            DispatchQueue.main.async {
                completion(placeholder)
            }
            return
        }
        
        // Check if URL previously failed to load
        var shouldSkip = false
        failedURLsQueue.sync {
            shouldSkip = failedURLs.contains(url)
        }
        
        if shouldSkip {
            Debug.shared.log(message: "Skipping previously failed URL: \(url.lastPathComponent)", type: .debug)
            DispatchQueue.main.async {
                completion(placeholder)
            }
            return
        }

        // Check memory cache first (main thread)
        let cacheKey = NSString(string: url.absoluteString)
        if let cachedImage = memoryCache.object(forKey: cacheKey) {
            Debug.shared.log(message: "Image cache hit (memory): \(url.lastPathComponent)", type: .debug)
            DispatchQueue.main.async {
                completion(cachedImage)
            }
            return
        }

        // Return placeholder while we check disk/network
        if let placeholder = placeholder {
            DispatchQueue.main.async {
                completion(placeholder)
            }
        }

        // Check if operation already in progress for this URL
        var isAlreadyInProgress = false
        operationQueue.sync {
            isAlreadyInProgress = downloadOperations[url] != nil
        }
        
        if isAlreadyInProgress {
            Debug.shared.log(message: "Image download already in progress: \(url.lastPathComponent)", type: .debug)
            return
        }

        // Convert NSString to Swift String to avoid @Sendable capture issues
        let cacheKeyString = url.absoluteString

        // Create download operation
        let operation = BlockOperation { [weak self] in
            guard let self = self else { return }

            // Check disk cache
            if let diskCachedImage = self.loadImageFromDisk(url: url) {
                // Store in memory cache with cost based on image size
                let cacheKey = NSString(string: cacheKeyString)
                let cost = Int(diskCachedImage.size.width * diskCachedImage.size.height * 4) // Approximate bytes (RGBA)
                self.memoryCache.setObject(diskCachedImage, forKey: cacheKey, cost: cost)

                DispatchQueue.main.async {
                    completion(diskCachedImage)
                }

                // Remove operation from tracking
                self.operationQueue.async(flags: .barrier) {
                    self.downloadOperations.removeValue(forKey: url)
                }
                return
            }

            // Download image if not in cache
            self.downloadImage(from: url, downsampling: downsampling, targetSize: targetSize) { [weak self] image in
                guard let self = self else { return }
                
                if let image = image {
                    // Store in memory cache with cost based on image size
                    let cacheKey = NSString(string: cacheKeyString)
                    let cost = Int(image.size.width * image.size.height * 4) // Approximate bytes (RGBA)
                    self.memoryCache.setObject(image, forKey: cacheKey, cost: cost)

                    // Store in disk cache
                    self.saveImageToDisk(image: image, url: url)

                    // Return on main thread
                    DispatchQueue.main.async {
                        completion(image)
                    }
                } else {
                    // Mark URL as failed
                    self.failedURLsQueue.async {
                        self.failedURLs.insert(url)
                    }
                    
                    DispatchQueue.main.async {
                        completion(nil)
                    }
                }

                // Remove operation from tracking
                self.operationQueue.async(flags: .barrier) {
                    self.downloadOperations.removeValue(forKey: url)
                }
            }
        }

        // Add operation to tracking dictionary
        operationQueue.async(flags: .barrier) {
            self.downloadOperations[url] = operation
        }

        // Start the operation
        downloadQueue.addOperation(operation)
    }

    /// Cancel the loading of an image from the given URL
    /// - Parameter url: The URL to cancel loading for
    func cancelLoading(for url: URL) {
        operationQueue.async(flags: .barrier) { [weak self] in
            guard let self = self, let operation = self.downloadOperations[url] else { return }
            operation.cancel()
            self.downloadOperations.removeValue(forKey: url)
        }
    }

    /// Clear all caches (memory and disk)
    func clearCache() {
        // Clear memory cache
        memoryCache.removeAllObjects()
        
        // Clear failed URLs cache
        failedURLsQueue.async {
            self.failedURLs.removeAll()
        }

        // Clear disk cache
        diskQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                let contents = try self.fileManager.contentsOfDirectory(at: self.cacheDirectory, includingPropertiesForKeys: nil)
                for url in contents {
                    try self.fileManager.removeItem(at: url)
                }
                Debug.shared.log(message: "Image cache cleared", type: .info)
            } catch {
                Debug.shared.log(message: "Failed to clear image cache: \(error)", type: .error)
            }
        }
    }

    // MARK: - Private Methods

    /// Download image from URL
    /// - Parameters:
    ///   - url: The URL to download from
    ///   - downsampling: Whether to downsample the image
    ///   - targetSize: Target size for downsampling
    ///   - completion: Completion handler with the downloaded image
    private func downloadImage(from url: URL,
                               downsampling: Bool,
                               targetSize: CGSize,
                               completion: @escaping (UIImage?) -> Void)
    {
        let task = URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            guard let self = self,
                  let data = data,
                  error == nil,
                  let response = response as? HTTPURLResponse,
                  response.statusCode == 200
            else {
                Debug.shared.log(message: "Image download failed: \(url.lastPathComponent) - \(error?.localizedDescription ?? "Unknown error")", type: .error)
                completion(nil)
                return
            }

            self.processingQueue.async {
                if downsampling, let uiImage = UIImage(data: data) {
                    // Downsample the image to reduce memory usage
                    if let downsampledImage = self.downsample(image: uiImage, to: targetSize) {
                        Debug.shared.log(message: "Image downloaded and downsampled: \(url.lastPathComponent)", type: .debug)
                        completion(downsampledImage)
                    } else {
                        Debug.shared.log(message: "Image downloaded but downsampling failed: \(url.lastPathComponent)", type: .warning)
                        completion(uiImage)
                    }
                } else if let uiImage = UIImage(data: data) {
                    Debug.shared.log(message: "Image downloaded: \(url.lastPathComponent)", type: .debug)
                    completion(uiImage)
                } else {
                    Debug.shared.log(message: "Downloaded data could not be converted to image: \(url.lastPathComponent)", type: .error)
                    completion(nil)
                }
            }
        }
        task.resume()
    }

    /// Load image from disk cache
    /// - Parameter url: The URL of the image
    /// - Returns: The cached image, if available
    private func loadImageFromDisk(url: URL) -> UIImage? {
        let fileURL = cacheFileURL(for: url)

        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                if let image = UIImage(data: data) {
                    Debug.shared.log(message: "Image cache hit (disk): \(url.lastPathComponent)", type: .debug)
                    return image
                }
            } catch {
                Debug.shared.log(message: "Failed to load image from disk: \(error)", type: .error)
            }
        }

        return nil
    }

    /// Save image to disk cache
    /// - Parameters:
    ///   - image: The image to save
    ///   - url: The URL of the image
    private func saveImageToDisk(image: UIImage, url: URL) {
        diskQueue.async { [weak self] in
            guard let self = self else { return }

            let fileURL = self.cacheFileURL(for: url)

            if let data = image.jpegData(compressionQuality: 0.8) {
                do {
                    try data.write(to: fileURL)
                    Debug.shared.log(message: "Image saved to disk: \(url.lastPathComponent)", type: .debug)
                } catch {
                    Debug.shared.log(message: "Failed to save image to disk: \(error)", type: .error)
                }
            }
        }
    }

    /// Get the file URL for caching an image
    /// - Parameter url: The source URL of the image
    /// - Returns: The file URL for caching
    private func cacheFileURL(for url: URL) -> URL {
        // Use URL's absoluteString hashed as the filename
        let urlString = url.absoluteString
        let filename = urlString.hash.magnitude.description
        return cacheDirectory.appendingPathComponent(filename)
    }

    /// Downsample an image to reduce memory usage
    /// - Parameters:
    ///   - image: The image to downsample
    ///   - targetSize: The target size for downsampling
    /// - Returns: The downsampled image
    private func downsample(image: UIImage, to targetSize: CGSize) -> UIImage? {
        let imageSourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary

        guard let data = image.jpegData(compressionQuality: 1.0) else { return nil }
        guard let imageSource = CGImageSourceCreateWithData(data as CFData, imageSourceOptions) else { return nil }

        let downsamplingOptions = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(targetSize.width, targetSize.height) * UIScreen.main.scale,
        ] as CFDictionary

        guard let downsampledImage = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, downsamplingOptions) else {
            return nil
        }

        return UIImage(cgImage: downsampledImage)
    }

    /// Handle memory warning notification
    @objc private func handleMemoryWarning() {
        // Clear memory cache on memory warning
        memoryCache.removeAllObjects()
        Debug.shared.log(message: "Cleared image memory cache due to memory warning", type: .warning)
    }

    /// Handle app entering background
    @objc private func handleAppDidEnterBackground() {
        // Perform cleanup when app enters background
        performCleanup()
    }

    /// Perform cleanup of disk cache
    private func performCleanup() {
        diskQueue.async { [weak self] in
            guard let self = self else { return }

            // Get all cached files
            do {
                let contents = try self.fileManager.contentsOfDirectory(at: self.cacheDirectory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])

                // Get total size and sort by date
                var totalSize: UInt = 0
                let sortedFiles = contents.sorted { url1, url2 -> Bool in
                    do {
                        let values1 = try url1.resourceValues(forKeys: [.contentModificationDateKey])
                        let values2 = try url2.resourceValues(forKeys: [.contentModificationDateKey])

                        if let date1 = values1.contentModificationDate, let date2 = values2.contentModificationDate {
                            return date1 < date2
                        }
                    } catch {}
                    return false
                }

                // Calculate total size
                for url in contents {
                    do {
                        let values = try url.resourceValues(forKeys: [.fileSizeKey])
                        if let size = values.fileSize {
                            totalSize += UInt(size)
                        }
                    } catch {
                        Debug.shared.log(message: "Failed to get file size: \(error)", type: .error)
                    }
                }

                // Remove oldest files if total size exceeds limit
                if totalSize > self.maxDiskCacheSize {
                    Debug.shared.log(message: "Image cache size (\(totalSize / 1024 / 1024) MB) exceeds limit (\(self.maxDiskCacheSize / 1024 / 1024) MB)", type: .info)

                    var currentSize = totalSize
                    let targetSize = self.maxDiskCacheSize * 80 / 100 // Target 80% of max size
                    
                    for url in sortedFiles {
                        if currentSize <= targetSize {
                            break
                        }

                        do {
                            let values = try url.resourceValues(forKeys: [.fileSizeKey])
                            if let size = values.fileSize {
                                try self.fileManager.removeItem(at: url)
                                currentSize -= UInt(size)
                                Debug.shared.log(message: "Removed old cached image: \(url.lastPathComponent)", type: .debug)
                            }
                        } catch {
                            Debug.shared.log(message: "Failed to remove cached file: \(error)", type: .error)
                        }
                    }
                }
            } catch {
                Debug.shared.log(message: "Failed to perform cache cleanup: \(error)", type: .error)
            }
        }
    }
}

// MARK: - UIImageView Extension

extension UIImageView {
    /// Load an image from a URL with caching
    /// - Parameters:
    ///   - url: The URL to load the image from
    ///   - placeholder: Optional placeholder image to use while loading
    func loadImage(from url: URL?, placeholder: UIImage? = nil) {
        // Cancel any previous loads
        if let url = url {
            ImageCache.shared.cancelLoading(for: url)
        }

        // Set placeholder immediately
        if let placeholder = placeholder {
            self.image = placeholder
        }

        // Load from cache or download
        ImageCache.shared.loadImage(from: url, placeholder: placeholder, targetSize: self.bounds.size) { [weak self] image in
            guard let self = self else { return }

            // Only update if the image view is still in the view hierarchy
            if self.window != nil {
                // Fade in the image
                if let image = image, self.image != image {
                    UIView.transition(with: self,
                                      duration: 0.2,
                                      options: .transitionCrossDissolve,
                                      animations: {
                                          self.image = image
                                      },
                                      completion: nil)
                } else if let image = image {
                    self.image = image
                }
            }
        }
    }
}
