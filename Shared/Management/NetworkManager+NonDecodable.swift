// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

// Extension to NetworkManager for batch requests that don't need Decodable conformance
extension NetworkManager {
    /// Perform a network request without requiring Decodable conformance
    /// - Parameters:
    ///   - request: The URL request to perform
    ///   - caching: Whether to use caching (default is based on configuration)
    ///   - completion: Completion handler with the result
    /// - Returns: A cancellable task identifier
    @discardableResult
    func performRequestWithoutDecoding(
        _ request: URLRequest,
        caching: Bool? = nil,
        completion: @escaping (Result<Any, Error>) -> Void
    ) -> URLSessionTask? {
        // Determine whether to use caching
        let useCache = caching ?? (configuration.useCache && request.httpMethod?.uppercased() == "GET")
        
        // Check if request is already in progress
        let existingTask = self.operationQueueAccessQueue.sync { self.activeOperations[request] }
        if let existingTask = existingTask {
            Debug.shared.log(message: "Request already in progress: \(request.url?.absoluteString ?? "Unknown URL")", type: .debug)
            return existingTask
        }
        
        // Check cache if caching is enabled
        if useCache, let url = request.url {
            let cacheKey = NSString(string: url.absoluteString)
            if let cachedResponse = self.responseCache.object(forKey: cacheKey) {
                if !self.isCacheExpired(cachedResponse) {
                    Debug.shared.log(message: "Cache hit for non-decodable request: \(url.absoluteString)", type: .debug)
                    
                    do {
                        if let jsonObject = try JSONSerialization.jsonObject(with: cachedResponse.data) as? [String: Any] {
                            DispatchQueue.main.async {
                                completion(.success(jsonObject))
                            }
                            return nil
                        } else if let jsonArray = try JSONSerialization.jsonObject(with: cachedResponse.data) as? [Any] {
                            DispatchQueue.main.async {
                                completion(.success(jsonArray))
                            }
                            return nil
                        }
                    } catch {
                        Debug.shared.log(message: "Failed to parse cached response: \(error.localizedDescription)", type: .error)
                        // Continue with network request if parsing fails
                    }
                }
            }
        }
        
        // Create a task using the configured session
        let task = self.session.dataTask(with: request) { [weak self] (data: Data?, response: URLResponse?, error: Error?) in
            guard let self = self else { return }
            
            // Remove from active operations if it was tracked
            self.operationQueueAccessQueue.sync {
                self.activeOperations.removeValue(forKey: request)
            }
            
            // Handle network error
            if let error = error {
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
            if useCache, let url = request.url {
                self.cacheResponse(data: data, for: request)
            }

            // Parse the response using JSONSerialization instead of JSONDecoder
            // Do this on a background queue to avoid blocking the main thread
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let jsonObject: Any
                    if let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        jsonObject = dict
                    } else if let array = try JSONSerialization.jsonObject(with: data) as? [Any] {
                        jsonObject = array
                    } else {
                        jsonObject = try JSONSerialization.jsonObject(with: data)
                    }
                    
                    DispatchQueue.main.async {
                        completion(.success(jsonObject))
                    }
                } catch {
                    Debug.shared.log(message: "Failed to parse response: \(error.localizedDescription)", type: .error)
                    DispatchQueue.main.async {
                        completion(.failure(NetworkError.decodingError(error)))
                    }
                }
            }
        }
        
        // Add to active operations
        self.operationQueueAccessQueue.sync {
            self.activeOperations[request] = task
        }

        // Start the task
        task.resume()

        return task
    }
}
