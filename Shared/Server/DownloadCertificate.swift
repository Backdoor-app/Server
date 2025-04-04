// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

func getCertificates(completion: (() -> Void)? = nil) {
    let sourceGET = SourceGET()
    let uri = URL(string: "https://backloop.dev/pack.json")!

    func writeToFile(content: String, filename: String) throws {
        let path = getDocumentsDirectory().appendingPathComponent(filename)
        try content.write(to: path, atomically: true, encoding: .utf8)
    }

    // Create default empty files to prevent crashes if download fails
    func createDefaultFiles() {
        do {
            let emptyString = ""
            try writeToFile(content: emptyString, filename: "server.pem")
            try writeToFile(content: emptyString, filename: "server.crt")
            try writeToFile(content: "default.backdoor.local", filename: "commonName.txt")
            Debug.shared.log(message: "Created default certificate files as fallback", type: .warning)
        } catch {
            Debug.shared.log(message: "Error creating default certificate files: \(error.localizedDescription)", type: .error)
        }
    }

    // First create default files to ensure we have something
    createDefaultFiles()

    // Then try to download the real certificates
    sourceGET.downloadURL(from: uri) { result in
        defer {
            // Always call completion handler
            completion?()
        }

        switch result {
            case let .success((data, _)):
                switch sourceGET.parseCert(data: data) {
                    case let .success(serverPack):
                        do {
                            try writeToFile(content: serverPack.key, filename: "server.pem")
                            try writeToFile(content: serverPack.cert, filename: "server.crt")
                            try writeToFile(content: serverPack.info.domains.commonName, filename: "commonName.txt")
                            Debug.shared.log(message: "Successfully downloaded and saved certificates", type: .success)
                        } catch {
                            Debug.shared.log(message: "Error writing certificate files: \(error.localizedDescription)", type: .error)
                        }
                    case let .failure(error):
                        Debug.shared.log(message: "Error parsing certificate: \(error.localizedDescription)", type: .error)
                }
            case let .failure(error):
                Debug.shared.log(message: "Error fetching certificates from \(uri): \(error.localizedDescription)", type: .error)
        }
    }
}

func getDocumentsDirectory() -> URL {
    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    let documentsDirectory = paths[0]
    return documentsDirectory
}
