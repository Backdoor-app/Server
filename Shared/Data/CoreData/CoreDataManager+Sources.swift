// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData

extension CoreDataManager {
    /// Clear all sources from Core Data
    func clearSources(context: NSManagedObjectContext? = nil) throws {
        let ctx = try context ?? self.context
        try clear(request: Source.fetchRequest(), context: ctx)
    }

    /// Fetch all sources sorted alphabetically by name
    func getAZSources(context: NSManagedObjectContext? = nil) -> [Source] {
        let request: NSFetchRequest<Source> = Source.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "name", ascending: true)]

        do {
            let ctx = try context ?? self.context
            return try ctx.fetch(request)
        } catch {
            Debug.shared.log(message: "Error in getAZSources: \(error)", type: .error)
            return []
        }
    }

    /// Fetch a source by its identifier
    func getSource(identifier: String, context: NSManagedObjectContext? = nil) -> Source? {
        do {
            let ctx = try context ?? self.context
            let request: NSFetchRequest<Source> = Source.fetchRequest()
            request.predicate = NSPredicate(format: "identifier == %@", identifier)
            request.fetchLimit = 1

            return try ctx.fetch(request).first
        } catch {
            Debug.shared.log(message: "Error in getSource: \(error)", type: .error)
            return nil
        }
    }

    /// Fetch and save source data from a given URL
    func getSourceData(urlString: String, completion: @escaping (Error?) -> Void) {
        guard let url = URL(string: urlString) else {
            let error = FileProcessingError.invalidPath
            Debug.shared.log(message: "Invalid URL: \(urlString)")
            completion(error)
            return
        }

        let repoManager = SourceGET()
        repoManager.downloadURL(from: url) { [weak self] result in
            guard let self = self else { return }

            switch result {
                case let .success((data, _)):
                    switch repoManager.parse(data: data) {
                        case let .success(source):
                            self.saveSource(source, url: urlString, completion: completion)
                        case let .failure(error):
                            Debug.shared.log(message: "Error parsing data: \(error)")
                            completion(error)
                    }
                case let .failure(error):
                    Debug.shared.log(message: "Error downloading data: \(error)")
                    completion(error)
            }
        }
    }

    /// Check if a source exists with a specific identifier
    private func sourceExists(withIdentifier identifier: String, context: NSManagedObjectContext) -> Bool {
        let request: NSFetchRequest<Source> = Source.fetchRequest()
        request.predicate = NSPredicate(format: "identifier == %@", identifier)
        do {
            return try context.count(for: request) > 0
        } catch {
            Debug.shared.log(message: "Error checking for existing source: \(error)")
            return false
        }
    }

    /// Create a new source entity from source data
    private func createNewSourceEntity(
        from sourceData: SourcesData,
        url: String,
        iconURL: URL?,
        context: NSManagedObjectContext
    ) -> Source {
        let newSource = Source(context: context)
        newSource.name = sourceData.name
        newSource.identifier = sourceData.identifier
        newSource.sourceURL = URL(string: url)

        if sourceData.iconURL != nil {
            newSource.iconURL = sourceData.iconURL
        } else if iconURL != nil {
            newSource.iconURL = iconURL
        }

        return newSource
    }

    /// Create a new source entity manually
    private func createNewSourceEntity(
        name: String,
        id: String,
        url: String,
        iconURL: URL?,
        context: NSManagedObjectContext
    ) -> Source {
        let newSource = Source(context: context)
        newSource.name = name
        newSource.identifier = id
        newSource.sourceURL = URL(string: url)

        newSource.iconURL = iconURL

        return newSource
    }

    /// Save SourcesData in Core Data
    private func saveSource(_ source: SourcesData, url: String, completion: @escaping (Error?) -> Void) {
        do {
            let ctx = try self.context

            ctx.perform {
                do {
                    if !self.sourceExists(withIdentifier: source.identifier, context: ctx) {
                        if !source.apps.isEmpty {
                            _ = self.createNewSourceEntity(from: source, url: url, iconURL: source.apps[0].iconURL, context: ctx)
                        } else {
                            _ = self.createNewSourceEntity(from: source, url: url, iconURL: nil, context: ctx)
                        }
                    }

                    try ctx.save()
                    completion(nil)
                } catch {
                    Debug.shared.log(message: "Error saving data: \(error)")
                    completion(error)
                }
            }
        } catch {
            Debug.shared.log(message: "Error accessing context: \(error)")
            completion(error)
        }
    }

    /// Save source data in Core Data
    public func saveSource(name: String, id: String, iconURL: URL? = nil, url: String, completion: @escaping (Error?) -> Void) {
        do {
            let ctx = try self.context

            ctx.perform {
                do {
                    if !self.sourceExists(withIdentifier: id, context: ctx) {
                        _ = self.createNewSourceEntity(name: name, id: id, url: url, iconURL: iconURL, context: ctx)
                    }

                    try ctx.save()
                    completion(nil)
                } catch {
                    Debug.shared.log(message: "Error saving data: \(error)")
                    completion(error)
                }
            }
        } catch {
            Debug.shared.log(message: "Error accessing context: \(error)")
            completion(error)
        }
    }

    /// Save source data in Core Data with proper error handling
    public func saveSourceWithThrow(name: String, id: String, iconURL: URL? = nil, url: String) throws {
        let ctx = try self.context

        if !self.sourceExists(withIdentifier: id, context: ctx) {
            _ = self.createNewSourceEntity(name: name, id: id, url: url, iconURL: iconURL, context: ctx)
        }

        try ctx.save()
    }
}
