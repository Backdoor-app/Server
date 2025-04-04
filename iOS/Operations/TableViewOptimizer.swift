// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

/// A class that optimizes UITableView loading and scrolling performance
final class TableViewOptimizer: NSObject {
    // MARK: - Configuration

    /// Configuration for the table view optimizer
    struct Configuration {
        /// Number of items to load per page
        var pageSize: Int = 20

        /// Whether to prefetch items
        var enablePrefetching: Bool = true

        /// Number of items to prefetch
        var prefetchDistance: Int = 10

        /// Whether to use skeleton loading cells
        var useSkeletonLoading: Bool = true

        /// Maximum number of cells to keep in memory
        var maxCachedCells: Int = 50

        /// Batch size for Core Data fetch requests
        var fetchBatchSize: Int = 20
    }

    // MARK: - Properties

    /// The configuration for this optimizer
    private let configuration: Configuration

    /// The UITableView being optimized
    private weak var tableView: UITableView?

    /// The fetched results controller for Core Data integration
    private var fetchedResultsController: NSFetchedResultsController<NSManagedObject>?

    /// Queue for background operations
    private let backgroundQueue = DispatchQueue(label: "com.backdoor.TableViewOptimizer", qos: .userInitiated)

    /// Flag to track whether the table is currently loading data
    private var isLoading = false

    /// Current page for pagination
    private var currentPage = 0

    /// Total number of items available
    private var totalItems = 0

    /// Cache of cell heights for variable height cells
    private var cellHeightCache = [IndexPath: CGFloat]()

    // MARK: - Initialization

    /// Initialize with a table view and configuration
    /// - Parameters:
    ///   - tableView: The table view to optimize
    ///   - configuration: Optional custom configuration
    init(tableView: UITableView, configuration: Configuration = Configuration()) {
        self.tableView = tableView
        self.configuration = configuration

        super.init()

        // Set up the table view for optimized performance after super.init
        setupTableView()
    }

    // MARK: - Setup

    /// Set up the table view for optimized performance
    private func setupTableView() {
        guard let tableView = tableView else { return }

        // Enable prefetching if configured
        if configuration.enablePrefetching {
            tableView.prefetchDataSource = self
        } else {
            tableView.isPrefetchingEnabled = false
        }

        // Optimize table view performance
        tableView.estimatedRowHeight = 100
        tableView.rowHeight = UITableView.automaticDimension

        // Optimize scrolling performance
        tableView.showsVerticalScrollIndicator = true
        tableView.showsHorizontalScrollIndicator = false

        // Configure content insets for better appearance
        tableView.contentInset = UIEdgeInsets(top: 0, left: 0, bottom: 20, right: 0)

        Debug.shared.log(message: "TableViewOptimizer: Table view configured for optimized performance", type: .debug)
    }

    // MARK: - Core Data Integration

    /// Configure with a fetched results controller for Core Data integration
    /// - Parameters:
    ///   - fetchRequest: The fetch request
    ///   - managedObjectContext: The managed object context
    ///   - sectionNameKeyPath: Optional key path for sections
    ///   - cacheName: Optional cache name
    func configureFetchedResultsController(
        fetchRequest: NSFetchRequest<NSManagedObject>,
        managedObjectContext: NSManagedObjectContext,
        sectionNameKeyPath: String? = nil,
        cacheName: String? = nil
    ) {
        // Configure fetch request for optimization
        fetchRequest.fetchBatchSize = configuration.fetchBatchSize

        // Create fetched results controller
        fetchedResultsController = NSFetchedResultsController(
            fetchRequest: fetchRequest,
            managedObjectContext: managedObjectContext,
            sectionNameKeyPath: sectionNameKeyPath,
            cacheName: cacheName
        )

        // Set the delegate
        fetchedResultsController?.delegate = self

        // Perform initial fetch
        performFetch()
    }

    /// Perform fetch with the fetched results controller
    func performFetch() {
        guard let fetchedResultsController = fetchedResultsController else {
            Debug.shared.log(message: "TableViewOptimizer: No fetched results controller configured", type: .warning)
            return
        }

        // Perform fetch in background
        backgroundQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                try fetchedResultsController.performFetch()

                // Update UI on main thread
                DispatchQueue.main.async {
                    self.tableView?.reloadData()
                }

                Debug.shared.log(message: "TableViewOptimizer: Fetch completed successfully", type: .debug)
            } catch {
                Debug.shared.log(message: "TableViewOptimizer: Failed to fetch data: \(error)", type: .error)
            }
        }
    }

    // MARK: - Pagination

    /// Load the next page of data
    func loadNextPage() {
        guard !isLoading, hasMorePages() else { return }

        isLoading = true
        currentPage += 1

        Debug.shared.log(message: "TableViewOptimizer: Loading page \(currentPage)", type: .debug)

        // Simulate loading delay for better UX
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self = self else { return }

            // Update table view with new data
            self.tableView?.reloadData()
            self.isLoading = false

            Debug.shared.log(message: "TableViewOptimizer: Loaded page \(self.currentPage)", type: .debug)
        }
    }

    /// Check if there are more pages to load
    /// - Returns: True if there are more pages
    func hasMorePages() -> Bool {
        let itemsLoaded = currentPage * configuration.pageSize
        return itemsLoaded < totalItems
    }

    // MARK: - Utility Methods

    /// Get the object at the given index path
    /// - Parameter indexPath: The index path
    /// - Returns: The managed object at the index path
    func object(at indexPath: IndexPath) -> NSManagedObject? {
        return fetchedResultsController?.object(at: indexPath)
    }

    /// Get the number of sections
    /// - Returns: The number of sections
    func numberOfSections() -> Int {
        return fetchedResultsController?.sections?.count ?? 0
    }

    /// Get the number of objects in the given section
    /// - Parameter section: The section index
    /// - Returns: The number of objects in the section
    func numberOfObjects(in section: Int) -> Int {
        return fetchedResultsController?.sections?[section].numberOfObjects ?? 0
    }

    /// Clear the cell height cache
    func clearCellHeightCache() {
        cellHeightCache.removeAll()
    }

    /// Cache the height for a cell at the given index path
    /// - Parameters:
    ///   - height: The height to cache
    ///   - indexPath: The index path
    func cacheHeight(_ height: CGFloat, for indexPath: IndexPath) {
        // Limit cache size
        if cellHeightCache.count > configuration.maxCachedCells {
            cellHeightCache.removeAll()
        }

        cellHeightCache[indexPath] = height
    }

    /// Get the cached height for a cell at the given index path
    /// - Parameter indexPath: The index path
    /// - Returns: The cached height, if available
    func cachedHeight(for indexPath: IndexPath) -> CGFloat? {
        return cellHeightCache[indexPath]
    }

    /// Reset the pagination state
    func resetPagination() {
        currentPage = 0
        isLoading = false
        clearCellHeightCache()
    }

    /// Set the total number of items available
    /// - Parameter count: The total number of items
    func setTotalItems(_ count: Int) {
        totalItems = count
    }
}

// MARK: - UITableViewDataSourcePrefetching

extension TableViewOptimizer: UITableViewDataSourcePrefetching {
    func tableView(_: UITableView, prefetchRowsAt indexPaths: [IndexPath]) {
        guard configuration.enablePrefetching else { return }

        // Check if any of the prefetched rows would trigger loading the next page
        let maxIndexPath = indexPaths.max { $0.row < $1.row }
        if let maxRow = maxIndexPath?.row {
            let currentItems = (currentPage + 1) * configuration.pageSize
            if maxRow >= currentItems - configuration.prefetchDistance {
                loadNextPage()
            }
        }

        // Prefetch images or other data for these rows
        for indexPath in indexPaths {
            if let object = fetchedResultsController?.object(at: indexPath) {
                // Check if object has an image URL property
                if let imageURLString = object.value(forKey: "iconURL") as? String,
                   let imageURL = URL(string: imageURLString)
                {
                    // Prefetch image for this object
                    ImageCache.shared.loadImage(from: imageURL, completion: { _ in })
                }
            }
        }
    }

    func tableView(_: UITableView, cancelPrefetchingForRowsAt indexPaths: [IndexPath]) {
        // Cancel any prefetching operations for these rows
        for indexPath in indexPaths {
            if let object = fetchedResultsController?.object(at: indexPath) {
                if let imageURLString = object.value(forKey: "iconURL") as? String,
                   let imageURL = URL(string: imageURLString)
                {
                    ImageCache.shared.cancelLoading(for: imageURL)
                }
            }
        }
    }
}

// MARK: - NSFetchedResultsControllerDelegate

extension TableViewOptimizer: NSFetchedResultsControllerDelegate {
    func controllerWillChangeContent(_: NSFetchedResultsController<NSFetchRequestResult>) {
        tableView?.beginUpdates()
    }

    func controller(_: NSFetchedResultsController<NSFetchRequestResult>,
                    didChange _: NSFetchedResultsSectionInfo,
                    atSectionIndex sectionIndex: Int,
                    for type: NSFetchedResultsChangeType)
    {
        switch type {
            case .insert:
                tableView?.insertSections(IndexSet(integer: sectionIndex), with: .fade)
            case .delete:
                tableView?.deleteSections(IndexSet(integer: sectionIndex), with: .fade)
            case .move, .update:
                tableView?.reloadSections(IndexSet(integer: sectionIndex), with: .fade)
            @unknown default:
                tableView?.reloadSections(IndexSet(integer: sectionIndex), with: .fade)
        }
    }

    func controller(_: NSFetchedResultsController<NSFetchRequestResult>,
                    didChange _: Any,
                    at indexPath: IndexPath?,
                    for type: NSFetchedResultsChangeType,
                    newIndexPath: IndexPath?)
    {
        switch type {
            case .insert:
                if let newIndexPath = newIndexPath {
                    tableView?.insertRows(at: [newIndexPath], with: .fade)
                }
            case .delete:
                if let indexPath = indexPath {
                    tableView?.deleteRows(at: [indexPath], with: .fade)
                }
            case .update:
                if let indexPath = indexPath {
                    tableView?.reloadRows(at: [indexPath], with: .fade)
                }
            case .move:
                if let indexPath = indexPath, let newIndexPath = newIndexPath {
                    tableView?.moveRow(at: indexPath, to: newIndexPath)
                }
            @unknown default:
                if let indexPath = indexPath {
                    tableView?.reloadRows(at: [indexPath], with: .fade)
                }
        }
    }

    func controllerDidChangeContent(_: NSFetchedResultsController<NSFetchRequestResult>) {
        tableView?.endUpdates()
    }
}

// MARK: - SkeletonLoadingCell

/// A table view cell that shows a skeleton loading animation
class SkeletonLoadingCell: UITableViewCell {
    // MARK: - UI Components

    private let containerView = UIView()
    private let iconView = UIView()
    private let titleView = UIView()
    private let subtitleView = UIView()

    // MARK: - Animation Layers

    private var animationLayers = [CAGradientLayer]()

    // MARK: - Initialization

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupViews()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Setup

    private func setupViews() {
        selectionStyle = .none
        backgroundColor = .clear

        // Add container view
        containerView.backgroundColor = .systemBackground
        containerView.layer.cornerRadius = 8
        containerView.clipsToBounds = true
        containerView.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(containerView)

        // Add skeleton views
        iconView.backgroundColor = .systemGray5
        iconView.layer.cornerRadius = 25
        iconView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(iconView)

        titleView.backgroundColor = .systemGray5
        titleView.layer.cornerRadius = 4
        titleView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(titleView)

        subtitleView.backgroundColor = .systemGray5
        subtitleView.layer.cornerRadius = 4
        subtitleView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(subtitleView)

        // Set up constraints
        NSLayoutConstraint.activate([
            containerView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            containerView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            containerView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            containerView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8),

            iconView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: 12),
            iconView.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            iconView.widthAnchor.constraint(equalToConstant: 50),
            iconView.heightAnchor.constraint(equalToConstant: 50),

            titleView.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: 12),
            titleView.topAnchor.constraint(equalTo: containerView.topAnchor, constant: 16),
            titleView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor, constant: -60),
            titleView.heightAnchor.constraint(equalToConstant: 16),

            subtitleView.leadingAnchor.constraint(equalTo: titleView.leadingAnchor),
            subtitleView.topAnchor.constraint(equalTo: titleView.bottomAnchor, constant: 8),
            subtitleView.widthAnchor.constraint(equalTo: titleView.widthAnchor, multiplier: 0.7),
            subtitleView.heightAnchor.constraint(equalToConstant: 12),
        ])

        // Add gradient layers for shimmer effect
        addGradientLayers()
    }

    private func addGradientLayers() {
        let views = [iconView, titleView, subtitleView]

        for view in views {
            let gradientLayer = CAGradientLayer()
            gradientLayer.frame = view.bounds
            gradientLayer.startPoint = CGPoint(x: 0, y: 0.5)
            gradientLayer.endPoint = CGPoint(x: 1, y: 0.5)

            // Set colors for shimmer effect
            gradientLayer.colors = [
                UIColor.systemGray5.cgColor,
                UIColor.systemGray4.cgColor,
                UIColor.systemGray5.cgColor,
            ]

            // Set locations for shimmer effect
            gradientLayer.locations = [0, 0.5, 1]

            // Add gradient layer to view
            view.layer.addSublayer(gradientLayer)

            // Store for animation
            animationLayers.append(gradientLayer)
        }
    }

    // MARK: - Animation

    func startAnimating() {
        // Create animation for shimmer effect
        let animation = CABasicAnimation(keyPath: "locations")
        animation.fromValue = [-1.0, -0.5, 0.0] // Start positions
        animation.toValue = [1.0, 1.5, 2.0] // End positions
        animation.duration = 1.5
        animation.repeatCount = .infinity

        // Add animation to gradient layers
        for layer in animationLayers {
            layer.add(animation, forKey: "shimmerAnimation")
        }
    }

    func stopAnimating() {
        // Remove animations
        for layer in animationLayers {
            layer.removeAnimation(forKey: "shimmerAnimation")
        }
    }

    // MARK: - Layout

    override func layoutSubviews() {
        super.layoutSubviews()

        // Update gradient layer frames
        for (index, layer) in animationLayers.enumerated() {
            switch index {
                case 0: layer.frame = iconView.bounds
                case 1: layer.frame = titleView.bounds
                case 2: layer.frame = subtitleView.bounds
                default: break
            }
        }
    }

    // MARK: - Lifecycle

    override func didMoveToWindow() {
        super.didMoveToWindow()

        if window != nil {
            startAnimating()
        } else {
            stopAnimating()
        }
    }
}
