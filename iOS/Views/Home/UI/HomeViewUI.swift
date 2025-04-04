// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class HomeViewUI {
    static let navigationBar: UINavigationBar = {
        let navBar = UINavigationBar()
        navBar.translatesAutoresizingMaskIntoConstraints = false
        navBar.barTintColor = UIColor.systemCyan.withAlphaComponent(0.8)
        navBar.titleTextAttributes = [.foregroundColor: UIColor.white, .font: UIFont.boldSystemFont(ofSize: 20)]
        navBar.layer.cornerRadius = 15
        navBar.layer.applyFuturisticShadow()
        navBar.isAccessibilityElement = true
        navBar.accessibilityLabel = "Navigation Bar"
        return navBar
    }()

    static let fileListTableView: UITableView = {
        let tableView = UITableView()
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.separatorStyle = .singleLine
        tableView.rowHeight = UITableView.automaticDimension
        tableView.estimatedRowHeight = 60
        tableView.isAccessibilityElement = true
        tableView.accessibilityLabel = "File List Table"
        return tableView
    }()

    static let activityIndicator: UIActivityIndicatorView = {
        let indicator = UIActivityIndicatorView(style: .large)
        indicator.translatesAutoresizingMaskIntoConstraints = false
        indicator.hidesWhenStopped = true
        indicator.color = .systemCyan
        indicator.isAccessibilityElement = true
        indicator.accessibilityLabel = "Activity Indicator"
        return indicator
    }()

    static let uploadButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Upload File", for: .normal)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setTitleColor(.white, for: .normal)
        button.backgroundColor = .systemCyan
        button.layer.cornerRadius = 10
        button.clipsToBounds = true
        button.titleLabel?.font = .systemFont(ofSize: 16, weight: .bold)
        button.isAccessibilityElement = true
        button.accessibilityLabel = "Upload File Button"
        return button
    }()
}

extension UIButton {
    func addGradientBackground() {
        let gradient = CAGradientLayer()
        gradient.frame = bounds
        gradient.colors = [UIColor.systemBlue.cgColor, UIColor.systemCyan.cgColor]
        gradient.startPoint = CGPoint(x: 0, y: 0)
        gradient.endPoint = CGPoint(x: 1, y: 1)
        gradient.cornerRadius = 10
        layer.insertSublayer(gradient, at: 0)
    }
}
