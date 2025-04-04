// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// Cell for improved AI learning settings
class ImprovedLearningSettingsCell: UITableViewCell {
    
    // UI Elements
    private let titleLabel = UILabel()
    private let toggleSwitch = UISwitch()
    private let descriptionLabel = UILabel()
    private let statusLabel = UILabel()
    
    // Action closure
    var toggleAction: ((Bool) -> Void)?
    
    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        // Configure cell
        backgroundColor = UIColor(named: "SettingsCell") ?? .systemBackground
        selectionStyle = .none
        
        // Configure title label
        titleLabel.font = .systemFont(ofSize: 16, weight: .medium)
        titleLabel.textColor = .label
        titleLabel.numberOfLines = 1
        
        // Configure toggle switch
        toggleSwitch.addTarget(self, action: #selector(switchToggled), for: .valueChanged)
        
        // Configure description label
        descriptionLabel.font = .systemFont(ofSize: 14)
        descriptionLabel.textColor = .secondaryLabel
        descriptionLabel.numberOfLines = 0
        
        // Configure status label
        statusLabel.font = .systemFont(ofSize: 14, weight: .medium)
        statusLabel.textColor = .systemBlue
        statusLabel.numberOfLines = 1
        statusLabel.textAlignment = .right
        
        // Add subviews
        contentView.addSubview(titleLabel)
        contentView.addSubview(toggleSwitch)
        contentView.addSubview(descriptionLabel)
        contentView.addSubview(statusLabel)
        
        // Configure constraints
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        toggleSwitch.translatesAutoresizingMaskIntoConstraints = false
        descriptionLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            titleLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            titleLabel.trailingAnchor.constraint(equalTo: toggleSwitch.leadingAnchor, constant: -8),
            
            toggleSwitch.centerYAnchor.constraint(equalTo: titleLabel.centerYAnchor),
            toggleSwitch.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            
            descriptionLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 4),
            descriptionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            descriptionLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            
            statusLabel.topAnchor.constraint(equalTo: descriptionLabel.bottomAnchor, constant: 8),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            statusLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -12)
        ])
    }
    
    func configure(title: String, description: String, isOn: Bool, status: String = "") {
        titleLabel.text = title
        descriptionLabel.text = description
        toggleSwitch.isOn = isOn
        statusLabel.text = status
        
        // Hide status label if empty
        statusLabel.isHidden = status.isEmpty
    }
    
    @objc private func switchToggled() {
        toggleAction?(toggleSwitch.isOn)
    }
}
