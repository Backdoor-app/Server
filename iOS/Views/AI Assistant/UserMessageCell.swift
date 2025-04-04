// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

class UserMessageCell: UITableViewCell {
    private let bubbleView = UIView()
    private let messageLabel = UILabel()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupViews()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupViews() {
        selectionStyle = .none
        backgroundColor = .clear

        // Create the bubble view
        bubbleView.layer.cornerRadius = 16
        bubbleView.layer.maskedCorners = [.layerMinXMinYCorner, .layerMinXMaxYCorner, .layerMaxXMinYCorner]
        
        // Add gradient to bubble (manual implementation)
        addGradientToBubble()
        
        // Add subtle shadow for depth
        bubbleView.layer.shadowColor = UIColor.black.withAlphaComponent(0.2).cgColor
        bubbleView.layer.shadowOffset = CGSize(width: 0, height: 2)
        bubbleView.layer.shadowRadius = 4
        bubbleView.layer.shadowOpacity = 0.5
        bubbleView.layer.masksToBounds = false

        // Configure message label
        messageLabel.numberOfLines = 0
        messageLabel.textColor = .white
        messageLabel.font = .systemFont(ofSize: 16)

        // Add subviews
        contentView.addSubview(bubbleView)
        bubbleView.addSubview(messageLabel)
        
        // Setup for Auto Layout
        bubbleView.translatesAutoresizingMaskIntoConstraints = false
        messageLabel.translatesAutoresizingMaskIntoConstraints = false

        // Setup constraints with native AutoLayout
        NSLayoutConstraint.activate([
            bubbleView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),
            bubbleView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 6),
            bubbleView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -6),
            bubbleView.widthAnchor.constraint(lessThanOrEqualToConstant: 280),
            
            messageLabel.leadingAnchor.constraint(equalTo: bubbleView.leadingAnchor, constant: 12),
            messageLabel.trailingAnchor.constraint(equalTo: bubbleView.trailingAnchor, constant: -12),
            messageLabel.topAnchor.constraint(equalTo: bubbleView.topAnchor, constant: 8),
            messageLabel.bottomAnchor.constraint(equalTo: bubbleView.bottomAnchor, constant: -8)
        ])
    }

    private func addGradientToBubble() {
        // Create gradient layer
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [
            UIColor.systemBlue.cgColor,
            UIColor(red: 0.1, green: 0.6, blue: 1.0, alpha: 1.0).cgColor
        ]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.cornerRadius = bubbleView.layer.cornerRadius
        
        // Set frame and insert at index 0
        gradientLayer.frame = bubbleView.bounds
        bubbleView.layer.insertSublayer(gradientLayer, at: 0)
        
        // Update gradient when layout changes
        bubbleView.layer.layoutIfNeeded()
    }
    
    // Update gradient frame on layout change
    override func layoutSubviews() {
        super.layoutSubviews()
        if let gradientLayer = bubbleView.layer.sublayers?.first as? CAGradientLayer {
            gradientLayer.frame = bubbleView.bounds
        }
    }

    func configure(with message: ChatMessage) {
        messageLabel.text = message.content
    }
}
