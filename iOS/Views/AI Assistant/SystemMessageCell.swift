// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import CoreData
import UIKit

class SystemMessageCell: UITableViewCell {
    private let messageLabel = UILabel()
    private let containerView = UIView()
    private var animationImageView: UIImageView?

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
        
        // Add a container view for better visual grouping and animations
        containerView.backgroundColor = .clear
        contentView.addSubview(containerView)
        
        // Configure message label
        messageLabel.numberOfLines = 0
        messageLabel.textColor = .systemGray
        messageLabel.font = .systemFont(ofSize: 14, weight: .medium)
        messageLabel.textAlignment = .center
        messageLabel.translatesAutoresizingMaskIntoConstraints = false
        
        containerView.addSubview(messageLabel)
        containerView.translatesAutoresizingMaskIntoConstraints = false

        // Setup constraints with native AutoLayout
        NSLayoutConstraint.activate([
            containerView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            containerView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            containerView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 3),
            containerView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -3),
            
            messageLabel.centerXAnchor.constraint(equalTo: containerView.centerXAnchor),
            messageLabel.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            messageLabel.leadingAnchor.constraint(greaterThanOrEqualTo: containerView.leadingAnchor, constant: 12),
            messageLabel.trailingAnchor.constraint(lessThanOrEqualTo: containerView.trailingAnchor, constant: -12),
            messageLabel.topAnchor.constraint(equalTo: containerView.topAnchor, constant: 3),
            messageLabel.bottomAnchor.constraint(equalTo: containerView.bottomAnchor, constant: -3)
        ])
    }

    func configure(with message: ChatMessage) {
        // Clear any existing animation
        clearAnimation()
        
        // Process the message content
        let content = message.content ?? ""
        
        // Handle different system message types with specialized styling
        if content.contains("error") || content.contains("failed") || content.contains("Error:") {
            // Style for error messages
            messageLabel.textColor = .systemRed
            messageLabel.text = content
            
            // Add error icon/animation
            addIconAnimation(iconName: "exclamationmark.triangle.fill", tintColor: .systemRed)
            
        } else if content.contains("success") || content.contains("completed") {
            // Style for success messages
            messageLabel.textColor = .systemGreen
            messageLabel.text = content
            
            // Add success icon/animation
            addIconAnimation(iconName: "checkmark.circle.fill", tintColor: .systemGreen)
            
        } else if content == "Assistant is thinking..." {
            // This should be handled by AIMessageCell, but just in case
            messageLabel.textColor = .systemGray
            messageLabel.text = content
            
        } else {
            // Default styling
            messageLabel.textColor = .systemGray
            messageLabel.text = content
        }
    }
    
    private func addIconAnimation(iconName: String, tintColor: UIColor) {
        // Create an image view with SF Symbol
        let imageView = UIImageView()
        if let image = UIImage(systemName: iconName) {
            imageView.image = image
        } else {
            // Fallback if SF Symbol not available
            imageView.backgroundColor = tintColor
            imageView.layer.cornerRadius = 10
        }
        
        imageView.tintColor = tintColor
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(imageView)
        animationImageView = imageView
        
        // Position icon next to the text
        NSLayoutConstraint.activate([
            imageView.leadingAnchor.constraint(equalTo: messageLabel.trailingAnchor, constant: 4),
            imageView.centerYAnchor.constraint(equalTo: messageLabel.centerYAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 20),
            imageView.heightAnchor.constraint(equalToConstant: 20)
        ])
        
        // Add simple pulse animation
        UIView.animate(withDuration: 0.5, delay: 0, options: [.autoreverse, .repeat], animations: {
            imageView.transform = CGAffineTransform(scaleX: 1.2, y: 1.2)
        }, completion: nil)
    }
    
    private func clearAnimation() {
        // Remove animation view if exists
        animationImageView?.layer.removeAllAnimations()
        animationImageView?.removeFromSuperview()
        animationImageView = nil
    }
    
    override func prepareForReuse() {
        super.prepareForReuse()
        clearAnimation()
        messageLabel.textColor = .systemGray
    }
}
