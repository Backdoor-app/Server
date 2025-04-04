// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

// MARK: - UIView Extensions for AutoLayout
extension UIView {
    /// Set up constraints with native AutoLayout
    /// - Parameter constraints: Array of constraints to activate
    func setupConstraints(_ constraints: [NSLayoutConstraint]) {
        self.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate(constraints)
    }
    
    /// Set up constraints with a closure
    /// - Parameter setup: Closure that returns constraints to activate
    func setupConstraints(_ setup: (UIView) -> [NSLayoutConstraint]) {
        self.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate(setup(self))
    }
    
    /// Update existing constraints
    /// - Parameter update: Closure that performs constraint updates
    func updateConstraints(_ update: () -> Void) {
        update()
        self.layoutIfNeeded()
    }
    
    /// Create a stack view with standard configuration
    /// - Parameters:
    ///   - axis: Axis for the stack view
    ///   - spacing: Spacing between items
    ///   - views: Views to add to the stack
    ///   - insets: Insets to apply to the stack view
    /// - Returns: Configured UIStackView
    static func createStack(axis: NSLayoutConstraint.Axis,
                           spacing: CGFloat = 8,
                           views: [UIView],
                           insets: UIEdgeInsets = .zero) -> UIStackView {
        let stackView = UIStackView(arrangedSubviews: views)
        stackView.axis = axis
        stackView.spacing = spacing
        stackView.layoutMargins = insets
        stackView.isLayoutMarginsRelativeArrangement = true
        return stackView
    }
}

// MARK: - Internal Animation Helper
class InternalAnimationHelper {
    /// Add an animated icon to a view (replacement for Lottie)
    /// - Parameters:
    ///   - systemName: SF Symbol name
    ///   - view: Parent view to add the animation to
    ///   - loopMode: Animation loop mode (continuous, once, etc.)
    ///   - size: Size for the animation view
    /// - Returns: The configured UIImageView
    static func addAnimation(systemName: String, to view: UIView,
                            loopMode: UIView.AnimationRepeatCount = .infinity,
                            size: CGSize? = nil) -> UIImageView {
        // Create image view with SF Symbol
        let imageView = UIImageView()
        if let image = UIImage(systemName: systemName) {
            imageView.image = image
        } else {
            // Fallback icon if SF Symbol not available
            imageView.backgroundColor = .systemBlue.withAlphaComponent(0.2)
            imageView.layer.cornerRadius = (size?.width ?? 50) / 2
        }
        
        imageView.tintColor = .systemBlue
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        
        // Add to parent view
        view.addSubview(imageView)
        
        // Setup constraints with native AutoLayout
        if let size = size {
            NSLayoutConstraint.activate([
                imageView.widthAnchor.constraint(equalToConstant: size.width),
                imageView.heightAnchor.constraint(equalToConstant: size.height),
                imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                imageView.centerYAnchor.constraint(equalTo: view.centerYAnchor)
            ])
        } else {
            NSLayoutConstraint.activate([
                imageView.topAnchor.constraint(equalTo: view.topAnchor),
                imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
            ])
        }
        
        // Add animation
        // Using animation options directly instead of creating unused repeatCount variable
        UIView.animate(withDuration: 1.5, delay: 0, options: [.autoreverse, .repeat, .curveEaseInOut], animations: {
            imageView.transform = CGAffineTransform(scaleX: 1.2, y: 1.2)
        }, completion: nil)
        
        return imageView
    }
    
    /// Show an animated loading indicator
    /// - Parameters:
    ///   - view: View to add the loader to
    ///   - message: Optional message to display
    /// - Returns: Container view with the animation that can be removed later
    static func showLoader(in view: UIView, message: String? = nil) -> UIView {
        // Create container for the loader
        let container = UIView()
        container.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        container.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(container)
        
        // Set constraints for full screen
        NSLayoutConstraint.activate([
            container.topAnchor.constraint(equalTo: view.topAnchor),
            container.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            container.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            container.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        
        // Create content container with blur effect
        let blurEffect = UIBlurEffect(style: .systemMaterial)
        let contentContainer = UIVisualEffectView(effect: blurEffect)
        contentContainer.layer.cornerRadius = 16
        contentContainer.clipsToBounds = true
        contentContainer.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(contentContainer)
        
        // Set up constraints for the content container
        NSLayoutConstraint.activate([
            contentContainer.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            contentContainer.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            contentContainer.widthAnchor.constraint(equalToConstant: 200),
            contentContainer.heightAnchor.constraint(equalToConstant: message != nil ? 200 : 150)
        ])
        
        // Add activity indicator to the content container
        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.startAnimating()
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        contentContainer.contentView.addSubview(activityIndicator)
        
        // Set up activity indicator constraints
        NSLayoutConstraint.activate([
            activityIndicator.centerXAnchor.constraint(equalTo: contentContainer.contentView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: contentContainer.contentView.centerYAnchor, 
                                                     constant: message != nil ? -20 : 0),
            activityIndicator.widthAnchor.constraint(equalToConstant: 50),
            activityIndicator.heightAnchor.constraint(equalToConstant: 50)
        ])
        
        // Add message label if provided
        if let message = message {
            let label = UILabel()
            label.text = message
            label.textAlignment = .center
            label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
            label.textColor = .label
            label.translatesAutoresizingMaskIntoConstraints = false
            contentContainer.contentView.addSubview(label)
            
            // Set up label constraints
            NSLayoutConstraint.activate([
                label.centerXAnchor.constraint(equalTo: contentContainer.contentView.centerXAnchor),
                label.topAnchor.constraint(equalTo: activityIndicator.bottomAnchor, constant: 20),
                label.leadingAnchor.constraint(equalTo: contentContainer.contentView.leadingAnchor, constant: 16),
                label.trailingAnchor.constraint(equalTo: contentContainer.contentView.trailingAnchor, constant: -16)
            ])
        }
        
        return container
    }
    
    /// Hide the loader
    /// - Parameter container: Container view returned by showLoader
    static func hideLoader(_ container: UIView) {
        UIView.animate(withDuration: 0.3, animations: {
            container.alpha = 0
        }, completion: { _ in
            container.removeFromSuperview()
        })
    }
}

// MARK: - Internal UI Components
class InternalUIComponents {
    /// Create a beautifully styled button with gradient
    /// - Parameters:
    ///   - title: Button title
    ///   - colors: Gradient colors (default blue gradient)
    ///   - cornerRadius: Corner radius (default 12)
    ///   - fontSize: Font size (default 16)
    /// - Returns: Configured button
    static func createGradientButton(title: String,
                                    colors: [UIColor] = [.systemBlue, UIColor(red: 0, green: 0.5, blue: 1, alpha: 1)],
                                    cornerRadius: CGFloat = 12,
                                    fontSize: CGFloat = 16) -> UIButton {
        let button = UIButton(type: .custom)
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
        button.layer.cornerRadius = cornerRadius
        button.clipsToBounds = true
        
        // Create gradient layer
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = colors.map { $0.cgColor }
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.cornerRadius = cornerRadius
        
        // Ensure the gradient is applied after layout
        button.layer.insertSublayer(gradientLayer, at: 0)
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 3)
        button.layer.shadowOpacity = 0.2
        button.layer.shadowRadius = 4
        button.layer.masksToBounds = false
        
        // Update gradient frame when layout changes
        button.layoutIfNeeded()
        gradientLayer.frame = button.bounds
        
        return button
    }
    
    /// Create a card view with shadow
    /// - Parameters:
    ///   - backgroundColor: Card background color
    ///   - cornerRadius: Corner radius
    /// - Returns: Configured card view
    static func createCardView(backgroundColor: UIColor = .systemBackground,
                             cornerRadius: CGFloat = 16) -> UIView {
        let cardView = UIView()
        cardView.backgroundColor = backgroundColor
        cardView.layer.cornerRadius = cornerRadius
        
        // Add shadow
        cardView.layer.shadowColor = UIColor.black.cgColor
        cardView.layer.shadowOffset = CGSize(width: 0, height: 4)
        cardView.layer.shadowOpacity = 0.1
        cardView.layer.shadowRadius = 6
        cardView.layer.masksToBounds = false
        
        return cardView
    }
    
    /// Create a beautiful text field with floating label
    /// - Parameters:
    ///   - placeholder: Placeholder text
    ///   - backgroundColor: Background color
    ///   - borderColor: Border color
    /// - Returns: Configured text field with container
    static func createFloatingTextField(placeholder: String,
                                      backgroundColor: UIColor = .systemBackground,
                                      borderColor: UIColor = .systemGray4) -> UIView {
        let container = UIView()
        container.backgroundColor = backgroundColor
        container.layer.cornerRadius = 8
        container.layer.borderWidth = 1
        container.layer.borderColor = borderColor.cgColor
        
        let textField = UITextField()
        textField.placeholder = placeholder
        textField.borderStyle = .none
        textField.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(textField)
        
        NSLayoutConstraint.activate([
            textField.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            textField.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            textField.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -16),
            textField.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12)
        ])
        
        return container
    }
}

/// Extension to define animation repeat options
extension UIView {
    enum AnimationRepeatCount: Equatable {
        case once
        case finite(count: Int)
        case infinity
        
        var floatValue: Float {
            switch self {
            case .once:
                return 1.0
            case .finite(let count):
                return Float(count)
            case .infinity:
                return .infinity
            }
        }
        
        // Implementation of Equatable
        static func == (lhs: UIView.AnimationRepeatCount, rhs: UIView.AnimationRepeatCount) -> Bool {
            switch (lhs, rhs) {
            case (.once, .once):
                return true
            case (.infinity, .infinity):
                return true
            case let (.finite(lhsCount), .finite(rhsCount)):
                return lhsCount == rhsCount
            default:
                return false
            }
        }
    }
}
