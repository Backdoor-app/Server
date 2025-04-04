// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

// MARK: - UIView Extensions 
extension UIView {
    /// Add and setup constraints for a child view in a single call
    /// - Parameters:
    ///   - child: Child view to add
    ///   - setup: Closure for configuring constraints
    func addSubviewWithConstraints(_ child: UIView, setup: (UIView) -> [NSLayoutConstraint]) {
        child.translatesAutoresizingMaskIntoConstraints = false
        addSubview(child)
        NSLayoutConstraint.activate(setup(child))
    }
    
    /// Create a stack of views with equal spacing
    /// - Parameters:
    ///   - views: Views to include in the stack
    ///   - axis: Stack axis (horizontal or vertical)
    ///   - spacing: Spacing between views
    ///   - distribution: Distribution type
    ///   - alignment: Alignment type
    /// - Returns: Configured stack view
    func createStack(
        with views: [UIView],
        axis: NSLayoutConstraint.Axis,
        spacing: CGFloat = 8,
        distribution: UIStackView.Distribution = .fill,
        alignment: UIStackView.Alignment = .fill
    ) -> UIStackView {
        let stack = UIStackView(arrangedSubviews: views)
        stack.axis = axis
        stack.spacing = spacing
        stack.distribution = distribution
        stack.alignment = alignment
        stack.layoutMargins = .zero
        stack.isLayoutMarginsRelativeArrangement = true
        return stack
    }
    
    /// Add a loading indicator with optional text
    /// - Parameters:
    ///   - text: Optional loading text
    ///   - style: Activity indicator style
    /// - Returns: The container view that can be removed later
    func addLoadingIndicator(text: String? = nil, style: UIActivityIndicatorView.Style = .large) -> UIView {
        // Create container
        let container = UIView()
        container.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        container.translatesAutoresizingMaskIntoConstraints = false
        self.addSubview(container)
        
        NSLayoutConstraint.activate([
            container.leadingAnchor.constraint(equalTo: self.leadingAnchor),
            container.trailingAnchor.constraint(equalTo: self.trailingAnchor),
            container.topAnchor.constraint(equalTo: self.topAnchor),
            container.bottomAnchor.constraint(equalTo: self.bottomAnchor)
        ])
        
        // Create content container with blur effect
        let blurEffect = UIBlurEffect(style: .systemMaterial)
        let contentContainer = UIVisualEffectView(effect: blurEffect)
        contentContainer.layer.cornerRadius = 16
        contentContainer.clipsToBounds = true
        contentContainer.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(contentContainer)
        
        NSLayoutConstraint.activate([
            contentContainer.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            contentContainer.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            contentContainer.widthAnchor.constraint(equalToConstant: 200),
            contentContainer.heightAnchor.constraint(equalToConstant: text != nil ? 200 : 150)
        ])
        
        // Add activity indicator
        let activityIndicator = UIActivityIndicatorView(style: style)
        activityIndicator.startAnimating()
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        contentContainer.contentView.addSubview(activityIndicator)
        
        NSLayoutConstraint.activate([
            activityIndicator.centerXAnchor.constraint(equalTo: contentContainer.contentView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: contentContainer.contentView.centerYAnchor, 
                                                     constant: text != nil ? -20 : 0),
            activityIndicator.widthAnchor.constraint(equalToConstant: 50),
            activityIndicator.heightAnchor.constraint(equalToConstant: 50)
        ])
        
        // Add message label if provided
        if let text = text {
            let label = UILabel()
            label.text = text
            label.textAlignment = .center
            label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
            label.textColor = .label
            label.translatesAutoresizingMaskIntoConstraints = false
            contentContainer.contentView.addSubview(label)
            
            NSLayoutConstraint.activate([
                label.centerXAnchor.constraint(equalTo: contentContainer.contentView.centerXAnchor),
                label.topAnchor.constraint(equalTo: activityIndicator.bottomAnchor, constant: 20),
                label.leadingAnchor.constraint(equalTo: contentContainer.contentView.leadingAnchor, constant: 16),
                label.trailingAnchor.constraint(equalTo: contentContainer.contentView.trailingAnchor, constant: -16)
            ])
        }
        
        return container
    }
    
    /// Add an animated icon as a child view (replacement for Lottie)
    /// - Parameters:
    ///   - systemName: SF Symbol name
    ///   - tintColor: Icon tint color
    ///   - size: Size for the icon
    /// - Returns: The configured image view
    func addAnimatedIcon(
        systemName: String, 
        tintColor: UIColor = .systemBlue,
        size: CGSize = CGSize(width: 100, height: 100)
    ) -> UIImageView {
        let imageView = UIImageView()
        if let image = UIImage(systemName: systemName) {
            imageView.image = image
        } else {
            // Fallback if SF Symbol not available
            imageView.backgroundColor = tintColor.withAlphaComponent(0.2)
            imageView.layer.cornerRadius = min(size.width, size.height) / 2
        }
        
        imageView.tintColor = tintColor
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        self.addSubview(imageView)
        
        NSLayoutConstraint.activate([
            imageView.widthAnchor.constraint(equalToConstant: size.width),
            imageView.heightAnchor.constraint(equalToConstant: size.height)
        ])
        
        // Add animation
        UIView.animate(withDuration: 1.5, delay: 0, options: [.autoreverse, .repeat, .curveEaseInOut], animations: {
            imageView.transform = CGAffineTransform(scaleX: 1.2, y: 1.2)
        }, completion: nil)
        
        return imageView
    }
    
    /// Apply elegant card styling to the view
    /// - Parameters:
    ///   - cornerRadius: Corner radius for the card
    ///   - shadowOpacity: Shadow opacity (0-1)
    ///   - backgroundColor: Background color
    func applyCardStyling(
        cornerRadius: CGFloat = 16,
        shadowOpacity: Float = 0.1,
        backgroundColor: UIColor = .systemBackground
    ) {
        self.backgroundColor = backgroundColor
        layer.cornerRadius = cornerRadius
        layer.masksToBounds = false
        
        // Add shadow
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOffset = CGSize(width: 0, height: 4)
        layer.shadowOpacity = shadowOpacity
        layer.shadowRadius = 6
    }
    
    /// Add a gradient background to the view
    /// - Parameters:
    ///   - colors: Gradient colors
    ///   - startPoint: Start point (default top-left)
    ///   - endPoint: End point (default bottom-right)
    func addGradientBackground(
        colors: [UIColor] = [.systemBlue, UIColor(red: 0, green: 0.5, blue: 1, alpha: 1)],
        startPoint: CGPoint = CGPoint(x: 0, y: 0),
        endPoint: CGPoint = CGPoint(x: 1, y: 1)
    ) {
        // Remove any existing gradient
        layer.sublayers?.filter { $0 is CAGradientLayer }.forEach { $0.removeFromSuperlayer() }
        
        // Create gradient layer
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = colors.map { $0.cgColor }
        gradientLayer.startPoint = startPoint
        gradientLayer.endPoint = endPoint
        gradientLayer.frame = bounds
        gradientLayer.cornerRadius = layer.cornerRadius
        
        // Insert at index 0 to be below other sublayers
        layer.insertSublayer(gradientLayer, at: 0)
        
        // Make sure gradient updates when view is resized
        layoutIfNeeded()
    }
    
    /// Apply futuristic shadow effect to the view
    func applyFuturisticShadow() {
        layer.masksToBounds = false
        layer.shadowColor = UIColor.systemBlue.cgColor
        layer.shadowOffset = CGSize(width: 0, height: 4)
        layer.shadowOpacity = 0.2
        layer.shadowRadius = 8
    }
}

// MARK: - UIButton Extensions
extension UIButton {
    /// Convert a standard UIButton to a gradient button
    /// - Parameters:
    ///   - colors: Gradient colors
    ///   - startPoint: Start point of gradient
    ///   - endPoint: End point of gradient
    func convertToGradientButton(
        colors: [UIColor] = [.systemBlue, UIColor(red: 0, green: 0.5, blue: 1, alpha: 1)],
        startPoint: CGPoint = CGPoint(x: 0, y: 0),
        endPoint: CGPoint = CGPoint(x: 1, y: 1)
    ) {
        // Get the button's title color for styling consideration
        // (Currently not used but could be used for contrast calculations)
        
        // Add gradient background
        addGradientBackground(colors: colors, startPoint: startPoint, endPoint: endPoint)
        
        // Add shadow
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOffset = CGSize(width: 0, height: 3)
        layer.shadowOpacity = 0.2
        layer.shadowRadius = 4
        layer.masksToBounds = false
    }
    
    /// Create a gradient button
    /// - Parameters:
    ///   - title: Button title
    ///   - colors: Gradient colors
    ///   - cornerRadius: Corner radius
    ///   - fontSize: Font size
    /// - Returns: A configured button with gradient
    static func createGradientButton(
        title: String,
        colors: [UIColor] = [.systemBlue, UIColor(red: 0, green: 0.5, blue: 1, alpha: 1)],
        cornerRadius: CGFloat = 12,
        fontSize: CGFloat = 16
    ) -> UIButton {
        let button = UIButton(type: .custom)
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
        button.layer.cornerRadius = cornerRadius
        button.clipsToBounds = true
        
        // Add gradient
        button.addGradientBackground(colors: colors)
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 3)
        button.layer.shadowOpacity = 0.2
        button.layer.shadowRadius = 4
        button.layer.masksToBounds = false
        
        return button
    }
}

// MARK: - UIViewController Extensions
extension UIViewController {
    /// Show a loading overlay
    /// - Parameters:
    ///   - message: Optional loading message
    /// - Returns: The container view that can be removed later
    func showLoadingOverlay(message: String? = "Loading...") -> UIView {
        return view.addLoadingIndicator(text: message)
    }
    
    /// Hide the loading overlay
    /// - Parameter overlay: The overlay container view returned by showLoadingOverlay
    func hideLoadingOverlay(_ overlay: UIView) {
        UIView.animate(withDuration: 0.3, animations: {
            overlay.alpha = 0
        }, completion: { _ in
            overlay.removeFromSuperview()
        })
    }
    
    /// Show a brief success animation
    /// - Parameter message: Optional success message
    func showSuccessAnimation(message: String? = nil) {
        // Create success icon
        let imageView = view.addAnimatedIcon(
            systemName: "checkmark.circle.fill",
            tintColor: .systemGreen,
            size: CGSize(width: 100, height: 100)
        )
        
        // Center the animation
        NSLayoutConstraint.activate([
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
        
        // Add message label if provided
        if let message = message {
            let label = UILabel()
            label.text = message
            label.textAlignment = .center
            label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
            label.textColor = .label
            label.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(label)
            
            NSLayoutConstraint.activate([
                label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                label.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 8)
            ])
            
            // Remove label after animation
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                label.removeFromSuperview()
            }
        }
        
        // Remove animation after playing
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            imageView.layer.removeAllAnimations()
            imageView.removeFromSuperview()
        }
    }
}

// MARK: - CALayer Extensions
extension CALayer {
    /// Apply a shadow with blue tint to the layer
    func applyBlueTintedShadow() {
        masksToBounds = false
        shadowColor = UIColor.systemBlue.cgColor
        shadowOffset = CGSize(width: 0, height: 4)
        shadowOpacity = 0.2
        shadowRadius = 8
    }
}

// MARK: - Animation Helper (replacing the Lottie-based implementation)
class AnimationHelper {
    /// Show a loading animation overlay
    /// - Parameters:
    ///   - view: View to add the loader to
    ///   - message: Optional message to display
    /// - Returns: Container view that can be removed later
    static func showLoader(in view: UIView, message: String? = nil) -> UIView {
        return view.addLoadingIndicator(text: message)
    }
    
    /// Hide the loader animation
    /// - Parameter container: Container view returned by showLoader
    static func hideLoader(_ container: UIView) {
        UIView.animate(withDuration: 0.3, animations: {
            container.alpha = 0
        }, completion: { _ in
            container.removeFromSuperview()
        })
    }
}

// MARK: - ElegantUIComponents (Replacement without SnapKit & Lottie)
class ElegantUIComponents {
    /// Create a beautifully styled button with gradient
    /// - Parameters:
    ///   - title: Button title
    ///   - colors: Gradient colors (default blue gradient)
    ///   - cornerRadius: Corner radius (default 12)
    ///   - fontSize: Font size (default 16)
    /// - Returns: Configured button
    static func createGradientButton(
        title: String,
        colors: [UIColor] = [.systemBlue, UIColor(red: 0, green: 0.5, blue: 1, alpha: 1)],
        cornerRadius: CGFloat = 12,
        fontSize: CGFloat = 16
    ) -> UIButton {
        return UIButton.createGradientButton(title: title, colors: colors, cornerRadius: cornerRadius, fontSize: fontSize)
    }
    
    /// Create a card view with shadow
    /// - Parameters:
    ///   - backgroundColor: Card background color
    ///   - cornerRadius: Corner radius
    /// - Returns: Configured card view
    static func createCardView(
        backgroundColor: UIColor = .systemBackground,
        cornerRadius: CGFloat = 16
    ) -> UIView {
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
    static func createFloatingTextField(
        placeholder: String,
        backgroundColor: UIColor = .systemBackground,
        borderColor: UIColor = .systemGray4
    ) -> UIView {
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
            textField.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            textField.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -16),
            textField.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            textField.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12)
        ])
        
        return container
    }
}
