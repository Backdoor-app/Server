// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// Floating button with custom AI assistant functionality
final class FloatingAIButton: UIView {
    // MARK: - UI Components

    private let button: UIButton = {
        let btn = UIButton(type: .custom)
        btn.layer.cornerRadius = 30
        btn.setImage(UIImage(systemName: "bubble.left.and.bubble.right.fill", withConfiguration: UIImage.SymbolConfiguration(pointSize: 24, weight: .medium)), for: .normal)
        btn.tintColor = .white
        btn.accessibilityLabel = "AI Assistant"

        return btn
    }()

    // MARK: - Properties

    private var initialPoint: CGPoint = .zero
    private var lastStoredPosition: CGPoint?
    private let userDefaultsKey = "AIButtonPosition"

    // MARK: - Initialization

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
        setupGradientBackground()
        loadSavedPosition()
        startPulseAnimation()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Setup

    private func setupView() {
        // Add and configure the button
        addSubview(button)
        button.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            button.topAnchor.constraint(equalTo: topAnchor),
            button.leadingAnchor.constraint(equalTo: leadingAnchor),
            button.trailingAnchor.constraint(equalTo: trailingAnchor),
            button.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])

        // Set frame size
        frame = CGRect(x: 0, y: 0, width: 60, height: 60)

        // Add shadow
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.4
        layer.shadowOffset = CGSize(width: 0, height: 4)
        layer.shadowRadius = 6

        // Add gestures
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan))
        addGestureRecognizer(panGesture)

        // Add tap target
        button.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
    }

    private func setupGradientBackground() {
        // Create a gradient background that matches app theme
        let gradient = CAGradientLayer()
        gradient.frame = CGRect(x: 0, y: 0, width: 60, height: 60)

        // Use app tint color for gradient
        let tintColor = Preferences.appTintColor.uiColor
        let lighterTint = tintColor.adjustBrightness(by: 0.2)

        gradient.colors = [tintColor.cgColor, lighterTint.cgColor]
        gradient.startPoint = CGPoint(x: 0, y: 0)
        gradient.endPoint = CGPoint(x: 1, y: 1)
        gradient.cornerRadius = 30
        button.layer.insertSublayer(gradient, at: 0)
    }

    private func startPulseAnimation() {
        // Create a subtle pulse animation
        let pulse = CASpringAnimation(keyPath: "transform.scale")
        pulse.duration = 0.8
        pulse.fromValue = 1.0
        pulse.toValue = 1.08
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.initialVelocity = 0.5
        pulse.damping = 1.0
        layer.add(pulse, forKey: "pulse")
    }

    // MARK: - Position Management

    private func loadSavedPosition() {
        if let positionData = UserDefaults.standard.data(forKey: userDefaultsKey),
           let position = try? JSONDecoder().decode(CGPoint.self, from: positionData)
        {
            lastStoredPosition = position

            // Apply the saved position if valid
            if let superview = superview, position.x > 0 && position.y > 0 {
                center = position

                // Make sure it's within bounds in case of screen size changes
                let safeArea = superview.safeAreaInsets
                let minX = 30 + safeArea.left
                let maxX = superview.bounds.width - 30 - safeArea.right
                let minY = 30 + safeArea.top
                let maxY = superview.bounds.height - 30 - safeArea.bottom

                center.x = min(max(center.x, minX), maxX)
                center.y = min(max(center.y, minY), maxY)

                Debug.shared.log(message: "Restored button position to: \(center)", type: .debug)
            }
        }
    }

    private func savePosition() {
        if let positionData = try? JSONEncoder().encode(center) {
            UserDefaults.standard.set(positionData, forKey: userDefaultsKey)
            UserDefaults.standard.synchronize() // Ensure the position is saved immediately
            lastStoredPosition = center
            Debug.shared.log(message: "Saved button position: \(center)", type: .debug)
        }
    }

    // Make this method public so FloatingButtonManager can access it
    func getSavedPosition() -> CGPoint? {
        if let positionData = UserDefaults.standard.data(forKey: userDefaultsKey),
           let position = try? JSONDecoder().decode(CGPoint.self, from: positionData)
        {
            return position
        }
        return lastStoredPosition
    }

    // Apply saved position whenever the button is added to a view
    override func didMoveToSuperview() {
        super.didMoveToSuperview()

        if superview != nil {
            DispatchQueue.main.async { [weak self] in
                self?.loadSavedPosition()
            }
        }
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let superview = superview else { return }
        let translation = gesture.translation(in: superview)

        switch gesture.state {
            case .began:
                initialPoint = center
                // Stop pulse animation during drag
                layer.removeAnimation(forKey: "pulse")

            case .changed:
                center = CGPoint(x: initialPoint.x + translation.x,
                                 y: initialPoint.y + translation.y)
                keepWithinBounds(superview: superview)

            case .ended, .cancelled:
                snapToEdge(superview: superview)
                // Save position after snap
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                    self?.savePosition()
                }
                // Restart pulse animation
                startPulseAnimation()

            default:
                break
        }
    }

    private func keepWithinBounds(superview: UIView) {
        let margin: CGFloat = 20
        center.x = max(margin + frame.width / 2, min(center.x, superview.bounds.width - margin - frame.width / 2))
        center.y = max(margin + frame.height / 2, min(center.y, superview.bounds.height - margin - frame.height / 2))
    }

    private func snapToEdge(superview: UIView) {
        let margin: CGFloat = 20
        let newX = center.x < superview.bounds.width / 2 ? margin + frame.width / 2 : superview.bounds.width - margin - frame.width / 2

        // Animate to edge with spring effect
        UIView.animate(withDuration: 0.3, delay: 0, usingSpringWithDamping: 0.7, initialSpringVelocity: 0.5) {
            self.center = CGPoint(x: newX, y: self.center.y)
            self.keepWithinBounds(superview: superview)
        }
    }

    // MARK: - Actions

    @objc private func buttonTapped() {
        // Add haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        // Notify that the AI assistant button was tapped
        NotificationCenter.default.post(name: .showAIAssistant, object: nil)
    }

    // MARK: - Public Methods

    /// Update the visual appearance of the button when app theme changes
    func updateAppearance() {
        // Remove existing gradient
        button.layer.sublayers?.first { $0 is CAGradientLayer }?.removeFromSuperlayer()

        // Re-apply gradient with new app theme color
        setupGradientBackground()
    }
}

// MARK: - Helper Extensions

extension UIColor {
    /// Adjust color brightness
    func adjustBrightness(by factor: CGFloat) -> UIColor {
        var hue: CGFloat = 0, saturation: CGFloat = 0, brightness: CGFloat = 0, alpha: CGFloat = 0

        if self.getHue(&hue, saturation: &saturation, brightness: &brightness, alpha: &alpha) {
            return UIColor(hue: hue,
                           saturation: saturation,
                           brightness: min(brightness + factor, 1.0),
                           alpha: alpha)
        }
        return self
    }
}

// Note: showAIAssistant notification name is declared elsewhere
// This section is intentionally empty to avoid duplicate declarations

// MARK: - Codable Extension for CGPoint

extension CGPoint: Codable {
    enum CodingKeys: String, CodingKey {
        case x, y
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(CGFloat.self, forKey: .x)
        let y = try container.decode(CGFloat.self, forKey: .y)
        self.init(x: x, y: y)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
    }
}
