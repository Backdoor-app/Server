// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// A popup that displays for exactly 5 seconds on first app launch
class StartupPopupViewController: UIViewController {
    // MARK: - UI Components

    private let contentView = UIView()
    private let titleLabel = UILabel()
    private let messageLabel = UILabel()
    private let iconImageView = UIImageView()
    private let progressView = UIProgressView()
    private let secondsRemainingLabel = UILabel()

    // MARK: - Properties

    private let displayDuration: TimeInterval = 5.0
    private var timer: Timer?
    private var startTime: Date?
    private var dismissInProgress = false

    // Callback when popup is dismissed
    var onDismiss: (() -> Void)?

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        // Make sure the view is properly accessible
        view.isUserInteractionEnabled = true
        view.accessibilityIdentifier = "StartupPopupView"
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

        // Start the timer when the view appears
        startTime = Date()
        startProgressTimer()

        // Schedule automatic dismissal after exactly 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + displayDuration) { [weak self] in
            guard let self = self, !self.dismissInProgress else { return }
            self.dismissPopup()
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Always invalidate timer to prevent leaks
        timer?.invalidate()
        timer = nil
    }

    deinit {
        Debug.shared.log(message: "StartupPopupViewController deinit", type: .debug)
        // Ensure timer is invalidated to prevent leaks
        timer?.invalidate()
    }

    // MARK: - UI Setup

    private func setupUI() {
        // Main view configuration - semi-transparent black background
        view.backgroundColor = UIColor.black.withAlphaComponent(0.7)

        // Content view setup - white card with shadow
        contentView.backgroundColor = .systemBackground
        contentView.layer.cornerRadius = 16
        contentView.layer.shadowColor = UIColor.black.cgColor
        contentView.layer.shadowOpacity = 0.3
        contentView.layer.shadowOffset = CGSize(width: 0, height: 4)
        contentView.layer.shadowRadius = 6
        contentView.accessibilityIdentifier = "StartupPopupContentView"
        view.addSubview(contentView)

        // Icon setup
        iconImageView.image = UIImage(named: "backdoor_glyph")
        iconImageView.contentMode = .scaleAspectFit
        iconImageView.tintColor = Preferences.appTintColor.uiColor
        contentView.addSubview(iconImageView)

        // Title setup
        titleLabel.text = "Welcome to Backdoor"
        titleLabel.font = .systemFont(ofSize: 22, weight: .bold)
        titleLabel.textAlignment = .center
        titleLabel.textColor = .label
        titleLabel.accessibilityTraits = .header
        contentView.addSubview(titleLabel)

        // Message setup
        messageLabel.text = "Logging Into Backdoor Please Wait"
        messageLabel.font = .systemFont(ofSize: 16)
        messageLabel.textAlignment = .center
        messageLabel.textColor = .secondaryLabel
        messageLabel.numberOfLines = 0
        contentView.addSubview(messageLabel)

        // Progress view setup
        progressView.progressTintColor = Preferences.appTintColor.uiColor
        progressView.trackTintColor = .systemGray4
        progressView.progress = 0.0
        progressView.accessibilityLabel = "Time remaining"
        contentView.addSubview(progressView)

        // Seconds remaining label
        secondsRemainingLabel.text = "Entering in 5 seconds..."
        secondsRemainingLabel.font = .systemFont(ofSize: 14, weight: .medium)
        secondsRemainingLabel.textAlignment = .center
        secondsRemainingLabel.textColor = .secondaryLabel
        contentView.addSubview(secondsRemainingLabel)

        // Setup auto layout
        setupConstraints()
    }

    private func setupConstraints() {
        contentView.translatesAutoresizingMaskIntoConstraints = false
        iconImageView.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        messageLabel.translatesAutoresizingMaskIntoConstraints = false
        progressView.translatesAutoresizingMaskIntoConstraints = false
        secondsRemainingLabel.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            // Content view constraints - centered with fixed width
            contentView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            contentView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            contentView.widthAnchor.constraint(equalToConstant: 280),

            // Icon constraints
            iconImageView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 24),
            iconImageView.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            iconImageView.widthAnchor.constraint(equalToConstant: 70),
            iconImageView.heightAnchor.constraint(equalToConstant: 70),

            // Title constraints
            titleLabel.topAnchor.constraint(equalTo: iconImageView.bottomAnchor, constant: 16),
            titleLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            titleLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Message constraints
            messageLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 8),
            messageLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            messageLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Seconds remaining label
            secondsRemainingLabel.topAnchor.constraint(equalTo: messageLabel.bottomAnchor, constant: 20),
            secondsRemainingLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            secondsRemainingLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Progress view constraints
            progressView.topAnchor.constraint(equalTo: secondsRemainingLabel.bottomAnchor, constant: 12),
            progressView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            progressView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            progressView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -24),
        ])
    }

    // MARK: - Timer Management

    private func startProgressTimer() {
        // Update progress every 0.1 seconds
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let startTime = self.startTime else { return }

            let elapsedTime = Date().timeIntervalSince(startTime)
            let timeRemaining = max(0, self.displayDuration - elapsedTime)
            let progress = Float(min(elapsedTime / self.displayDuration, 1.0))

            // Update UI on main thread
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.progressView.progress = progress

                // Update seconds remaining label
                let secondsRemaining = Int(ceil(timeRemaining))
                if secondsRemaining > 0 {
                    self.secondsRemainingLabel.text = "Entering in \(secondsRemaining) second\(secondsRemaining == 1 ? "" : "s")..."
                } else {
                    self.secondsRemainingLabel.text = "Loading app..."
                }

                // Update accessibility for VoiceOver users
                self.progressView.accessibilityValue = "\(secondsRemaining) seconds remaining"
            }

            // Stop the timer if progress is complete
            if progress >= 1.0 {
                self.timer?.invalidate()
                self.timer = nil
            }
        }

        // Make sure timer runs even when scrolling or during other UI operations
        RunLoop.current.add(timer!, forMode: .common)
    }

    private func dismissPopup() {
        // Check if dismissal is already in progress to prevent multiple calls
        guard !dismissInProgress else { return }
        dismissInProgress = true

        // Stop the timer
        timer?.invalidate()
        timer = nil

        // Animate fade out
        UIView.animate(withDuration: 0.3, animations: { [weak self] in
            self?.view.alpha = 0
        }) { [weak self] completed in
            guard let self = self, completed else { return }

            // Dismiss the view controller
            self.dismiss(animated: false) { [weak self] in
                // Call the completion handler
                self?.onDismiss?()
            }
        }
    }

    // MARK: - Actions

    // Add a method to manually dismiss if needed
    @objc func handleDismissTap() {
        dismissPopup()
    }

    // MARK: - Overrides

    override func touchesBegan(_ touches: Set<UITouch>, with _: UIEvent?) {
        // Allow touches only on certain areas if needed
        if let touch = touches.first {
            let location = touch.location(in: view)

            // Check if touch is outside the content view
            if !contentView.frame.contains(location) {
                // Optionally dismiss on background tap
                // dismissPopup()
                return
            }
        }

        // Don't call super to prevent the touch from being processed by parent views
    }
}
