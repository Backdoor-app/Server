// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit

/// View for collecting feedback on AI responses
class AIFeedbackView: UIView {
    
    // MARK: - Properties
    
    // Callback for when feedback is submitted
    var onFeedbackSubmitted: ((Int, String?) -> Void)?
    
    // UI Components
    private let titleLabel = UILabel()
    private let ratingControl = RatingControl()
    private let commentTextView = UITextView()
    private let submitButton = UIButton(type: .system)
    private let skipButton = UIButton(type: .system)
    private let stackView = UIStackView()
    
    // MARK: - Initialization
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }
    
    // MARK: - Setup
    
    private func setupView() {
        backgroundColor = .systemBackground
        layer.cornerRadius = 12
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.1
        layer.shadowOffset = CGSize(width: 0, height: 2)
        layer.shadowRadius = 4
        
        // Configure title label
        titleLabel.text = "How helpful was this response?"
        titleLabel.font = UIFont.systemFont(ofSize: 16, weight: .semibold)
        titleLabel.textAlignment = .center
        
        // Configure rating control
        ratingControl.starCount = 5
        ratingControl.rating = 0
        
        // Configure comment text view
        commentTextView.text = "Additional comments (optional)"
        commentTextView.textColor = .placeholderText
        commentTextView.font = UIFont.systemFont(ofSize: 14)
        commentTextView.layer.borderColor = UIColor.systemGray4.cgColor
        commentTextView.layer.borderWidth = 1
        commentTextView.layer.cornerRadius = 6
        commentTextView.delegate = self
        commentTextView.isScrollEnabled = false
        
        // Configure submit button
        submitButton.setTitle("Submit Feedback", for: .normal)
        submitButton.backgroundColor = .systemBlue
        submitButton.setTitleColor(.white, for: .normal)
        submitButton.layer.cornerRadius = 8
        submitButton.addTarget(self, action: #selector(submitFeedback), for: .touchUpInside)
        
        // Configure skip button
        skipButton.setTitle("Skip", for: .normal)
        skipButton.setTitleColor(.systemGray, for: .normal)
        skipButton.addTarget(self, action: #selector(skipFeedback), for: .touchUpInside)
        
        // Set up stack view
        stackView.axis = .vertical
        stackView.spacing = 12
        stackView.alignment = .fill
        stackView.distribution = .fill
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        // Add components to stack view
        stackView.addArrangedSubview(titleLabel)
        stackView.addArrangedSubview(ratingControl)
        stackView.addArrangedSubview(commentTextView)
        stackView.addArrangedSubview(submitButton)
        stackView.addArrangedSubview(skipButton)
        
        // Add stack view to main view
        addSubview(stackView)
        
        // Set constraints
        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: topAnchor, constant: 16),
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            stackView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -16),
            
            commentTextView.heightAnchor.constraint(greaterThanOrEqualToConstant: 80),
            
            submitButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    // MARK: - Actions
    
    @objc private func submitFeedback() {
        // Get the feedback values
        let rating = ratingControl.rating
        
        // Only consider comments if they're not the placeholder
        let comment: String? = commentTextView.textColor == .placeholderText ? nil : commentTextView.text
        
        // Call the feedback handler
        onFeedbackSubmitted?(rating, comment)
        
        // Reset the control
        resetFeedbackControl()
        
        // Hide the view with animation
        UIView.animate(withDuration: 0.3, animations: {
            self.alpha = 0
        }) { _ in
            self.removeFromSuperview()
        }
    }
    
    @objc private func skipFeedback() {
        // Reset and hide without submitting
        resetFeedbackControl()
        
        // Hide with animation
        UIView.animate(withDuration: 0.3, animations: {
            self.alpha = 0
        }) { _ in
            self.removeFromSuperview()
        }
    }
    
    /// Reset feedback controls to initial state
    private func resetFeedbackControl() {
        ratingControl.rating = 0
        commentTextView.text = "Additional comments (optional)"
        commentTextView.textColor = .placeholderText
    }
}

// MARK: - UITextView Delegate

extension AIFeedbackView: UITextViewDelegate {
    func textViewDidBeginEditing(_ textView: UITextView) {
        if textView.textColor == .placeholderText {
            textView.text = ""
            textView.textColor = .label
        }
    }
    
    func textViewDidEndEditing(_ textView: UITextView) {
        if textView.text.isEmpty {
            textView.text = "Additional comments (optional)"
            textView.textColor = .placeholderText
        }
    }
}

// MARK: - Rating Control

class RatingControl: UIView {
    
    // MARK: - Properties
    
    /// Number of stars to display
    var starCount: Int = 5 {
        didSet {
            setupStars()
        }
    }
    
    /// Current rating value (0 to starCount)
    var rating: Int = 0 {
        didSet {
            updateStarDisplay()
        }
    }
    
    /// Size of each star
    var starSize: CGFloat = 32
    
    /// Array of star buttons
    private var starButtons: [UIButton] = []
    
    // MARK: - Initialization
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupStars()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupStars()
    }
    
    // MARK: - Setup
    
    private func setupStars() {
        // Remove existing star buttons
        for button in starButtons {
            button.removeFromSuperview()
        }
        starButtons.removeAll()
        
        // Create the new star buttons
        for i in 0..<starCount {
            let button = UIButton()
            
            // Set the button images
            button.setImage(UIImage(systemName: "star"), for: .normal)
            button.setImage(UIImage(systemName: "star.fill"), for: .selected)
            button.setImage(UIImage(systemName: "star.fill"), for: [.highlighted, .selected])
            
            // Add constraints to set the button's width and height
            button.translatesAutoresizingMaskIntoConstraints = false
            button.widthAnchor.constraint(equalToConstant: starSize).isActive = true
            button.heightAnchor.constraint(equalToConstant: starSize).isActive = true
            
            // Set accessibility label
            button.accessibilityLabel = "Set \(i + 1) star rating"
            
            // Add the button action
            button.tag = i
            button.addTarget(self, action: #selector(starButtonTapped(_:)), for: .touchUpInside)
            
            // Add the button to the stack
            addSubview(button)
            starButtons.append(button)
        }
        
        // Update layout
        updateLayout()
        
        // Update the button state
        updateStarDisplay()
    }
    
    private func updateLayout() {
        // Position the buttons horizontally with equal spacing
        let totalStarWidth = CGFloat(starCount) * starSize
        let spacing = (bounds.width - totalStarWidth) / CGFloat(starCount + 1)
        
        for (index, button) in starButtons.enumerated() {
            let xPosition = spacing + CGFloat(index) * (starSize + spacing)
            button.frame = CGRect(x: xPosition, y: 0, width: starSize, height: starSize)
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        updateLayout()
    }
    
    // MARK: - Button Action
    
    @objc private func starButtonTapped(_ sender: UIButton) {
        guard let index = starButtons.firstIndex(of: sender) else {
            return
        }
        
        // Update the rating
        rating = index + 1
    }
    
    // MARK: - Update UI
    
    private func updateStarDisplay() {
        for (index, button) in starButtons.enumerated() {
            // Set the button's selected state based on the rating
            button.isSelected = index < rating
            
            // Update tint color for better visual feedback
            button.tintColor = button.isSelected ? .systemYellow : .systemGray
        }
    }
}
