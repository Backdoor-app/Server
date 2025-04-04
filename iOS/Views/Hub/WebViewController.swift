// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import UIKit
import WebKit

class WebViewController: UIViewController, WKNavigationDelegate {
    private let webView: WKWebView = {
        let webView = WKWebView()
        webView.translatesAutoresizingMaskIntoConstraints = false
        return webView
    }()

    private let backButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Back", for: .normal)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    private let forwardButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Forward", for: .normal)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadWebsite()
        webView.navigationDelegate = self // Set delegate to handle navigation events
        backButton.addTarget(self, action: #selector(goBack), for: .touchUpInside)
        forwardButton.addTarget(self, action: #selector(goForward), for: .touchUpInside)
    }

    private func setupUI() {
        view.backgroundColor = .white
        view.addSubview(webView)
        view.addSubview(backButton)
        view.addSubview(forwardButton)

        NSLayoutConstraint.activate([
            webView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            webView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            webView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: view.trailingAnchor),

            backButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
            backButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),

            forwardButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
            forwardButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
        ])

        updateButtonStates()
    }

    private func loadWebsite() {
        if let url = URL(string: "https://backdoor-bdg.store") {
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }

    @objc private func goBack() {
        if webView.canGoBack {
            webView.goBack()
        }
    }

    @objc private func goForward() {
        if webView.canGoForward {
            webView.goForward()
        }
    }

    // Delegate method to update button states after navigation
    func webView(_: WKWebView, didFinish _: WKNavigation!) {
        updateButtonStates()
    }

    private func updateButtonStates() {
        backButton.isEnabled = webView.canGoBack
        forwardButton.isEnabled = webView.canGoForward
    }
}
