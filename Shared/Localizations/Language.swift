// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation

struct Language {
    static var availableLanguages: [Language] {
        return Bundle.main.localizations.compactMap { languageCode in
            // Skip over 'Base', it means nothing
            guard languageCode != "Base",
                  let subtitle = Locale.current.localizedString(forLanguageCode: languageCode)
            else {
                return nil
            }

            let displayLocale = Locale(identifier: languageCode)
            guard let displayName = displayLocale.localizedString(forLanguageCode: languageCode)?.capitalized(with: displayLocale) else {
                return nil
            }

            return Language(displayName: displayName, subtitleText: subtitle, languageCode: languageCode)
        }
    }

    /// The display name, being the language's name in itself, such as 'русский' in Russian
    let displayName: String

    /// The subtitle, being the language's name in the current language,
    /// such as 'Russian' when the user is currently using English.
    let subtitleText: String

    /// The language code, such as 'ru'
    let languageCode: String
}
