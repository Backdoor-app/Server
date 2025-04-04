// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

import Foundation
import NaturalLanguage

/// Helper class for natural language processing features
class NaturalLanguageHelper {
    
    // Singleton instance
    static let shared = NaturalLanguageHelper()
    
    private init() {}
    
    /// Detect the language of a given text
    func detectLanguage(in text: String) -> String {
        // Use Apple's NaturalLanguage framework
        let tagger = NLTagger(tagSchemes: [.language])
        tagger.string = text
        if let language = tagger.dominantLanguage?.rawValue {
            return language
        }
        
        return "unknown"
    }
    
    /// Get sentiment analysis for text
    /// Returns score from -1.0 (negative) to 1.0 (positive)
    func analyzeSentiment(in text: String) -> Double {
        guard !text.isEmpty else { return 0.0 }
        
        // Use NLTagger for sentiment analysis where available
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text
        
        let (sentiment, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        
        // Convert the sentiment string to a Double, defaulting to 0 if parsing fails
        return Double(sentiment?.rawValue ?? "0") ?? 0.0
    }
    
    /// Extract entities from text using Apple's NL framework
    func extractEntities(from text: String) -> [String: String] {
        var entities: [String: String] = [:]
        
        // Use NLTagger for named entity recognition
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        
        // Process entire text for entities
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType) { tag, range in
            if let tag = tag {
                let entity = String(text[range])
                entities[entity] = tag.rawValue
            }
            return true
        }
        
        return entities
    }
    
    /// Tokenize text into words using Apple's tokenizer
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            tokens.append(String(text[range]))
            return true
        }
        
        return tokens
    }
}
