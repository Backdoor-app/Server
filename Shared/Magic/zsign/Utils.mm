/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

/*
 */

#include "Utils.hpp"
#import <Foundation/Foundation.h>

extern "C" {

const char *getDocumentsDirectory() {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    const char *documentsPath = [documentsDirectory UTF8String];
    return documentsPath;
}
}
