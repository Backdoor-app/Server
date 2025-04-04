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

#ifndef LSApplicationWorkspace_h
#define LSApplicationWorkspace_h

#import <Foundation/Foundation.h>

@interface LSApplicationWorkspace : NSObject
+ (instancetype)defaultWorkspace;
- (bool)openApplicationWithBundleID:(NSString *)bundleID;
@end

#endif /* LSApplicationWorkspace_h */
