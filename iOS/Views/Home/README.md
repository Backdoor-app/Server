# Home Directory Organization

This directory has been organized into a logical structure to improve maintainability and code organization:

## Directory Structure

### Core
Contains the main view controllers:
- HomeViewController.swift - The primary file browser view controller
- DirectoryViewController.swift - Extension of HomeViewController for browsing directories

### Editors
Contains all editor view controllers:
- BaseEditorViewController.swift - Base class for editors
- HexEditorViewController.swift - For editing binary files
- IPAEditorViewController.swift - For editing iOS app packages
- PlistEditorViewController.swift - For editing property list files
- TextEditorViewController.swift - For editing text files

### Extensions
Contains extensions to view controllers and classes:
- HomeViewControllerExtensions.swift - Extensions for HomeViewController
- HomeViewExtras.swift - Additional extension methods

### Operations
Contains file operation classes and handlers:
- FileOperations.swift - Core file operations (copy, move, delete)
- FileDragAndDrop.swift - Support for drag and drop functionality
- FileContextMenu.swift - Context menu support
- HomeViewFileHandlers.swift - File handling utilities
- HomeViewTableHandlers.swift - Table view operations

### UI
Contains UI components:
- HomeViewUI.swift - UI elements and styles
- FileTableViewCell.swift - Custom table view cell for files

### Utilities
Contains helper and utility classes:
- HomeViewUtilities.swift - Various utility functions
- FilePreviewController.swift - File preview functionality
- FilePreviewManager.swift - Manager for file previews

## Notes
- The HomeViewController.swift file contains all the core functionality, with enhancements merged from previous fixes
- Redundant files have been removed or consolidated
