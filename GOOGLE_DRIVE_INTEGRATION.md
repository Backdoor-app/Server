# Google Drive Integration for Backdoor AI Server

This document provides instructions on how to set up and use the Google Drive integration for database and model storage in the Backdoor AI Server.

## Overview

The Backdoor AI Server can now store its database and model files on Google Drive, providing:
- Cloud backup of all data
- Accessibility from multiple servers
- Persistent storage across deployments
- Automatic synchronization

## Prerequisites

1. Google Cloud Platform account
2. A Google Cloud Project with the Google Drive API enabled
3. Service account credentials with appropriate permissions

## Setup Instructions

### 1. Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" and provide a project name
3. Wait for the project to be created and select it

### 2. Enable the Google Drive API

1. In your project, navigate to "APIs & Services" > "Library"
2. Search for "Google Drive API"
3. Click on "Google Drive API" and press "Enable"

### 3. Create Service Account Credentials

1. Navigate to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Provide a name for your service account and click "Create"
4. For the role, select "Project" > "Editor" (or a more restrictive role if preferred)
5. Click "Continue" and then "Done"
6. Click on the created service account
7. Go to the "Keys" tab and click "Add Key" > "Create new key"
8. Select "JSON" format and click "Create"
9. Save the JSON file that is downloaded - this is your service account credentials file

### 4. Share a Google Drive Folder (Optional)

If you want to use a specific Google Drive folder:

1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder or select an existing one
3. Right-click the folder and select "Share"
4. Add the email address of your service account (found in the JSON file) with "Editor" access
5. Click "Share"

### 5. Configure Backdoor AI Server

1. Place your credentials JSON file in a secure location, ideally outside the repository
2. Set these environment variables:

```bash
export GOOGLE_DRIVE_ENABLED=true
export GOOGLE_CREDENTIALS_PATH=/path/to/your/google_credentials.json
export GOOGLE_DB_FILENAME=backdoor_ai_db.db
export GOOGLE_MODELS_FOLDER=backdoor_models
```

Or edit the `config.py` file to set the appropriate values:

```python
# Google Drive Integration Settings
GOOGLE_DRIVE_ENABLED = True
GOOGLE_CREDENTIALS_PATH = '/path/to/your/google_credentials.json'
GOOGLE_DB_FILENAME = 'backdoor_ai_db.db'
GOOGLE_MODELS_FOLDER = 'backdoor_models'
```

## How It Works

When the server starts with Google Drive integration enabled:

1. The system authenticates with Google Drive using the service account credentials
2. The database file is downloaded from Drive if it exists, otherwise a new one is created
3. Models folder is found or created on Google Drive
4. All database operations are performed on a local copy and then synchronized to Drive
5. Model files are automatically uploaded to and downloaded from Google Drive

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `GOOGLE_DRIVE_ENABLED` | Enable/disable Google Drive integration | `False` |
| `GOOGLE_CREDENTIALS_PATH` | Path to service account credentials JSON | `google_credentials.json` |
| `GOOGLE_DB_FILENAME` | Name of the database file in Google Drive | `backdoor_ai_db.db` |
| `GOOGLE_MODELS_FOLDER` | Name of the folder for models in Google Drive | `backdoor_models` |
| `GOOGLE_DB_SYNC_INTERVAL` | How often to sync the database (seconds) | `60` |
| `GOOGLE_MODELS_SYNC_INTERVAL` | How often to sync model metadata (seconds) | `300` |

## Synchronization Details

- **Database**: Synced every 60 seconds (configurable) and after each transaction
- **Models**: Downloaded on-demand when requested by the application
- **Uploaded Models**: Immediately uploaded to Google Drive when received

## Security Considerations

1. Keep your credentials JSON file secure and outside version control
2. Use appropriate Google Drive permissions for the service account
3. Consider encrypting sensitive data before storing it in Google Drive
4. Regularly rotate service account keys (generate new ones and delete old ones)

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure the credentials file is valid and accessible
   - Check that the Google Drive API is enabled in your project
   - Verify the service account has appropriate permissions

2. **Sync Failures**
   - Check internet connectivity
   - Ensure you have sufficient Google Drive storage space
   - Verify file permissions in Google Drive

3. **Missing Files**
   - Check if the files exist in Google Drive
   - Ensure file naming conventions match expectations
   - Look for error messages in the server logs

### Logs

Monitor the application logs for messages related to Google Drive operations:
- Successful operations are logged at INFO level
- Errors are logged at ERROR level
- Fallbacks to local storage are logged at WARNING level

### Disabling Google Drive Integration

If you encounter persistent issues, you can disable Google Drive integration:

1. Set `GOOGLE_DRIVE_ENABLED` to `False` in configuration
2. Restart the server

The system will automatically fall back to local storage.

## Limitations

1. Concurrent access from multiple server instances may lead to conflicts
2. Large database files may cause performance issues during synchronization
3. Network connectivity issues can impact system reliability
4. Google Drive API rate limits may apply to frequent operations

## Best Practices

1. Use a dedicated Google Cloud project for the integration
2. Monitor storage usage and performance
3. Implement regular backups of key data
4. Use a staging environment to test changes before production
5. Keep dependencies updated for security

By following these guidelines, you can effectively use Google Drive as a database and model storage solution for your Backdoor AI server.
