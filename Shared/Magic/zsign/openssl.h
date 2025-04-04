/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted
 * under the terms of the Proprietary Software License.
 */

#pragma once
#include "common/json.h"

bool GetCertSubjectCN(const string &strCertData, string &strSubjectCN);
bool GetCMSInfo(uint8_t *pCMSData, uint32_t uCMSLength, JValue &jvOutput);
bool GetCMSContent(const string &strCMSDataInput, string &strContentOutput);
bool GenerateCMS(const string &strSignerCertData, const string &strSignerPKeyData, const string &strCDHashData,
                 const string &strCDHashPlist, string &strCMSOutput);

class ZSignAsset {
public:
    ZSignAsset();

public:
    bool GenerateCMS(const string &strCDHashData, const string &strCDHashesPlist,
                     const string &strCodeDirectorySlotSHA1, const string &strAltnateCodeDirectorySlot256,
                     string &strCMSOutput);
    bool Init(const string &strSignerCertFile, const string &strSignerPKeyFile, const string &strProvisionFile,
              const string &strEntitlementsFile, const string &strPassword);

public:
    string m_strTeamId;
    string m_strSubjectCN;
    string m_strProvisionData;
    string m_strEntitlementsData;

private:
    void *m_evpPKey;
    void *m_x509Cert;
};
