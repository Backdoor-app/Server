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

#pragma once
#include "archo.h"

class ZMachO {
public:
    ZMachO();
    ~ZMachO();

public:
    bool Init(const char *szFile);
    bool InitV(const char *szFormatPath, ...);
    bool Free();
    void PrintInfo();
    bool Sign(ZSignAsset *pSignAsset, bool bForce, string strBundleId, string strInfoPlistSHA1,
              string strInfoPlistSHA256, const string &strCodeResourcesData);
    bool InjectDyLib(bool bWeakInject, const char *szDyLibPath, bool &bCreate);
    bool ChangeDylibPath(const char *oldPath, const char *newPath);
    std::vector<std::string> ListDylibs();
    bool RemoveDylib(const std::set<std::string> &dylibNames);

private:
    bool OpenFile(const char *szPath);
    bool CloseFile();

    bool NewArchO(uint8_t *pBase, uint32_t uLength);
    void FreeArchOes();
    bool ReallocCodeSignSpace();

private:
    size_t m_sSize;
    string m_strFile;
    uint8_t *m_pBase;
    bool m_bCSRealloced;
    vector<ZArchO *> m_arrArchOes;
};
