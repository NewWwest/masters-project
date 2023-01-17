#!/usr/bin/env python3
#
# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------

# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

from src.mining.VulnerabilityMiners.GitHubReferencesMiner import GitHubReferencesMiner
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
from src.proxies.GitHubProxy import GithubProxy

path_to_nvd_dump_json_files = 'path_to_nvd_dump_json_files'
path_to_osv_data_dump = 'path_to_osv_data_dump'
path_to_ghsa_data_dump = 'path_to_ghsa_data_dump'
checkpoint_path = 'results/github_references_in_vuln'
checkpoint_frequency = 100


def main():
    nvd = NvdLoader(path_to_nvd_dump_json_files)
    osv = OsvLoader(path_to_osv_data_dump)
    ghsa = OsvLoader(path_to_ghsa_data_dump)
    omni = OmniLoader(nvd, osv, ghsa)
    githubProxy = GithubProxy()
    
    miner = GitHubReferencesMiner(githubProxy, checkpoint_path, checkpoint_frequency)
    miner.process_vulnerabilities(omni)


if __name__ == '__main__':
    main()
    