#!/usr/bin/env python3
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
    