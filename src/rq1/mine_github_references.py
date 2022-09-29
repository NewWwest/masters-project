import sys
sys.path.insert(0, r'D:\Projects\2022')

from src.mining.VulnerabilityMiners.GitHubReferencesMiner import GitHubReferencesMiner
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
from src.proxies.GitHubProxy import GithubProxy

checkpoint_path = 'results/github_references_in_vuln'
checkpoint_frequency = 100


def main():
    nvd = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    osv = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    ghsa = OsvLoader('/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)
    githubProxy = GithubProxy()
    
    miner = GitHubReferencesMiner(githubProxy, checkpoint_path, checkpoint_frequency)
    miner.process_vulnerabilities(omni)


if __name__ == '__main__':
    main()
    