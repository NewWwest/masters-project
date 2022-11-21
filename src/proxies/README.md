# Proxies

Proxies implement the communication with external systems. Their main roles are simplifying the API, enforcing rate limits or providing on share location of data storage.

## GitHubProxy

Provides a number of methods for fetching objects from the GitHub API. Most importantly it sleeps prior to each request to enforce rate limits. It can do multiple requests per one entity, for example when fetching an issue it will make multiple call for the timeline, comments and pull request data as well.

## RepoDownloader

Used to clone repositories.

## GcCloudStorageProxy & LocalStorageProxy

Used to intract with Google Cloud - Cloud Storage. Provides ways to upload a file, zip and upload multiple files and check if file exists. The Localstorage proxy may be used instead as a mockup.