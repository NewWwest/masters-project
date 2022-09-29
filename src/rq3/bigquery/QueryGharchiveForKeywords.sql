{GHA_TABLE}
{REPOS_TABLE}
{KEYWORDS_TABLE}
{RESULT_TABLE}

CREATE TABLE `{RESULT_TABLE}` as (

    SELECT 
        parsed_events.*, 
        d.string_field_0 as found_keyword
    FROM 
    (
        (SELECT
            JSON_QUERY(gha.payload, '$.issue.url') as issue_url, 
            '' as commit_url,
            JSON_QUERY(gha.payload, '$.comment.url') as entity_url, 
            JSON_QUERY(gha.payload, '$.comment.body') as text_value, 
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'IssueCommentEvent' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
            JSON_QUERY(gha.payload, '$.pull_request.issue_url') as issue_url, 
            '' as commit_url,
            JSON_QUERY(gha.payload, '$.pull_request.url') as entity_url, 
            JSON_QUERY(gha.payload, '$.pull_request.body') as text_value
            FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'PullRequestEvent' and 
            JSON_QUERY(gha.payload, '$.action') = '"closed"' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
        JSON_QUERY(gha.payload, '$.pull_request.issue_url') as issue_url, 
        '' as commit_url,
        JSON_QUERY(gha.payload, '$.pull_request.url') as entity_url, 
        JSON_QUERY(gha.payload, '$.pull_request.title') as text_value
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'PullRequestEvent' and 
            JSON_QUERY(gha.payload, '$.action') = '"closed"' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
        '' as issue_url,
        JSON_QUERY(gha.payload, '$.comment.url') as commit_url,
        JSON_QUERY(gha.payload, '$.comment.html_url') as entity_url, 
        JSON_QUERY(gha.payload, '$.comment.body') as text_value, 
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'CommitCommentEvent' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
        JSON_QUERY(gha.payload, '$.pull_request.issue_url') as issue_url, 
        '' as commit_url,
        JSON_QUERY(gha.payload, '$.comment.url') as entity_url, 
        JSON_QUERY(gha.payload, '$.comment.body') as text_value
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'PullRequestReviewCommentEvent' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
        JSON_QUERY(gha.payload, '$.issue.url') as issue_url, 
        '' as commit_url,
        JSON_QUERY(gha.payload, '$.issue.url') as entity_url, 
        JSON_QUERY(gha.payload, '$.issue.title') as text_value
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'IssuesEvent' and 
            JSON_QUERY(gha.payload, '$.action') = '"closed"' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (SELECT
        JSON_QUERY(gha.payload, '$.issue.url') as issue_url, 
        '' as commit_url,
        JSON_QUERY(gha.payload, '$.issue.url') as entity_url, 
        JSON_QUERY(gha.payload, '$.issue.body') as text_value
        FROM `{GHA_TABLE}` as gha
        WHERE 
            gha.type = 'IssuesEvent' and 
            JSON_QUERY(gha.payload, '$.action') = '"closed"' and 
            gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        UNION ALL
        (WITH commit_sequences AS (
            SELECT 
                JSON_EXTRACT_ARRAY(gha.payload, "$.commits") as commits
            FROM `{GHA_TABLE}` as gha
            WHERE 
                gha.type='PushEvent' and 
                gha.repo.name IN (SELECT string_field_0 FROM `{REPOS_TABLE}`) 
        )
        SELECT 
            '' as issue_url, 
            JSON_QUERY(commits, '$.url') as commit_url,
            JSON_QUERY(commits, '$.url') as entity_url, 
            JSON_QUERY(commits, '$.message') as text_value
        FROM commit_sequences, commit_sequences.commits AS commits
        )
    ) as parsed_events
    LEFT JOIN `{KEYWORDS_TABLE}` as d ON REGEXP_CONTAINS(parsed_events.text_value, d.string_field_0)
)

