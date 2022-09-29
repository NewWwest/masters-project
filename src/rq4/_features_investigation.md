
| FEATURE  | Influence | Reason |
|---|---|---|
|hours_diff|drop by around 0.01++ | Fixes are merged fast
|changed_files|drop by around 0.01| Fixes are small
|changed_lines|drop below 0.01| Fixes are small
|is_merge|slightly improved precission| ??
|avg_removed_count|drop below 0.01| fixes will remove less code
|avg_mod_count') | sligthly improved recall| fixes will remove less code
|avg_added_count|drop below 0.01| fixes should mostly add code
|avg_file_size|drop by around 0.01++| Fixes will apear in large files
|dmm_unit_complexity|drop by around 0.01++| Fixes complicate code
|dmm_unit_interfacing|drop by around 0.02++| Fixes complicate code
|dmm_unit_size|drop by around 0.01| Fixes complicate code
|avg_changed_methods|drop below 0.001, recall dropped by 0.02| Fixes will not add many methods
|avg_complexity|drop below 0.01| Fixes will appear in complicated code
|avg_nloc|drop by around 0.02++| Fixes will appear in complicated code
|avg_tokens|drop by around 0.01| Fixes will appear in complicated code
|contains_cwe_title|drop below 0.01| Security fix may be indicated in code/message
|contains_security_info|drop by around 0.03| Security fix may be indicated in code/message
|contains_security_in_message|drop below 0.01| Security fix may be indicated in code/message
|biggest_file_size | Unknown | Fix may indicate a release which introduces a large file
|biggest_file_change | Unknown | Fix may indicate a release which introduces a large file