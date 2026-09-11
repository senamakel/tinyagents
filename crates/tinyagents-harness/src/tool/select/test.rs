//! Unit tests for prompt-driven tool selection: verb classification,
//! abbreviation expansion, stopword filtering, and ranked overlap scoring.

use super::*;

fn tool<'a>(name: &'a str, desc: &'a str) -> SelectableTool<'a> {
    SelectableTool::new(name, desc)
}

fn github_sample() -> Vec<SelectableTool<'static>> {
    vec![
        tool(
            "GITHUB_CREATE_A_PULL_REQUEST",
            "Creates a pull request in a GitHub repository, requiring existing base and head branches.",
        ),
        tool(
            "GITHUB_CREATE_A_REVIEW_FOR_A_PULL_REQUEST",
            "Creates a pull request review, allowing approval, change requests, or comments.",
        ),
        tool(
            "GITHUB_CREATE_A_DEPLOYMENT_BRANCH_POLICY",
            "Creates a deployment branch or tag policy for an existing environment in a repository.",
        ),
        tool(
            "GITHUB_DELETE_A_REVIEW_COMMENT_FOR_A_PULL_REQUEST",
            "Deletes a review comment on a pull request.",
        ),
        tool(
            "GITHUB_FIND_PULL_REQUESTS",
            "Primary tool to find and search pull requests.",
        ),
        tool(
            "GITHUB_GET_A_PULL_REQUEST",
            "Retrieves a specific pull request by number.",
        ),
        tool(
            "GITHUB_LIST_ASSIGNEES",
            "Lists users who can be assigned to issues in a repository.",
        ),
    ]
}

#[test]
fn create_pr_ranks_create_a_pull_request_first() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("create a PR from my feature branch to main", &tools, 5);
    assert!(!idx.is_empty());
    // Top match must be a CREATE verb tool (not DELETE/GET).
    let top_name = tools[idx[0]].name;
    assert!(
        top_name.contains("CREATE") && top_name.contains("PULL_REQUEST"),
        "expected top match to be a CREATE + PULL_REQUEST tool, got {top_name}"
    );
    // The DELETE tool must not appear — verb gate should drop it.
    for &i in &idx {
        assert!(
            !tools[i].name.starts_with("GITHUB_DELETE"),
            "DELETE tool leaked past verb gate: {}",
            tools[i].name
        );
    }
}

#[test]
fn list_prs_ranks_find_pull_requests_first() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("list open PRs assigned to me", &tools, 5);
    assert!(!idx.is_empty());
    let top_name = tools[idx[0]].name;
    assert!(
        top_name == "GITHUB_FIND_PULL_REQUESTS" || top_name == "GITHUB_LIST_ASSIGNEES",
        "expected FIND_PULL_REQUESTS or LIST_ASSIGNEES on top, got {top_name}"
    );
}

/// Exact-ordering snapshot, captured from the pre-extraction implementation in
/// the OpenHuman host crate before this module was moved up. This is a
/// relevance ranker whose output decides which tools a model is shown, so a
/// scoring change is a silent behaviour change; pin the whole ordering, not
/// just the winner.
#[test]
fn ranking_order_matches_the_pre_extraction_snapshot() {
    let tools = github_sample();
    let cases: &[(&str, &[&str])] = &[
        (
            "create a PR from my feature branch to main",
            &[
                "GITHUB_CREATE_A_PULL_REQUEST",
                "GITHUB_CREATE_A_REVIEW_FOR_A_PULL_REQUEST",
                "GITHUB_CREATE_A_DEPLOYMENT_BRANCH_POLICY",
            ],
        ),
        (
            // `GITHUB_GET_A_PULL_REQUEST` is a `Read` tool under a `List`
            // query. It ranks last of the three: compatible intents earn a
            // smaller bonus than an exact match, so the enumerating tools
            // still lead. Before Read and List were compatible it was gated
            // out entirely, which is what left a search surface unable to
            // retrieve what it found.
            "list open PRs assigned to me",
            &[
                "GITHUB_FIND_PULL_REQUESTS",
                "GITHUB_LIST_ASSIGNEES",
                "GITHUB_GET_A_PULL_REQUEST",
            ],
        ),
        (
            "delete a review comment",
            &["GITHUB_DELETE_A_REVIEW_COMMENT_FOR_A_PULL_REQUEST"],
        ),
        // No MERGE-prefixed tool in the sample, and the gate drops every
        // non-merge verb, so this query is empty by construction.
        ("merge pull request 42", &[]),
    ];
    for (prompt, expected) in cases {
        let got: Vec<&str> = rank_tools_by_prompt(prompt, &tools, 10)
            .into_iter()
            .map(|i| tools[i].name)
            .collect();
        assert_eq!(&got, expected, "ranking drifted for prompt {prompt:?}");
    }
}

#[test]
fn empty_prompt_returns_empty() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("", &tools, 5);
    assert!(idx.is_empty());
}

#[test]
fn empty_catalogue_returns_empty() {
    assert!(rank_tools_by_prompt("create a PR", &[], 5).is_empty());
}

#[test]
fn abbreviation_expansion_works() {
    let qt = query_tokens("create a PR from feature branch");
    assert!(qt.contains("pr"));
    assert!(qt.contains("pull"));
    assert!(qt.contains("request"));
}

#[test]
fn stopwords_removed() {
    let qt = query_tokens("send the email to my manager");
    assert!(!qt.contains("the"));
    assert!(!qt.contains("to"));
    assert!(!qt.contains("my"));
    assert!(qt.contains("send"));
    assert!(qt.contains("email"));
    assert!(qt.contains("manager"));
}

#[test]
fn verb_detection_handles_aliases() {
    // Exact assertion, not `contains(Send) || contains(Create)`: the
    // regression this pins is specifically that `Send` must be retained
    // ALONGSIDE `Create` here, not merely that one of the two survives — an
    // implementation that suppresses `Send` whenever ANY verb is found would
    // still pass a permissive `||` assertion while reintroducing the exact
    // ranking regression (`SLACK_SEND_MESSAGE` falling out of the top-k).
    let v = detect_verbs("post a message to general channel");
    assert_eq!(v, HashSet::from([ToolVerb::Create, ToolVerb::Send]));

    let v = detect_verbs("delete all promotional emails");
    assert!(v.contains(&ToolVerb::Delete));

    let v = detect_verbs("merge pull request 42");
    assert!(v.contains(&ToolVerb::Merge));
}

#[test]
fn resource_noun_does_not_add_send_alongside_an_explicit_conflicting_verb() {
    // "read email" and "delete a message" must not ALSO detect Send from the
    // resource noun — only the explicit action verb should be present.
    let v = detect_verbs("read email");
    assert_eq!(v, HashSet::from([ToolVerb::Read]));

    let v = detect_verbs("delete a message");
    assert_eq!(v, HashSet::from([ToolVerb::Delete]));

    // No explicit verb at all: the resource noun alone may still imply Send.
    let v = detect_verbs("message support channel");
    assert!(v.contains(&ToolVerb::Send));
}

#[test]
fn tool_verb_handles_plurals() {
    assert_eq!(tool_verb("SLACK_DELETES_A_MESSAGE"), Some(ToolVerb::Delete));
    assert_eq!(
        tool_verb("GITHUB_CREATE_A_PULL_REQUEST"),
        Some(ToolVerb::Create)
    );
    assert_eq!(tool_verb("GMAIL_SEND_EMAIL"), Some(ToolVerb::Send));
    assert_eq!(tool_verb("NOTION_QUERY_DATABASE"), Some(ToolVerb::List));
    // Neutral — no verb prefix recognised
    assert_eq!(tool_verb("GITHUB_GIST_COMMENT"), None);
}

#[test]
fn tool_verb_classifies_canonical_lowercase_names() {
    // `ToolSchema::name` is canonical snake_case; comparing raw case against
    // the (uppercase) prefix tables previously left every lowercase name
    // unclassified, silently keeping the verb gate a no-op for real tool
    // catalogues.
    assert_eq!(
        tool_verb("github_delete_a_pull_request"),
        Some(ToolVerb::Delete)
    );
    assert_eq!(
        tool_verb("github_create_a_pull_request"),
        Some(ToolVerb::Create)
    );
    assert_eq!(tool_verb("gmail_send_email"), Some(ToolVerb::Send));
}

#[test]
fn tool_verb_classifies_an_unprefixed_action_slug() {
    // A name with no vendor prefix carries its verb in the first segment;
    // unconditionally stripping the first segment as an assumed vendor
    // prefix discarded the only verb present.
    assert_eq!(tool_verb("create_a_pull_request"), Some(ToolVerb::Create));
    assert_eq!(tool_verb("CREATE_A_PULL_REQUEST"), Some(ToolVerb::Create));
    assert_eq!(tool_verb("delete_message"), Some(ToolVerb::Delete));
}

#[test]
fn delete_query_excludes_create_tools() {
    let tools = vec![
        tool("GMAIL_SEND_EMAIL", "Sends an email."),
        tool("GMAIL_DELETE_MESSAGE", "Deletes a message by id."),
        tool("GMAIL_DELETE_THREAD", "Deletes a thread."),
        tool("GMAIL_BATCH_DELETE_MESSAGES", "Bulk delete messages."),
    ];
    let idx = rank_tools_by_prompt("delete all promotional emails", &tools, 10);
    for &i in &idx {
        assert!(
            tools[i].name.contains("DELETE"),
            "non-DELETE tool leaked: {}",
            tools[i].name
        );
    }
    assert!(idx.len() >= 3);
}

#[test]
fn a_list_query_keeps_the_tool_that_retrieves_what_it_found() {
    // Finding something and reading it are one task for the user and two
    // verbs for the catalogue. A `List` query must not gate out every
    // `Read` tool, or the surface can enumerate ids and never return
    // content.
    let tools = github_sample();
    let got: Vec<&str> = rank_tools_by_prompt("find the pull requests about auth", &tools, 10)
        .into_iter()
        .map(|i| tools[i].name)
        .collect();
    assert!(
        got.contains(&"GITHUB_GET_A_PULL_REQUEST"),
        "a Read tool must survive a List query: {got:?}"
    );
}

#[test]
fn an_exact_verb_match_still_outranks_a_merely_compatible_one() {
    let tools = github_sample();
    let got: Vec<&str> = rank_tools_by_prompt("list open PRs assigned to me", &tools, 10)
        .into_iter()
        .map(|i| tools[i].name)
        .collect();
    let list_at = got
        .iter()
        .position(|n| *n == "GITHUB_FIND_PULL_REQUESTS")
        .expect("the List tool is kept");
    let read_at = got
        .iter()
        .position(|n| *n == "GITHUB_GET_A_PULL_REQUEST")
        .expect("the Read tool is kept");
    assert!(
        list_at < read_at,
        "compatibility must not promote a Read tool above an exact List match: {got:?}"
    );
}

#[test]
fn compatibility_is_one_directional_and_narrow() {
    use super::verbs_are_compatible;
    assert!(verbs_are_compatible(ToolVerb::List, ToolVerb::Read));
    // Not the reverse: "read message 5" should not pull in every list tool.
    assert!(!verbs_are_compatible(ToolVerb::Read, ToolVerb::List));
    // And nothing else pairs up.
    assert!(!verbs_are_compatible(ToolVerb::List, ToolVerb::Delete));
    assert!(!verbs_are_compatible(ToolVerb::Create, ToolVerb::Read));
    assert!(verbs_are_compatible(ToolVerb::Create, ToolVerb::Create));
}

#[test]
fn strong_name_overlap_can_outrank_a_bare_verb_match_and_that_is_deliberate() {
    // The score is `weighted_overlap + verb_bonus`, and the two bonuses
    // differ by 2 — so a compatible `Read` tool whose *name* matches the
    // query can rank above an exact-verb `List` tool that matches nothing
    // else. That is the ranking working, not a defect: a name hit is worth
    // 3 and is the stronger relevance signal. Sorting by verb class first
    // would bury the tool the user actually named.
    let tools = vec![
        // Exact verb (List, +3), zero token overlap.
        SelectableTool::new("GITHUB_LIST_GISTS", "List gists"),
        // Compatible verb (Read, +1) but three name hits (3 * 3 = 9).
        SelectableTool::new(
            "GITHUB_GET_A_PULL_REQUEST_COMMENT",
            "Get one review comment",
        ),
    ];
    let got: Vec<&str> = rank_tools_by_prompt("find the pull request comment", &tools, 10)
        .into_iter()
        .map(|i| tools[i].name)
        .collect();
    assert_eq!(
        got.first(),
        Some(&"GITHUB_GET_A_PULL_REQUEST_COMMENT"),
        "the tool the query names must lead: {got:?}"
    );
}

#[test]
fn with_overlap_equal_the_exact_verb_wins() {
    // The companion to the case above: strip the overlap advantage and the
    // verb bonus is what decides, so an exact match leads a compatible one.
    let tools = vec![
        SelectableTool::new("GITHUB_GET_PULL_REQUEST", "Get a pull request"),
        SelectableTool::new("GITHUB_LIST_PULL_REQUEST", "List pull requests"),
    ];
    let got: Vec<&str> = rank_tools_by_prompt("find the pull request", &tools, 10)
        .into_iter()
        .map(|i| tools[i].name)
        .collect();
    assert_eq!(
        got.first(),
        Some(&"GITHUB_LIST_PULL_REQUEST"),
        "equal overlap → the exact verb match leads: {got:?}"
    );
}
