#!/usr/bin/env python3
"""
GitHub Issues Summarizer using LangChain, LangGraph, and MCP.

This script fetches issues from a GitHub repository, filters them by topic,
and generates a summary using an LLM.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class IssuesState(TypedDict, total=False):
    """State for the issues summarization graph."""
    repo: str
    topic: str
    raw_issues: List[Dict[str, Any]]
    filtered_issues: List[Dict[str, Any]]
    summary_text: str


async def fetch_issues_from_github(repo: str, github_token: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch issues from a GitHub repository using MCP.
    
    Args:
        repo: Repository in format "owner/name"
        github_token: GitHub personal access token
        limit: Maximum number of issues to fetch
        
    Returns:
        List of issue dictionaries
    """
    owner, repo_name = repo.split('/')
    
    # Configure the MCP server for GitHub
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={
            **os.environ.copy(),
            "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
        }
    )
    
    # Use MCP client to fetch issues
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Call the MCP tool to list issues
            result = await session.call_tool(
                "list_issues",
                arguments={
                    "owner": owner,
                    "repo": repo_name,
                    "perPage": min(limit, 100),
                    "state": "all"  # Get both open and closed issues
                }
            )
            
            # Parse the result
            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    data = json.loads(content.text)
                    # The response structure may vary, try different keys
                    if isinstance(data, list):
                        return data
                    elif 'items' in data:
                        return data['items']
                    elif 'edges' in data:
                        # GraphQL format
                        return [edge['node'] for edge in data['edges']]
                    else:
                        return [data] if data else []
            
            return []


def fetch_issues_node(state: IssuesState) -> IssuesState:
    """
    Node 1: Fetch issues from GitHub using MCP.
    
    Retrieves the last 100 issues from the specified repository.
    """
    import asyncio
    
    repo = state['repo']
    github_token = os.getenv('GITHUB_TOKEN')
    
    raw_issues = asyncio.run(fetch_issues_from_github(repo, github_token, limit=100))
    
    print(f"✓ Fetched {len(raw_issues)} issues from {repo}")
    
    return {
        **state,
        'raw_issues': raw_issues
    }


def filter_issues_node(state: IssuesState) -> IssuesState:
    """
    Node 2: Filter issues by topic using keyword matching.
    
    Filters issues based on whether the topic keywords appear in the
    issue title or body.
    """
    raw_issues = state['raw_issues']
    topic = state['topic'].lower()
    topic_keywords = topic.split()
    
    filtered = []
    for issue in raw_issues:
        title = issue.get('title', '').lower()
        body = issue.get('body', '') or ''
        body = body.lower()
        
        # Check if any topic keyword appears in title or body
        if any(keyword in title or keyword in body for keyword in topic_keywords):
            filtered.append(issue)
    
    print(f"✓ Filtered to {len(filtered)} issues matching topic '{state['topic']}'")
    
    return {
        **state,
        'filtered_issues': filtered
    }


def summarize_node(state: IssuesState) -> IssuesState:
    """
    Node 3: Generate a summary of filtered issues using LLM.
    
    Uses ChatOpenAI to analyze the filtered issues and produce a
    human-readable summary with statistics and insights.
    """
    filtered_issues = state['filtered_issues']
    repo = state['repo']
    topic = state['topic']
    
    # Count open vs closed
    open_count = sum(1 for issue in filtered_issues if issue.get('state') == 'open')
    closed_count = len(filtered_issues) - open_count
    
    # Prepare issue details for LLM
    issue_summaries = []
    for issue in filtered_issues[:50]:  # Limit to first 50 to avoid token limits
        summary = {
            'number': issue.get('number'),
            'title': issue.get('title'),
            'state': issue.get('state'),
            'body_preview': (issue.get('body') or '')[:300],  # First 300 chars
            'labels': [label.get('name') for label in issue.get('labels', [])]
        }
        issue_summaries.append(summary)
    
    # Create prompt for LLM
    prompt = f"""You are analyzing GitHub issues for the repository: {repo}
Topic of interest: {topic}

Total matching issues: {len(filtered_issues)}
Open: {open_count}
Closed: {closed_count}

Here are details of the matching issues (showing up to 50):

{json.dumps(issue_summaries, indent=2)}

Please provide a comprehensive summary that includes:
1. Overview of the total number of issues and their states
2. Common themes and patterns in these issues
3. Typical error messages or problems mentioned
4. Common resolutions or workarounds (especially from closed issues)
5. Any notable trends or insights

Format your response as a clear, human-readable summary."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    print("✓ Generating summary with LLM...")
    
    # Generate summary
    response = llm.invoke(prompt)
    summary_text = response.content
    
    return {
        **state,
        'summary_text': summary_text
    }


def create_graph() -> Any:
    """
    Create and compile the LangGraph workflow.
    
    The graph executes three nodes in sequence:
    fetch_issues_node → filter_issues_node → summarize_node
    """
    workflow = StateGraph(IssuesState)
    
    # Add nodes
    workflow.add_node("fetch_issues", fetch_issues_node)
    workflow.add_node("filter_issues", filter_issues_node)
    workflow.add_node("summarize", summarize_node)
    
    # Define edges (execution order)
    workflow.set_entry_point("fetch_issues")
    workflow.add_edge("fetch_issues", "filter_issues")
    workflow.add_edge("filter_issues", "summarize")
    workflow.add_edge("summarize", END)
    
    # Compile the graph
    return workflow.compile()


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Summarize GitHub issues by topic using LangChain and MCP"
    )
    parser.add_argument(
        '--repo',
        default='nvm-sh/nvm',
        help='GitHub repository in format owner/name (default: nvm-sh/nvm)'
    )
    parser.add_argument(
        '--topic',
        default='build issues',
        help='Topic to filter issues by (default: build issues)'
    )
    args = parser.parse_args()
    
    # Check required environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)
    
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"GitHub Issues Summarizer")
    print(f"{'='*60}")
    print(f"Repository: {args.repo}")
    print(f"Topic: {args.topic}")
    print(f"{'='*60}\n")
    
    # Create the graph
    graph = create_graph()
    
    # Initialize state
    initial_state: IssuesState = {
        'repo': args.repo,
        'topic': args.topic
    }
    
    # Run the graph
    print("Starting analysis...\n")
    final_state = graph.invoke(initial_state)
    
    # Print the final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    print(final_state['summary_text'])
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
