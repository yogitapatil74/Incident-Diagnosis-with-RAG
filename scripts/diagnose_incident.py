#!/usr/bin/env python3
"""
scripts/diagnose_incident.py
─────────────────────────────
CLI tool to run a single incident through the RAG diagnosis pipeline.
Useful for quick testing and demonstrating the system end-to-end.

Usage:
  python scripts/diagnose_incident.py --incident data/samples/incidents/sample_incidents.json
  python scripts/diagnose_incident.py --title "DB timeout" --description "HikariPool errors" --service payments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def print_result(result):
    """Pretty-print a DiagnosisResult to the terminal."""

    # Header
    escalation_indicator = "[bold red]⚠ ESCALATION REQUIRED[/bold red]" if result.requires_escalation else "[green]✓ No escalation needed[/green]"
    console.print(Panel(
        f"[bold]Incident:[/bold] {result.incident_id}\n"
        f"[bold]Confidence:[/bold] {result.confidence_score:.0%}\n"
        f"{escalation_indicator}\n\n"
        f"[italic]{result.diagnostic_summary}[/italic]",
        title="[bold blue]RAG-AIOps Diagnosis Result",
        border_style="blue",
    ))

    # Root Causes
    console.print("\n[bold cyan]Root Cause Hypotheses:[/bold cyan]")
    for i, rc in enumerate(result.root_causes, 1):
        bar = "█" * int(rc.confidence * 20)
        console.print(f"  {i}. {rc.hypothesis}")
        console.print(f"     Confidence: [green]{bar}[/green] {rc.confidence:.0%}")
        console.print(f"     Evidence: [dim]{rc.evidence}[/dim]")

    # Remediation Steps
    console.print("\n[bold cyan]Remediation Steps:[/bold cyan]")
    for i, step in enumerate(result.remediation_steps, 1):
        console.print(f"  [bold]{i}.[/bold] {step}")

    # Related Docs
    if result.related_docs:
        console.print("\n[bold cyan]Related Knowledge Base Documents:[/bold cyan]")
        for doc in result.related_docs:
            console.print(f"  → [blue]{doc.title}[/blue]")
            console.print(f"    [dim]{doc.relevance}[/dim]")

    # Stats
    # DiagnosisResult has no `retrieved_context_count` attribute (that key
    # only exists in to_dict()); using it crashed every CLI run at the end.
    console.print(f"\n[dim]Retrieved {len(result.retrieved_chunks)} knowledge base chunks | "
                  f"Latency: {result.latency_ms:.0f}ms[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Run RAG-AIOps incident diagnosis")
    parser.add_argument("--incident", help="Path to incident JSON file")
    parser.add_argument("--index", type=int, default=0,
                        help="Which incident to use if the JSON file holds a list (default: 0)")
    parser.add_argument("--title", help="Incident title (alternative to --incident)")
    parser.add_argument("--description", help="Incident description")
    parser.add_argument("--service", help="Affected service name")
    parser.add_argument("--priority", default="P2")
    parser.add_argument("--error-message", help="Error message or stack trace")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--local-embeddings", action="store_true",
                        help="Use local embeddings (no OpenAI key needed)")
    parser.add_argument("--output", help="Save result to JSON file")
    args = parser.parse_args()

    # Build incident from args or file
    if args.incident:
        with open(args.incident) as f:
            data = json.load(f)
            if isinstance(data, list):
                if not data:
                    console.print("[red]Incident file contains an empty list[/red]")
                    sys.exit(1)
                if not 0 <= args.index < len(data):
                    console.print(f"[red]--index must be between 0 and {len(data) - 1}[/red]")
                    sys.exit(1)
                data = data[args.index]
    elif args.title:
        data = {
            "id": "CLI-001",
            "title": args.title,
            "description": args.description or args.title,
            "service": args.service,
            "priority": args.priority,
            "error_message": args.error_message,
        }
    else:
        console.print("[red]Provide --incident or --title[/red]")
        sys.exit(1)

    console.rule("[bold blue]RAG-AIOps Incident Diagnosis")
    console.print(f"\n[bold]Incident:[/bold] {data.get('title', 'Unknown')}")
    console.print(f"[bold]Service:[/bold] {data.get('service', 'N/A')}")
    console.print(f"[bold]Priority:[/bold] {data.get('priority', 'N/A')}\n")

    # Load config and build components
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.local_embeddings:
        config["embedding"]["provider"] = "local"

    from src.diagnosis.diagnoser import IncidentContext
    from src.pipeline import build_pipeline

    with console.status("[bold green]Initialising RAG pipeline..."):
        pipeline = build_pipeline(config)
        diagnoser = pipeline.diagnoser
        if pipeline.vector_store.total_vectors == 0:
            console.print("[yellow]⚠ Vector index is empty — run "
                          "scripts/index_knowledge_base.py first.[/yellow]")

    with console.status("[bold green]Running diagnosis..."):
        incident = IncidentContext(
            # Accept both ticket files ("id") and eval files ("incident_id").
            incident_id=str(data.get("id") or data.get("incident_id") or "CLI-001"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            service=data.get("service"),
            priority=data.get("priority"),
            environment=data.get("environment"),
            error_message=data.get("error_message"),
            affected_components=data.get("affected_components") or [],
            recent_deployments=data.get("recent_deployments") or [],
        )
        result = diagnoser.diagnose(incident)

    print_result(result)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[dim]Result saved to {args.output}[/dim]")


if __name__ == "__main__":
    main()
