"""
src/ingestion/ticket_ingester.py
─────────────────────────────────
Ingestion connector for incident tickets.
Supports ServiceNow REST API, PagerDuty API,
and local JSON/CSV file ingestion for PoC/testing.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Iterator

import requests
import structlog

from .base import BaseIngester, DocumentType, OperationalDocument

logger = structlog.get_logger(__name__)


class LocalTicketIngester(BaseIngester):
    """
    Ingests incident tickets from local JSON files.
    Used for PoC testing and evaluation dataset loading.

    Expected JSON format (list of objects):
    [
      {
        "id": "INC001",
        "title": "Payment service timeout",
        "description": "...",
        "resolution_notes": "...",
        "priority": "P1",
        "service": "payments-service",
        "status": "resolved",
        "created_at": "2024-01-15T10:30:00Z",
        "resolved_at": "2024-01-15T11:45:00Z",
        "root_cause": "Database connection pool exhaustion",
        "tags": ["database", "performance"]
      }
    ]
    """

    def validate_connection(self) -> bool:
        path = self.config.get("path", "data/samples/incidents")
        return os.path.exists(path)

    def ingest(self) -> Iterator[OperationalDocument]:
        path = self.config.get("path", "data/samples/incidents")

        if os.path.isfile(path):
            files = [path]
        else:
            files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith((".json", ".csv"))
            )

        for file_path in files:
            self.logger.info("Ingesting tickets file", path=file_path)
            yield from self._ingest_file(file_path)

    def _ingest_file(self, file_path: str) -> Iterator[OperationalDocument]:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                tickets = json.load(f)
            if isinstance(tickets, dict):
                tickets = [tickets]  # allow a single ticket object per file
            for ticket in tickets:
                yield self._ticket_to_document(ticket)

        elif file_path.endswith(".csv"):
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = dict(row)
                    # CSV has no lists — accept "a,b" or "a;b" for tags.
                    if isinstance(row.get("tags"), str):
                        row["tags"] = [
                            t.strip() for t in row["tags"].replace(";", ",").split(",")
                            if t.strip()
                        ]
                    yield self._ticket_to_document(row)

    def _ticket_to_document(self, ticket: dict) -> OperationalDocument:
        # Compose rich content text for embedding quality
        content_parts = []
        if ticket.get("title"):
            content_parts.append(f"Incident: {ticket['title']}")
        if ticket.get("priority"):
            content_parts.append(f"Priority: {ticket['priority']}")
        if ticket.get("service"):
            content_parts.append(f"Affected Service: {ticket['service']}")
        if ticket.get("description"):
            content_parts.append(f"Description:\n{ticket['description']}")
        if ticket.get("error_message"):
            content_parts.append(f"Error:\n{ticket['error_message']}")
        if ticket.get("resolution_notes"):
            content_parts.append(f"Resolution:\n{ticket['resolution_notes']}")
        if ticket.get("root_cause"):
            content_parts.append(f"Root Cause: {ticket['root_cause']}")

        content = self._clean_text("\n\n".join(content_parts))
        content = self._redact_sensitive(content)

        created = self._parse_datetime(ticket.get("created_at"))
        modified = self._parse_datetime(
            ticket.get("resolved_at") or ticket.get("created_at")
        )

        return OperationalDocument(
            document_id=f"ticket_{ticket.get('id') or ticket.get('incident_id') or self._stable_hash(content, 12)}",
            source_system="ticket_system",
            document_type=DocumentType.INCIDENT_TICKET,
            title=ticket.get("title", "Untitled Incident"),
            content_text=content,
            created_at=created,
            last_modified=modified,
            metadata={
                "priority": ticket.get("priority", "unknown"),
                "service": ticket.get("service", "unknown"),
                "status": ticket.get("status", "unknown"),
                "root_cause": ticket.get("root_cause", ""),
                "tags": ticket.get("tags", []),
                "environment": ticket.get("environment", "production"),
            },
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)


class ServiceNowIngester(BaseIngester):
    """
    Ingests incident tickets from ServiceNow REST API.
    Requires ServiceNow instance URL, username, and password.
    """

    BASE_FIELDS = (
        "sys_id,number,short_description,description,close_notes,"
        "priority,state,u_service_name,sys_created_on,sys_updated_on,"
        "u_root_cause,u_resolution_summary,category"
    )

    def validate_connection(self) -> bool:
        url = self.config.get("instance_url")
        try:
            resp = requests.get(
                f"{url}/api/now/table/incident?sysparm_limit=1",
                auth=(self.config["username"], self.config["password"]),
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            self.logger.error("ServiceNow connection failed", error=str(e))
            return False

    def ingest(self) -> Iterator[OperationalDocument]:
        url = self.config["instance_url"]
        auth = (self.config["username"], self.config["password"])
        batch_size = self.config.get("batch_size", 100)
        lookback_days = self.config.get("lookback_days", 90)
        offset = 0

        while True:
            resp = requests.get(
                f"{url}/api/now/table/incident",
                auth=auth,
                params={
                    # Return {"value", "display_value"} objects for reference /
                    # choice fields. _record_to_document reads display_value.
                    "sysparm_display_value": "all",
                    "sysparm_fields": self.BASE_FIELDS,
                    "sysparm_limit": batch_size,
                    "sysparm_offset": offset,
                    "sysparm_query": (
                        f"state=6^"  # Resolved
                        f"sys_created_on>javascript:gs.daysAgo({lookback_days})"
                    ),
                },
                timeout=30,
            )
            resp.raise_for_status()
            records = resp.json().get("result", [])
            if not records:
                break

            for record in records:
                yield self._record_to_document(record)

            offset += batch_size
            if len(records) < batch_size:
                break

    @staticmethod
    def _field(record: dict, name: str) -> str:
        """
        Read a ServiceNow field that may be a plain string (default API
        output) or a {"value", "display_value"} dict (sysparm_display_value).
        The old code called .get("display_value") on fields such as
        `priority` and `state`, which are plain strings unless display values
        are requested — an AttributeError on every single record.
        """
        val = record.get(name, "")
        if isinstance(val, dict):
            return str(val.get("display_value") or val.get("value") or "")
        return "" if val is None else str(val)

    def _record_to_document(self, record: dict) -> OperationalDocument:
        f = lambda name: self._field(record, name)  # noqa: E731
        content_parts = [
            f"Incident: {f('short_description')}",
            f"Category: {f('category')}",
            f"Priority: {f('priority')}",
            f"Service: {f('u_service_name')}",
            f"Description:\n{f('description')}",
            f"Resolution:\n{f('close_notes')}",
            f"Root Cause: {f('u_root_cause')}",
        ]
        content = self._clean_text("\n\n".join(content_parts))
        content = self._redact_sensitive(content)

        def _dt(name):
            raw = record.get(name, "")
            # With display values the timestamp's raw form is in "value".
            val = raw.get("value", "") if isinstance(raw, dict) else raw
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                return datetime.now(timezone.utc)

        sys_id = record.get("sys_id", "")
        if isinstance(sys_id, dict):
            sys_id = sys_id.get("value", "")

        return OperationalDocument(
            document_id=f"snow_{sys_id}",
            source_system="servicenow",
            document_type=DocumentType.INCIDENT_TICKET,
            title=f("short_description"),
            content_text=content,
            created_at=_dt("sys_created_on"),
            last_modified=_dt("sys_updated_on"),
            metadata={
                "number": f("number"),
                "priority": f("priority"),
                "service": f("u_service_name"),
                "state": f("state"),
            },
        )
