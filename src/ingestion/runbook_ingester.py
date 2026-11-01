"""
src/ingestion/runbook_ingester.py
──────────────────────────────────
Ingests runbooks and post-mortem reports from:
- Local Markdown / text files
- Confluence spaces (via REST API)
- GitHub repositories (via GitHub API)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import structlog

from .base import BaseIngester, DocumentType, OperationalDocument

logger = structlog.get_logger(__name__)


class LocalRunbookIngester(BaseIngester):
    """
    Ingests Markdown/text runbooks and post-mortems from local directories.
    Classifies document type based on directory name or file content keywords.
    """

    POST_MORTEM_KEYWORDS = {
        "post-mortem", "postmortem", "post_mortem",
        "incident report", "retrospective"
    }

    def validate_connection(self) -> bool:
        return os.path.exists(self.config.get("path", "data/samples"))

    def ingest(self) -> Iterator[OperationalDocument]:
        base_path = Path(self.config.get("path", "data/samples"))
        patterns = self.config.get("patterns", ["*.md", "*.txt", "*.rst"])
        recursive = self.config.get("recursive", True)

        for pattern in patterns:
            glob_fn = base_path.rglob if recursive else base_path.glob
            for file_path in glob_fn(pattern):
                try:
                    yield self._file_to_document(file_path)
                except Exception as e:
                    self.logger.warning(
                        "Failed to ingest file",
                        path=str(file_path),
                        error=str(e),
                    )

    def _file_to_document(self, file_path: Path) -> OperationalDocument:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        content = self._clean_text(content)
        content = self._redact_sensitive(content)

        doc_type = self._classify_document(file_path, content)
        title = self._extract_title(content) or file_path.stem.replace("-", " ").title()

        stat = file_path.stat()
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        # Extract section headings for metadata
        headings = re.findall(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)

        return OperationalDocument(
            document_id=f"runbook_{file_path.stem}_{self._stable_hash(content)}",
            source_system="local_files",
            document_type=doc_type,
            title=title,
            content_text=content,
            created_at=created,
            last_modified=modified,
            metadata={
                "file_path": str(file_path),
                "file_name": file_path.name,
                "section_headings": headings[:10],  # first 10 sections
                "language": self._detect_language(content),
                "word_count": len(content.split()),
            },
        )

    def _classify_document(self, path: Path, content: str) -> DocumentType:
        # The path (directory / file name) is the stronger signal, so check it
        # before content keywords. Previously a runbook that merely mentioned
        # "post-mortem" in its first 500 chars (e.g. "see the post-mortem for
        # INC-123") was misclassified as a post-mortem.
        path_str = str(path).lower()
        if any(k in path_str for k in self.POST_MORTEM_KEYWORDS):
            return DocumentType.POST_MORTEM
        if "runbook" in path_str or "playbook" in path_str:
            return DocumentType.RUNBOOK
        content_lower = content[:500].lower()
        if any(k in content_lower for k in self.POST_MORTEM_KEYWORDS):
            return DocumentType.POST_MORTEM
        return DocumentType.DOCUMENTATION

    @staticmethod
    def _extract_title(content: str) -> str | None:
        """Extract title from first Markdown H1 heading."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else None


class ConfluenceIngester(BaseIngester):
    """
    Ingests pages from Confluence Cloud spaces via REST API v2.
    Strips HTML to plain text before indexing.
    """

    def validate_connection(self) -> bool:
        try:
            import requests
            resp = requests.get(
                f"{self.config['url']}/rest/api/space",
                auth=(self.config["username"], self.config["api_token"]),
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def ingest(self) -> Iterator[OperationalDocument]:
        import requests
        from bs4 import BeautifulSoup

        base_url = self.config["url"]
        auth = (self.config["username"], self.config["api_token"])
        spaces = self.config.get("spaces", [])

        for space_key in spaces:
            start = 0
            limit = 50
            while True:
                resp = requests.get(
                    f"{base_url}/rest/api/content",
                    auth=auth,
                    params={
                        "spaceKey": space_key,
                        "type": "page",
                        "expand": "body.storage,version,space",
                        "start": start,
                        "limit": limit,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("results", [])
                if not pages:
                    break

                for page in pages:
                    html = page["body"]["storage"]["value"]
                    plain = BeautifulSoup(html, "html.parser").get_text(
                        separator="\n", strip=True
                    )
                    plain = self._clean_text(plain)
                    plain = self._redact_sensitive(plain)

                    yield OperationalDocument(
                        document_id=f"confluence_{page['id']}",
                        source_system="confluence",
                        document_type=DocumentType.DOCUMENTATION,
                        title=page["title"],
                        content_text=plain,
                        created_at=datetime.now(timezone.utc),
                        last_modified=datetime.now(timezone.utc),
                        metadata={
                            "space_key": space_key,
                            "confluence_id": page["id"],
                            "url": f"{base_url}/wiki{page.get('_links', {}).get('webui', '')}",
                        },
                    )

                start += limit
                if len(pages) < limit:
                    break
