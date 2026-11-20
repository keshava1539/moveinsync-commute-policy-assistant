import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings


class PolicyStore:
    """Thread-safe persistent store tracking active and staged policy versions per tenant."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or settings.POLICY_STORE_FILE
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Ensure the backing JSON file exists."""
        if not os.path.exists(self.file_path):
            dir_name = os.path.dirname(self.file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _read_data(self) -> Dict[str, Any]:
        """Read data from file under lock."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_data_atomic(self, data: Dict[str, Any]) -> None:
        """Write data atomically via a temporary file and atomic replace."""
        dir_name = os.path.dirname(os.path.abspath(self.file_path))
        os.makedirs(dir_name, exist_ok=True)
        
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name

        os.replace(temp_name, self.file_path)

    def get_active_version(self, company_id: str) -> Optional[int]:
        """Get currently active policy version for a company."""
        with self._lock:
            data = self._read_data()
            tenant_info = data.get(company_id.strip().lower())
            if not tenant_info:
                return None
            return tenant_info.get("active_version")

    def get_next_staging_version(self, company_id: str) -> int:
        """Determine the next monotonically increasing version number to stage."""
        with self._lock:
            data = self._read_data()
            tenant_info = data.get(company_id.strip().lower(), {})
            active_v = tenant_info.get("active_version", 0)
            max_v = tenant_info.get("max_version", active_v)
            return max(active_v, max_v) + 1

    def commit_version(
        self,
        company_id: str,
        version: int,
        doc_name: str,
        chunks_count: int
    ) -> None:
        """Phase 2 Atomic Pointer Swap:
        Sets active_version to the newly validated version and records audit history.
        """
        c_id = company_id.strip().lower()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with self._lock:
            data = self._read_data()
            if c_id not in data:
                data[c_id] = {
                    "active_version": version,
                    "max_version": version,
                    "versions": {}
                }
            else:
                data[c_id]["active_version"] = version
                data[c_id]["max_version"] = max(data[c_id].get("max_version", version), version)

            data[c_id]["versions"][str(version)] = {
                "doc_name": doc_name,
                "chunks_count": chunks_count,
                "updated_at": timestamp
            }
            self._write_data_atomic(data)

    def get_policy_info(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full version metadata for an active policy."""
        with self._lock:
            data = self._read_data()
            c_id = company_id.strip().lower()
            tenant_info = data.get(c_id)
            if not tenant_info or not tenant_info.get("active_version"):
                return None

            active_v = tenant_info["active_version"]
            v_meta = tenant_info.get("versions", {}).get(str(active_v), {})
            return {
                "company_id": c_id,
                "active_version": active_v,
                "document_name": v_meta.get("doc_name", f"{c_id.capitalize()} Policy"),
                "total_chunks": v_meta.get("chunks_count", 0),
                "updated_at": v_meta.get("updated_at", "Unknown"),
                "status": "active"
            }

    def reset_all(self) -> None:
        """Reset the store (used in testing)."""
        with self._lock:
            self._write_data_atomic({})


policy_store = PolicyStore()
