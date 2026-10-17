"""Police sign-in: accounts, sessions, lockout and the audit trail.

- Passwords: salted scrypt (stdlib). Accounts are created only with `aequitas create-user`, which prompts
  for the password, so it never appears in shell history or config files.
- Sessions: a random token in an HttpOnly cookie; only its sha256 is stored.
- Lockout: too many failed logins for a username from one address blocks it for a while.
- Audit: sign-ins and report actions are recorded (who, what, when, from where). Entries form a hash chain:
  each one's HMAC-SHA256 covers its fields and the previous entry's hash, keyed by a secret file next to the
  database (data/audit.key). Editing, deleting or reordering an entry breaks the chain from that entry on, and
  without the key the chain can't be recomputed. It is tamper-evident, not tamper-proof: whoever holds the key
  and the database could rewrite both, so the latest hash is also written into every evidence export."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import os
import threading
import time
from pathlib import Path

ROLES = ("admin", "officer")
GENESIS = "0" * 64  # prev_hash of the first audit entry
MIN_PASSWORD = 10
_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(base64.b64decode(digest)))
        return hmac.compare_digest(got, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password(secrets.token_hex(8))  # verified for unknown users, so timing does not reveal them


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _load_key(path: Path | None) -> bytes:
    """The audit chain's HMAC key: 32 random bytes in a file created on first use (owner-only where the OS
    allows). In-memory databases (tests) get a fresh key per run."""
    if path is None:
        return secrets.token_bytes(32)
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


class LockedOut(Exception):
    pass


class Auth:
    def __init__(self, store, session_hours: float = 12, max_failures: int = 5, lockout_s: float = 900,
                 key_path: str | Path | None = None):
        self.store = store
        self._key = _load_key(key_path or (None if str(store.path) == ":memory:" else store.path.parent / "audit.key"))
        self._seal_legacy()
        self.session_s = session_hours * 3600
        self.max_failures = max_failures
        self.lockout_s = lockout_s
        self._failures: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def _q(self, sql: str, params: tuple = (), commit: bool = False):
        with self.store._lock:
            if commit:
                with self.store._conn:
                    return self.store._conn.execute(sql, params)
            return self.store._conn.execute(sql, params).fetchall()

    # ---------- accounts ----------

    def create_user(self, username: str, password: str, role: str = "officer", allow_weak: bool = False) -> int:
        """allow_weak: demo accounts only (e.g. admin/admin on a laptop that serves 127.0.0.1)."""
        username = username.strip()
        if not username or len(username) > 40:
            raise ValueError("username must be 1-40 characters")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if len(password) < MIN_PASSWORD and not allow_weak:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
        cur = self._q("INSERT INTO users(username, password_hash, role, created_at) VALUES (?,?,?,?)",
                      (username, hash_password(password), role, int(time.time() * 1000)), commit=True)
        self.audit(None, "user_created", username, {"role": role, "weak_password": len(password) < MIN_PASSWORD})
        return cur.lastrowid

    def users(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT user_id, username, role, disabled, created_at FROM users")]

    # ---------- sign-in ----------

    def _locked(self, key: tuple[str, str]) -> bool:
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < self.lockout_s]
            self._failures[key] = recent
            return len(recent) >= self.max_failures

    def login(self, username: str, password: str, ip: str = "") -> tuple[str, dict] | None:
        """Returns (token, user) or None. Raises LockedOut after too many failures."""
        key = (ip, username.strip().lower())
        if self._locked(key):
            raise LockedOut()
        rows = self._q("SELECT * FROM users WHERE username=? AND disabled=0", (username.strip(),))
        user = dict(rows[0]) if rows else None
        ok = verify_password(password, user["password_hash"] if user else _DUMMY) and user is not None
        if not ok:
            with self._lock:
                self._failures.setdefault(key, []).append(time.time())
            self.audit(None, "login_failed", username.strip()[:40], None, ip)
            return None
        with self._lock:
            self._failures.pop(key, None)
        token = secrets.token_urlsafe(32)
        now = int(time.time() * 1000)
        self._q("INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                (_token_hash(token), user["user_id"], now, now + int(self.session_s * 1000)), commit=True)
        self.audit(user["user_id"], "login", user["username"], None, ip)
        return token, self.public(user)

    def session_user(self, token: str | None) -> dict | None:
        if not token:
            return None
        rows = self._q("SELECT u.* FROM sessions s JOIN users u USING(user_id) "
                       "WHERE s.token_hash=? AND s.expires_at>? AND u.disabled=0",
                       (_token_hash(token), int(time.time() * 1000)))
        return self.public(dict(rows[0])) if rows else None

    def logout(self, token: str | None) -> None:
        if token:
            self._q("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),), commit=True)

    @staticmethod
    def public(user: dict) -> dict:
        return {"user_id": user["user_id"], "username": user["username"], "role": user["role"]}

    # ---------- audit ----------

    def _entry_hash(self, prev: str, r) -> str:
        msg = json.dumps([prev, r["id"], r["user_id"], r["action"], r["target"], r["detail"], r["ip"],
                          r["created_at"]], separators=(",", ":"), ensure_ascii=False)
        return hmac.new(self._key, msg.encode(), hashlib.sha256).hexdigest()

    def _chain_locked(self, entry_id: int) -> None:
        """Seal one entry onto the chain. The caller holds the store lock inside a transaction."""
        conn = self.store._conn
        last = conn.execute("SELECT hash FROM audit_log WHERE id<? AND hash IS NOT NULL ORDER BY id DESC LIMIT 1",
                            (entry_id,)).fetchone()
        prev = last["hash"] if last else GENESIS
        row = conn.execute("SELECT * FROM audit_log WHERE id=?", (entry_id,)).fetchone()
        conn.execute("UPDATE audit_log SET prev_hash=?, hash=? WHERE id=?", (prev, self._entry_hash(prev, row), entry_id))

    def _seal_legacy(self) -> None:
        """Entries written before the chain existed get sealed once, in order, on the first start."""
        with self.store._lock, self.store._conn:
            ids = [r["id"] for r in self.store._conn.execute("SELECT id FROM audit_log WHERE hash IS NULL ORDER BY id")]
            for i in ids:
                self._chain_locked(i)

    def audit(self, user_id: int | None, action: str, target: str | None = None, detail=None, ip: str = "") -> None:
        with self.store._lock, self.store._conn:
            cur = self.store._conn.execute(
                "INSERT INTO audit_log(user_id, action, target, detail, ip, created_at) VALUES (?,?,?,?,?,?)",
                (user_id, action, target, json.dumps(detail) if detail is not None else None, ip,
                 int(time.time() * 1000)))
            self._chain_locked(cur.lastrowid)

    def verify_chain(self) -> dict:
        """Recompute the whole chain. ok=False names the first entry that doesn't match: edited, deleted
        (its successor's prev_hash no longer points at anything), inserted or reordered."""
        rows = self._q("SELECT * FROM audit_log ORDER BY id")
        prev = GENESIS
        for r in rows:
            if r["prev_hash"] != prev:
                return {"ok": False, "checked": len(rows), "broken_at": r["id"],
                        "reason": "the link to the previous entry is broken (an entry was deleted or inserted)"}
            if r["hash"] is None or not hmac.compare_digest(r["hash"], self._entry_hash(prev, r)):
                return {"ok": False, "checked": len(rows), "broken_at": r["id"],
                        "reason": "this entry was changed after it was written"}
            prev = r["hash"]
        return {"ok": True, "checked": len(rows), "broken_at": None, "reason": None,
                "head": prev, "head_id": rows[-1]["id"] if rows else None}

    def audit_log(self, limit: int = 200, action: str | None = None, user: str | None = None,
                  target: str | None = None, before_id: int | None = None) -> list[dict]:
        sql = ("SELECT a.id, a.action, a.target, a.detail, a.ip, a.created_at, a.hash, u.username, u.role "
               "FROM audit_log a LEFT JOIN users u USING(user_id) WHERE 1=1")
        params: list = []
        if action:
            sql += " AND a.action=?"
            params.append(action)
        if user:
            sql += " AND u.username=?"
            params.append(user)
        if target:
            sql += " AND a.target LIKE ?"
            params.append(f"{target}%")
        if before_id:
            sql += " AND a.id<?"
            params.append(before_id)
        rows = self._q(sql + " ORDER BY a.id DESC LIMIT ?", (*params, limit))
        return [{**dict(r), "detail": json.loads(r["detail"]) if r["detail"] else None} for r in rows]

    def audit_actions(self) -> list[str]:
        return [r[0] for r in self._q("SELECT DISTINCT action FROM audit_log ORDER BY action")]

    def audit_trail(self, target: str) -> list[dict]:
        rows = self._q("SELECT a.action, a.detail, a.created_at, u.username FROM audit_log a "
                       "LEFT JOIN users u USING(user_id) WHERE a.target=? ORDER BY a.id", (target,))
        return [dict(r) for r in rows]
