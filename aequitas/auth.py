"""Police sign-in: accounts, sessions, lockout and the audit trail.

- Passwords: salted scrypt (stdlib). Accounts are created only with `aequitas create-user`, which prompts
  for the password, so it never appears in shell history or config files.
- Sessions: a random token in an HttpOnly cookie; only its sha256 is stored.
- Lockout: too many failed logins for a username from one address blocks it for a while.
- Audit: sign-ins and report actions are recorded (who, what, when, from where)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time

ROLES = ("admin", "officer")
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


class LockedOut(Exception):
    pass


class Auth:
    def __init__(self, store, session_hours: float = 12, max_failures: int = 5, lockout_s: float = 900):
        self.store = store
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

    def audit(self, user_id: int | None, action: str, target: str | None = None, detail=None, ip: str = "") -> None:
        self._q("INSERT INTO audit_log(user_id, action, target, detail, ip, created_at) VALUES (?,?,?,?,?,?)",
                (user_id, action, target, json.dumps(detail) if detail is not None else None, ip,
                 int(time.time() * 1000)), commit=True)

    def audit_trail(self, target: str) -> list[dict]:
        rows = self._q("SELECT a.action, a.detail, a.created_at, u.username FROM audit_log a "
                       "LEFT JOIN users u USING(user_id) WHERE a.target=? ORDER BY a.id", (target,))
        return [dict(r) for r in rows]
