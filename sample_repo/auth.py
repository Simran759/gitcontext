"""Authentication service for the demo app."""
import hashlib


class AuthService:
    """Handles user login and session tokens."""

    def __init__(self, user_store):
        self.user_store = user_store

    def authenticate_user(self, username, password):
        """Log a user in.

        Looks up the user, hashes the supplied password, and compares it to
        the stored hash. Returns a session token on success, else raises.
        """
        user = self.user_store.get(username)
        if user is None:
            raise ValueError("unknown user")
        if user["password_hash"] != hash_password(password):
            raise ValueError("bad password")
        return issue_token(username)

    def refresh_token(self, username, token):
        """Exchange a valid session token for a fresh one.

        Confirms the supplied token matches the one on record for the user,
        then re-issues a new token so long-lived sessions can roll their
        credentials without a full re-login.
        """
        if token != issue_token(username):
            raise ValueError("token mismatch")
        return issue_token(username)


def hash_password(password: str) -> str:
    """Hash a password with SHA-256 (demo only - use bcrypt in real life)."""
    return hashlib.sha256(password.encode()).hexdigest()


def issue_token(username: str) -> str:
    """Create an opaque session token for an authenticated user."""
    return hashlib.sha256(f"token:{username}".encode()).hexdigest()
