# Demo App

A tiny example repository that GitContext indexes.

## Authentication

Login is handled by `AuthService.authenticate_user` in `auth.py`. Passwords
are hashed with SHA-256 and a session token is issued on success.

## Calculator

The `calculator.py` module provides `add`, `divide`, and a stateful
`Calculator` class. Dividing by zero raises an error.
