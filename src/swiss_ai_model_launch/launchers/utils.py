import secrets
import string


def create_salt(length: int) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))
