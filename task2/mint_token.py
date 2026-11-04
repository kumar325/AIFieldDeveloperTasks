"""Create a test bearer token. Usage: python mint_token.py admin --sub alice"""

import argparse
import os
import sys
import time

import jwt

parser = argparse.ArgumentParser()
parser.add_argument("role", choices=["admin", "viewer"])
parser.add_argument("--sub", default="dev-user")
parser.add_argument("--ttl", type=int, default=3600, help="seconds until expiry")
args = parser.parse_args()

secret = os.environ.get("GATEWAY_JWT_SECRET", "")
if len(secret) < 32:
    sys.exit("GATEWAY_JWT_SECRET must be set to at least 32 characters")

now = int(time.time())
print(jwt.encode({"sub": args.sub, "role": args.role, "iat": now, "exp": now + args.ttl}, secret, algorithm="HS256"))
