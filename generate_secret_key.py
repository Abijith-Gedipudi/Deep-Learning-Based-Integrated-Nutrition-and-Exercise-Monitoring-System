#!/usr/bin/env python3
"""
Quick script to generate a secure SECRET_KEY for Flask
Run this and copy the output to use as your SECRET_KEY environment variable
"""
import secrets

if __name__ == '__main__':
    secret_key = secrets.token_hex(32)
    print(f"Generated SECRET_KEY: {secret_key}")
    print("\nCopy this value and use it as your SECRET_KEY environment variable in Render.")

