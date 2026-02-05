# utils/crypto.py
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

def generate_keys():
    """
    Generate Ed25519 key pair
    :return: (private_key, public_key)
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def sign_update(private_key, data_bytes):
    """
    Sign data using private key
    :param private_key: Ed25519PrivateKey object
    :param data_bytes: bytes to sign
    :return: signature bytes
    """
    if private_key is None:
        # PoC: return dummy signature
        return b"dummy_signature"
    return private_key.sign(data_bytes)

def verify_signature(pubkey, data_bytes, signature):
    """
    Verify signature
    :param pubkey: Ed25519PublicKey object or None for PoC
    :param data_bytes: signed bytes
    :param signature: signature bytes
    :return: True/False
    """
    if pubkey is None:
        # PoC: always return True
        return True
    try:
        pubkey.verify(signature, data_bytes)
        return True
    except Exception:
        return False

def serialize_public_key(pubkey):
    """
    Convert public key to bytes
    """
    return pubkey.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

def load_public_key(pub_bytes):
    """
    Load public key from bytes
    """
    return Ed25519PublicKey.from_public_bytes(pub_bytes)
