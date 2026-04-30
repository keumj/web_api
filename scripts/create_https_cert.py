from __future__ import annotations

import argparse
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError as exc:  # pragma: no cover - helper script only.
    raise SystemExit("cryptography is required. Run setup_service.cmd, then retry.") from exc


def _local_ipv4_addresses() -> list[str]:
    addresses: set[str] = {"127.0.0.1"}
    hostname = socket.gethostname()
    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = str(info[4][0])
            if not ip.startswith("169.254."):
                addresses.add(ip)
    except OSError:
        pass
    return sorted(addresses)


def create_cert(certfile: Path, keyfile: Path, days: int) -> None:
    certfile.parent.mkdir(parents=True, exist_ok=True)
    keyfile.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    hostname = socket.gethostname()
    dns_names = {"localhost", hostname}
    ip_addresses = [ipaddress.ip_address(item) for item in _local_ipv4_addresses()]

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Keumj Portfolio Lab LAN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Keumj Local"),
        ]
    )
    alt_names = [x509.DNSName(name) for name in sorted(dns_names)] + [x509.IPAddress(ip) for ip in ip_addresses]
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=max(int(days), 1)))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    keyfile.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Wrote certificate: {certfile}")
    print(f"Wrote private key: {keyfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local self-signed HTTPS certificate.")
    parser.add_argument("--certfile", default="certs/keumjm-lan.crt")
    parser.add_argument("--keyfile", default="certs/keumjm-lan.key")
    parser.add_argument("--days", type=int, default=825)
    args = parser.parse_args()
    create_cert(Path(args.certfile), Path(args.keyfile), args.days)


if __name__ == "__main__":
    main()
