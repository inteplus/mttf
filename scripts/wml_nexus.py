#!/usr/bin/env python3

import asyncio
import sys
import subprocess
import sshtunnel

from mt import net, logg


async def main():
    argv = sys.argv
    logg.logger.setLevel(logg.INFO)

    if len(argv) < 2:
        print("Opens localhost:5443 as nexus https and runs a command.")
        print("Syntax: {} cmd arg1 arg2 ...".format(argv[0]))
        sys.exit(0)

    if net.is_port_open("localhost", 5443, timeout=0.1):
        raise RuntimeError("Local port 5443 already in use.")

    if net.is_port_open("nexus.winnow.tech", 443):
        server = await net.port_forwarder_actx(
            ":5443", ["nexus.winnow.tech:443"], logger=logg.logger
        )
        async with server:
            process = await asyncio.create_subprocess_exec(*argv[1:])
            returncode = await process.wait()
            sys.exit(returncode)

    with sshtunnel.open_tunnel(
        ("clujdc.edge.winnowsolutions.com", 22222),
        ssh_username="nexus",
        ssh_password="nexusshared",
        remote_bind_address=("192.168.110.4", 443),
        local_bind_address=("0.0.0.0", 5443),
    ) as tun:
        res = subprocess.run(argv[1:], shell=False, check=False)
        sys.exit(res.returncode)


if __name__ == "__main__":
    asyncio.run(main())
