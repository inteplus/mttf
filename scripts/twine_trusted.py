#!/usr/bin/python3

import twine.repository
import twine.__main__


def disable_server_certificate_validation():
    """Allow twine to just trust the hosts"""
    twine.repository.Repository.set_certificate_authority = lambda *args, **kwargs: None


def main():
    disable_server_certificate_validation()
    twine.__main__.main()


__name__ == "__main__" and main()
