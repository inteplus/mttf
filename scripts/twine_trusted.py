#!/usr/bin/python3

import twine.repository
import twine.__main__
from twine import package as package_file
import requests
import requests_toolbelt
import rich.progress


def _upload(self, package: package_file.PackageFile) -> requests.Response:
    print(f"Uploading {package.basefilename}")

    metadata = package.metadata_dictionary()
    data_to_send = self._convert_metadata_to_list_of_tuples(metadata)
    data_to_send.append((":action", "file_upload"))
    data_to_send.append(("protocol_version", "1"))
    with open(package.filename, "rb") as fp:
        data_to_send.append(
            (
                "content",
                (package.basefilename, fp, "application/octet-stream"),
            )
        )
        encoder = requests_toolbelt.MultipartEncoder(data_to_send)

        with rich.progress.Progress(
            "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.BarColumn(),
            rich.progress.DownloadColumn(),
            "•",
            rich.progress.TimeRemainingColumn(
                compact=True,
                elapsed_when_finished=True,
            ),
            "•",
            rich.progress.TransferSpeedColumn(),
            disable=self.disable_progress_bar,
        ) as progress:
            task_id = progress.add_task("", total=encoder.len)

            monitor = requests_toolbelt.MultipartEncoderMonitor(
                encoder,
                lambda monitor: progress.update(
                    task_id,
                    completed=monitor.bytes_read,
                ),
            )

            resp = self.session.post(
                self.url,
                data=monitor,
                allow_redirects=False,
                headers={"Content-Type": monitor.content_type},
                verify=False,  # MT-NOTE: I added this line only
            )

    return resp


def disable_server_certificate_validation():
    """Allow twine to just trust the hosts"""
    twine.repository.Repository.set_certificate_authority = lambda *args, **kwargs: None
    twine.repository.Repository._upload = _upload


def main():
    disable_server_certificate_validation()
    twine.__main__.main()


__name__ == "__main__" and main()
