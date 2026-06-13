from experiments.download_hf_file import remote_size


class Response:
    def __init__(self, headers):
        self.headers = headers


def test_remote_size_prefers_linked_size():
    response = Response(
        {"X-Linked-Size": "123", "Content-Length": "10"}
    )
    assert remote_size(response) == 123


def test_remote_size_parses_content_range():
    response = Response({"Content-Range": "bytes 50-99/200"})
    assert remote_size(response) == 200
