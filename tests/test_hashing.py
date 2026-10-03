from cctrain.hashing import file_size, sha256_bytes, sha256_file


def test_sha256_known_vector(tmp_path):
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    f = tmp_path / "a"
    f.write_bytes(b"abc")
    assert sha256_file(f) == sha256_bytes(b"abc")
    assert file_size(f) == 3
