"""Fetch small pinned public assets that are not datasets loaded through Hugging Face."""
import hashlib
import urllib.request

from common.data import repo_path

ASSETS = {
    "data/first_plot_questions.yaml": (
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
        "e40daef5791139a2417e22034ab0a0e72900524b/evaluation/first_plot_questions.yaml",
        "215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173"),
}


def main():
    for relative, (url, digest) in ASSETS.items():
        path = repo_path(relative)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{relative} hash {actual} differs from pinned {digest}")
        print("asset ok", relative)


if __name__ == "__main__":
    main()
