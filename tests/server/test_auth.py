"""Access control of the server: the token, the sign-in redirect, ``Host`` and ``Origin``."""

from __future__ import annotations

import re
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from monorl.core.errors import ConfigError
from monorl.server import auth
from monorl.server.app import ServerSettings, create_app
from monorl.server.jobs import JobManager

from .conftest import BASE_URL, TOKEN, signed_in

COOKIE = "monorl_session_8000"  # the port of BASE_URL


@pytest.fixture
def anonymous(settings: ServerSettings, jobs: JobManager) -> Iterator[TestClient]:
    """A client that addresses the server correctly but has not signed in."""
    with TestClient(create_app(settings, jobs=jobs), base_url=BASE_URL) as c:
        yield c


def _sign_in(client: TestClient) -> None:
    assert client.get("/", params={"token": TOKEN}, follow_redirects=False).status_code == 303


# -- token -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token-0123456789"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Cookie": f"{COOKIE}=wrong-token-0123456789"},
        {"Cookie": f"monorl_session_9000={TOKEN}"},  # the cookie of a server on another port
    ],
    ids=["none", "wrong-bearer", "not-bearer", "longer-bearer", "wrong-cookie", "other-port-cookie"],
)
def test_api_refuses_requests_without_the_token(
    anonymous: TestClient, jobs: JobManager, config_dict: dict[str, Any], headers: dict[str, str]
) -> None:
    for response in (
        anonymous.get("/api/system", headers=headers),
        anonymous.get("/api/openapi.json", headers=headers),
        anonymous.get("/api/datasets/browse", params={"uri": "/"}, headers=headers),
        anonymous.get("/api/does-not-exist", headers=headers),
        anonymous.post("/api/jobs", json={"config": config_dict}, headers=headers),
    ):
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert "?token=" in detail and "server_token" in detail  # how to sign in
        assert TOKEN not in response.text
    assert jobs.list() == []


def test_bearer_token_is_accepted(anonymous: TestClient) -> None:
    r = anonymous.get("/api/system", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    assert "python" in r.json()


def test_sign_in_url_sets_the_cookie_and_drops_the_token_from_the_url(anonymous: TestClient) -> None:
    r = anonymous.get("/runs/abc", params={"token": TOKEN, "tab": "logs"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/runs/abc?tab=logs"
    name_value, *attributes = (part.strip() for part in r.headers["set-cookie"].split(";"))
    assert name_value == f"{COOKIE}={TOKEN}"
    assert {a.lower() for a in attributes} == {"httponly", "samesite=strict", "path=/"}
    # The cookie alone signs the browser in from now on (the client keeps it).
    assert anonymous.get("/api/jobs").status_code == 200


def test_sign_in_redirect_stays_on_this_server(anonymous: TestClient) -> None:
    r = anonymous.get(f"{BASE_URL}//evil.example/x", params={"token": TOKEN}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/evil.example/x"


def test_sign_in_with_a_wrong_token_is_refused(anonymous: TestClient) -> None:
    r = anonymous.get("/", params={"token": "wrong-token-0123456789"}, follow_redirects=False)
    assert r.status_code == 401
    assert "set-cookie" not in r.headers
    assert "not valid" in r.json()["detail"]
    assert anonymous.get("/api/jobs").status_code == 401


def test_ui_files_load_without_the_token_and_the_api_does_not(
    settings: ServerSettings, tmp_path: Path
) -> None:
    dist = tmp_path / "web-dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>monorl ui</html>")
    with TestClient(create_app(settings), base_url=BASE_URL) as c:
        assert c.get("/").text == "<html>monorl ui</html>"
        assert c.get("/runs/abc").text == "<html>monorl ui</html>"
        assert c.get("/api/config/schema").status_code == 401
        assert c.post("/").status_code == 401


def test_swagger_and_redoc_pages_are_not_served(client: TestClient) -> None:
    assert client.get("/api/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/api/openapi.json").json()["info"]["title"] == "monorl"


# -- Host --------------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example:8000", "evil.example", "127.0.0.1.evil.example", ""])
def test_foreign_host_is_refused_even_with_the_token(
    client: TestClient, jobs: JobManager, config_dict: dict[str, Any], host: str
) -> None:
    # `client` sends the bearer token; only the Host is wrong.
    r = client.get("/api/system", headers={"Host": host})
    assert r.status_code == 403
    assert "--allowed-host" in r.json()["detail"] and "MONORL_ALLOWED_HOSTS" in r.json()["detail"]
    r = client.post("/api/jobs", json={"config": config_dict}, headers={"Host": host})
    assert r.status_code == 403
    assert jobs.list() == []


def test_rebinding_request_with_a_valid_cookie_is_refused(
    anonymous: TestClient, jobs: JobManager, config_dict: dict[str, Any]
) -> None:
    """A page that rebinds its name to 127.0.0.1 is same-origin with the server as far as the
    browser is concerned; whatever cookie it carries, its Host gives it away."""
    foreign = {"Host": "evil.example:8000", "Cookie": f"{COOKIE}={TOKEN}"}
    assert anonymous.get("/api/jobs", headers=foreign).status_code == 403
    r = anonymous.post(
        "/api/jobs",
        json={"config": config_dict},
        headers={**foreign, "Origin": "http://evil.example:8000"},
    )
    assert r.status_code == 403
    assert anonymous.get("/", params={"token": TOKEN}, headers=foreign).status_code == 403
    assert jobs.list() == []


@pytest.mark.parametrize("host", ["localhost:8000", "LOCALHOST", "127.0.0.1:5173", "[::1]:8000"])
def test_local_names_are_accepted_on_any_port(client: TestClient, host: str) -> None:
    assert client.get("/api/jobs", headers={"Host": host}).status_code == 200


def test_allowed_host_is_accepted(settings: ServerSettings) -> None:
    settings.allowed_hosts = ["monorl.lan", "fd00::7"]
    with signed_in(create_app(settings)) as c:
        assert c.get("/api/jobs", headers={"Host": "monorl.lan:8000"}).status_code == 200
        assert c.get("/api/jobs", headers={"Host": "[fd00::7]:8000"}).status_code == 200
        assert c.get("/api/jobs", headers={"Host": "other.lan:8000"}).status_code == 403


# -- Origin ------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["http://evil.example", "http://localhost:3000", "null"])
def test_foreign_origin_on_post_is_refused(
    anonymous: TestClient, jobs: JobManager, config_dict: dict[str, Any], origin: str
) -> None:
    _sign_in(anonymous)
    r = anonymous.post("/api/jobs", json={"config": config_dict}, headers={"Origin": origin})
    assert r.status_code == 403
    assert "origin" in r.json()["detail"]
    assert jobs.list() == []


def test_own_origin_on_post_is_accepted(anonymous: TestClient, config_dict: dict[str, Any]) -> None:
    _sign_in(anonymous)
    r = anonymous.post("/api/config/validate", json={"config": config_dict}, headers={"Origin": BASE_URL})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_configured_cors_origin_may_call_the_api(
    settings: ServerSettings, config_dict: dict[str, Any]
) -> None:
    settings.cors_origins = ["http://localhost:5173"]
    with signed_in(create_app(settings)) as c:
        preflight = c.options(
            "/api/config/validate",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
        r = c.post(
            "/api/config/validate",
            json={"config": config_dict},
            headers={"Origin": "http://localhost:5173"},
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_no_origin_is_allowed_by_default(client: TestClient) -> None:
    r = client.options(
        "/api/config/schema",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


# -- token file and environment -----------------------------------------------------------------


def test_token_file_is_private_and_read_by_clients(settings: ServerSettings, home: Path) -> None:
    home.mkdir(parents=True)
    stale = home / "server_token"
    stale.write_text("left-by-an-older-server\n")
    stale.chmod(0o644)
    create_app(settings)
    assert stat.S_IMODE(stale.stat().st_mode) == 0o600
    assert auth.read_token(home) == TOKEN
    assert auth.read_token() == TOKEN  # MONORL_HOME, as a client on the same machine finds it


def test_read_token_says_how_to_get_one(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="monorl serve"):
        auth.read_token(tmp_path)


def test_weak_token_is_refused_without_echoing_it() -> None:
    with pytest.raises(ConfigError) as e:
        auth.resolve_token("hunter2")
    assert "16 characters" in str(e.value)
    assert "hunter2" not in str(e.value)


def test_jobs_do_not_inherit_the_token(
    client: TestClient,
    config_dict: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_for,  # noqa: ANN001
) -> None:
    """The commands a config names run in the job's environment; the API token is not in it."""
    out = tmp_path / "job-env.txt"
    script = tmp_path / "print_env.py"
    script.write_text(
        f"import os, sys\nopen({str(out)!r}, 'w').write(os.environ.get('MONORL_SERVER_TOKEN', '<unset>'))\n"
    )
    monkeypatch.setenv("MONORL_SERVER_TOKEN", TOKEN)
    monkeypatch.setenv("MONORL_TRAIN_CMD", f"{sys.executable} {script}")
    assert client.post("/api/jobs", json={"config": config_dict}).status_code == 201
    wait_for(out.exists)
    wait_for(lambda: out.read_text())
    assert out.read_text() == "<unset>"


# -- monorl serve ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "url_base", "accepted_host"),
    [
        ([], "http://127.0.0.1:8123/", "127.0.0.1:8123"),
        (["--host", "0.0.0.0"], "http://127.0.0.1:8123/", "localhost:8123"),  # noqa: S104
        (["--host", "192.0.2.7"], "http://192.0.2.7:8123/", "192.0.2.7:8123"),
        (["--allowed-host", "monorl.lan"], "http://127.0.0.1:8123/", "monorl.lan:8123"),
    ],
    ids=["default", "wildcard", "bound-address", "allowed-host"],
)
def test_serve_prints_a_url_that_signs_in(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    url_base: str,
    accepted_host: str,
) -> None:
    """The URL `monorl serve` prints carries the token of the app it then runs."""
    import uvicorn

    from monorl.cli.main import app as cli

    # Set, so that monkeypatch restores what `serve` leaves in the environment for its app.
    monkeypatch.setenv("MONORL_SERVER_TOKEN", "")
    monkeypatch.setenv("MONORL_ALLOWED_HOSTS", "")
    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kwargs: served.update(kwargs, target=target))

    result = CliRunner().invoke(cli, ["serve", "--port", "8123", "--no-open", *args])
    assert result.exit_code == 0, result.output
    match = re.search(r"^(http://\S+/)\?token=(\S+)$", result.output, re.MULTILINE)
    assert match, result.output
    assert match.group(1) == url_base
    token = match.group(2)

    assert served["target"] == "monorl.server.app:create_app" and served["factory"] is True
    with TestClient(create_app(), base_url=f"http://{accepted_host}") as c:
        assert c.get("/api/jobs").status_code == 401
        assert c.get("/", params={"token": token}, follow_redirects=False).status_code == 303
        assert c.get("/api/jobs").status_code == 200
        assert c.get("/api/jobs", headers={"Host": "evil.example:8123"}).status_code == 403
    assert (home / "server_token").read_text().strip() == token


def test_serve_takes_the_token_from_the_environment(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from monorl.cli.main import app as cli

    monkeypatch.setenv("MONORL_SERVER_TOKEN", TOKEN)
    monkeypatch.setenv("MONORL_ALLOWED_HOSTS", "")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    result = CliRunner().invoke(cli, ["serve", "--port", "8123", "--no-open"])
    assert result.exit_code == 0, result.output
    assert f"http://127.0.0.1:8123/?token={TOKEN}" in result.output.split()


def test_serve_refuses_a_weak_token(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from monorl.cli.main import app as cli

    monkeypatch.setenv("MONORL_SERVER_TOKEN", "")
    monkeypatch.setenv("MONORL_ALLOWED_HOSTS", "")
    started: list[Any] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(a))
    result = CliRunner().invoke(cli, ["serve", "--no-open", "--token", "hunter2"])
    assert result.exit_code == 1
    assert started == []
    assert "hunter2" not in result.output
