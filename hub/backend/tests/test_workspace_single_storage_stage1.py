"""
Stage 1 of internal design notes — path-based Files API.

These tests pin the new `/rooms/{id}/workspace/fs/*` router, which is a thin
wrapper over `workspaced_client` and never touches the `repofile` table:

  * tree (snapshot, prefix filter)
  * read (text / binary data-URI / 404)
  * write (auto-commit, FILE_CHANGED event carries `path`, no `file_id`)
  * delete (ok / 404)
  * upload (binary → data-URI convention, structure preserved)
  * download (binary payload decoded to real bytes + Content-Disposition)
  * traversal (`..`) rejected at the boundary
  * membership enforced

The old `/rooms/{id}/files*` router stays untouched (coexistence until
stage 5).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.v1 import workspace_files as wf


def _db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


def _user():
    return MagicMock(id=uuid4())


def _room():
    # Plain mock, not AsyncMock: the endpoints read ``git_config`` and
    # ``creator_id`` synchronously (private-files policy, storage reservation).
    return MagicMock(git_config={}, creator_id=None)


# ── tree ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tree_returns_snapshot_files_filtered_by_prefix():
    room_id = uuid4()
    snapshot = {
        "generation": {"epoch": "e", "revision": 3},
        "files": [
            {"path": "/src/a.py", "size": 10, "mtime": 1.0, "binary": False, "fingerprint": "abc"},
            {"path": "/src/sub/b.py", "size": 20, "mtime": 2.0, "binary": False, "fingerprint": "def"},
            {"path": "/readme.md", "size": 5, "mtime": 3.0, "binary": False, "fingerprint": "ghi"},
        ],
    }

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "get_workspace_snapshot", new_callable=AsyncMock) as ms:
        ms.return_value = snapshot
        result = await wf.workspace_fs_tree(
            room_id=room_id, prefix="src", db=_db(), current_user=_user()
        )

    ms.assert_awaited_once_with(str(room_id))
    paths = [f.path for f in result.files]
    assert paths == ["/src/a.py", "/src/sub/b.py"]
    assert result.generation == {"epoch": "e", "revision": 3}


@pytest.mark.asyncio
async def test_tree_empty_prefix_returns_all_files():
    room_id = uuid4()
    snapshot = {
        "generation": None,
        "files": [
            {"path": "/a.py", "size": 1, "mtime": 0.0, "binary": False},
            {"path": "/b.py", "size": 2, "mtime": 0.0, "binary": True, "fingerprint": "x"},
        ],
    }

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "get_workspace_snapshot", new_callable=AsyncMock) as ms:
        ms.return_value = snapshot
        result = await wf.workspace_fs_tree(
            room_id=room_id, prefix="", db=_db(), current_user=_user()
        )

    assert [f.path for f in result.files] == ["/a.py", "/b.py"]
    assert result.files[1].binary is True
    assert result.files[1].fingerprint == "x"


# ── read ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_text():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = "hello world"
        result = await wf.workspace_fs_read_file(
            room_id=room_id, path="/src/a.py", db=_db(), current_user=_user()
        )

    assert result.path == "src/a.py"
    assert result.content == "hello world"
    assert result.binary is False
    mr.assert_awaited_once_with(str(room_id), "src/a.py")


@pytest.mark.asyncio
async def test_read_file_binary_data_uri():
    room_id = uuid4()
    data_uri = "data:image/png;base64,iVBORw0KGgo="
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = data_uri
        result = await wf.workspace_fs_read_file(
            room_id=room_id, path="/img.png", db=_db(), current_user=_user()
        )

    assert result.content == data_uri
    assert result.binary is True


@pytest.mark.asyncio
async def test_read_file_404_when_missing():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = None
        with pytest.raises(HTTPException) as exc:
            await wf.workspace_fs_read_file(
                room_id=room_id, path="/nope.txt", db=_db(), current_user=_user()
            )

    assert exc.value.status_code == 404


# ── write ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_autocommits_and_emits_path_only_event():
    room_id = uuid4()
    db = _db()
    write_result = {"path": "src/a.py", "size": 11, "commit_sha": "deadbeef",
                    "sync_ignored": False, "sync_ignore_source": None}

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "write_workspace_file", new_callable=AsyncMock) as mw, \
         patch.object(wf, "emit_event", new_callable=AsyncMock) as me:
        mw.return_value = write_result
        out = await wf.workspace_fs_write_file(
            room_id=room_id,
            path="/src/a.py",
            body=wf.FileWriteIn(content="hello world"),
            db=db,
            current_user=_user(),
        )

    mw.assert_awaited_once()
    args, kwargs = mw.call_args
    assert kwargs["commit_message"] == "user: write src/a.py"
    assert out.commit_sha == "deadbeef"
    assert out.path == "src/a.py"

    me.assert_awaited_once()
    meta = me.call_args.kwargs["metadata"]
    assert meta["path"] == "src/a.py"
    assert meta["commit_sha"] == "deadbeef"
    # The new API addresses by path; no file_id must leak through.
    assert "file_id" not in meta


# ── delete ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_file_ok():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "delete_workspace_file", new_callable=AsyncMock) as md, \
         patch.object(wf, "emit_event", new_callable=AsyncMock):
        md.return_value = True
        out = await wf.workspace_fs_delete_file(
            room_id=room_id, path="/src/a.py", db=_db(), current_user=_user()
        )

    assert out.ok is True
    assert out.path == "src/a.py"
    assert md.call_args.kwargs["commit_message"] == "user: delete src/a.py"


@pytest.mark.asyncio
async def test_delete_file_404_when_missing():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "delete_workspace_file", new_callable=AsyncMock) as md, \
         patch.object(wf, "emit_event", new_callable=AsyncMock):
        md.return_value = False
        with pytest.raises(HTTPException) as exc:
            await wf.workspace_fs_delete_file(
                room_id=room_id, path="/nope.txt", db=_db(), current_user=_user()
            )

    assert exc.value.status_code == 404


# ── upload ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_binary_wraps_in_data_uri_and_preserves_structure():
    room_id = uuid4()
    raw = b"\x89PNG\r\n\x1a\n\x00\x00"  # non-UTF8 binary
    upload = MagicMock()
    upload.filename = "logo.png"
    upload.content_type = "image/png"
    upload.read = AsyncMock(side_effect=[raw, b""])

    write_result = {"path": "assets/img/logo.png", "size": len(raw),
                    "commit_sha": "cafe", "sync_ignored": False, "sync_ignore_source": None}

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "write_workspace_file", new_callable=AsyncMock) as mw, \
         patch.object(wf, "emit_event", new_callable=AsyncMock):
        mw.return_value = write_result
        out = await wf.workspace_fs_upload(
            room_id=room_id,
            file=upload,
            target_path="/assets",
            relative_path="img/logo.png",
            commit_message=None,
            db=_db(),
            current_user=_user(),
        )

    out_path, out_content = mw.call_args.args[1], mw.call_args.args[2]
    assert out_path == "assets/img/logo.png"
    assert out_content.startswith("data:image/png;base64,")
    assert out.commit_sha == "cafe"
    assert mw.call_args.kwargs["commit_message"] == "user: upload assets/img/logo.png"


@pytest.mark.asyncio
async def test_upload_text_uses_filename_when_relative_path_empty():
    room_id = uuid4()
    upload = MagicMock()
    upload.filename = "notes.md"
    upload.content_type = "text/markdown"
    upload.read = AsyncMock(side_effect=[b"# hi", b""])

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "write_workspace_file", new_callable=AsyncMock) as mw, \
         patch.object(wf, "emit_event", new_callable=AsyncMock):
        mw.return_value = {"path": "notes.md", "size": 4, "commit_sha": None,
                           "sync_ignored": False, "sync_ignore_source": None}
        await wf.workspace_fs_upload(
            room_id=room_id,
            file=upload,
            target_path="/",
            relative_path="",
            commit_message=None,
            db=_db(),
            current_user=_user(),
        )

    assert mw.call_args.args[1] == "notes.md"
    assert mw.call_args.args[2] == "# hi"


# ── download ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_download_binary_decodes_data_uri_to_bytes():
    import base64 as b64

    room_id = uuid4()
    payload = b"\x89PNG\r\n\x1a\n\x00\x00"
    data_uri = "data:image/png;base64," + b64.b64encode(payload).decode()

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = data_uri
        response = await wf.workspace_fs_download(
            room_id=room_id, path="/img/logo.png", db=_db(), current_user=_user()
        )

    assert response.body == payload
    assert response.media_type == "image/png"
    assert 'filename="logo.png"' in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_download_text_returns_utf8_bytes():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = "héllo"
        response = await wf.workspace_fs_download(
            room_id=room_id, path="/readme.md", db=_db(), current_user=_user()
        )

    assert response.body == "héllo".encode("utf-8")
    assert response.media_type == "text/plain; charset=utf-8"
    assert 'filename="readme.md"' in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_download_404_when_missing():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        mr.return_value = None
        with pytest.raises(HTTPException) as exc:
            await wf.workspace_fs_download(
                room_id=room_id, path="/nope.bin", db=_db(), current_user=_user()
            )

    assert exc.value.status_code == 404


# ── traversal & membership ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_traversal_path_rejected_with_400():
    room_id = uuid4()
    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock) as mr:
        with pytest.raises(HTTPException) as exc:
            await wf.workspace_fs_read_file(
                room_id=room_id, path="/../etc/passwd", db=_db(), current_user=_user()
            )

    assert exc.value.status_code == 400
    mr.assert_not_awaited()


@pytest.mark.asyncio
async def test_membership_denied_returns_403():
    room_id = uuid4()

    async def deny(*_a, **_kw):
        raise HTTPException(status_code=403, detail="Not a room member")

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new=deny), \
         patch.object(wf.wsd, "get_workspace_snapshot", new_callable=AsyncMock) as ms:
        with pytest.raises(HTTPException) as exc:
            await wf.workspace_fs_tree(
                room_id=room_id, prefix="", db=_db(), current_user=_user()
            )

    assert exc.value.status_code == 403
    ms.assert_not_awaited()


# ── workspaced error mapping ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_workspaced_404_surfaces_as_404_not_502():
    room_id = uuid4()
    request = httpx.Request("GET", "http://x/file")
    response = httpx.Response(404, text="not found", request=request)
    exc = httpx.HTTPStatusError("404", request=request, response=response)

    with patch.object(wf, "_get_room", AsyncMock(return_value=_room())), \
         patch.object(wf, "_verify_membership", new_callable=AsyncMock), \
         patch.object(wf.wsd, "read_workspace_file", new_callable=AsyncMock, side_effect=exc):
        with pytest.raises(HTTPException) as raised:
            await wf.workspace_fs_read_file(
                room_id=room_id, path="/x", db=_db(), current_user=_user()
            )

    assert raised.value.status_code == 404
