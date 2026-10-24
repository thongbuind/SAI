"""Phát hành snapshot đã verify: lập kế hoạch, commit có kiểm tra cha, gắn tag và đối chiếu Hub.

Kế hoạch và tiến độ chỉ lưu local. Commit upload và tag là hai thao tác riêng;
nếu gắn tag lỗi, lần chạy sau tiếp tục từ commit đã ghi nhận, không upload lại.
"""

import hashlib
from pathlib import Path
from hf_release.artifact import (
    LOCAL_ONLY_FILES, MANIFEST_FILE, check_verified, read_json, sha256_file, write_json,
)

HUB_MANAGED_FILES = {".gitattributes"}

def plan_file_changes(local_files, remote_files) -> tuple[list[str], list[str]]:
    upload = sorted(set(local_files))
    delete = sorted(set(remote_files) - set(local_files) - HUB_MANAGED_FILES)
    return upload, delete

def _git_blob_id(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _verify_remote(api, repo_id: str, commit: str, folder: Path, manifest: dict) -> None:
    """Đối chiếu metadata nội dung tại commit cố định, không tải lại trọng số lớn."""
    info = api.model_info(repo_id, revision=commit, files_metadata=True)
    if info.sha != commit:
        raise RuntimeError("Hub trả về commit khác revision cần kiểm chứng")
    files = {entry.rfilename: entry for entry in info.siblings if entry.rfilename not in HUB_MANAGED_FILES}
    expected = manifest["files"] | {MANIFEST_FILE: sha256_file(folder / MANIFEST_FILE)}
    if set(files) != set(expected):
        raise RuntimeError("Danh sách file trên Hub khác bộ phát hành đã verify")
    for name, expected_hash in expected.items():
        entry = files[name]
        local = folder / name
        if entry.size != local.stat().st_size:
            raise RuntimeError(f"Kích thước file trên Hub không khớp: {name}")
        matches = entry.lfs.sha256 == expected_hash if entry.lfs is not None else entry.blob_id == _git_blob_id(local)
        if not matches:
            raise RuntimeError(f"Nội dung file trên Hub không khớp: {name}")

def _tag_commit(api, repo_id: str, version: str) -> str | None:
    if not any(tag.name == version for tag in api.list_repo_refs(repo_id).tags):
        return None
    # Annotated tag có object SHA riêng; resolve revision để lấy commit chứa nội dung.
    return api.model_info(repo_id, revision=version).sha

def push_release(export_dir, *, confirm: bool = False, create_repo: bool = False, api=None) -> dict:
    """Xem trước hoặc phát hành. Client có thể truyền vào để test không cần mạng/token."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    folder = Path(export_dir)
    manifest = check_verified(folder)
    repo_id, version = manifest["repo_id"], manifest["version"]
    manifest_hash = sha256_file(folder / MANIFEST_FILE)
    if confirm and (manifest["git"]["dirty"] is not False or not manifest["git"]["commit"]):
        raise RuntimeError("Phát hành chính thức cần export từ Git commit sạch; hãy tự commit rồi export và verify lại")
    api = HfApi() if api is None else api
    account = api.whoami()["name"]
    exists = api.repo_exists(repo_id)
    if not exists and not create_repo:
        raise RuntimeError(f"Repo {repo_id} chưa tồn tại; cần --create-repo")
    tagged = _tag_commit(api, repo_id, version) if exists else None
    if tagged is not None:
        # Tag chỉ được chấp nhận khi nó đã trỏ đúng toàn bộ nội dung cần phát hành.
        _verify_remote(api, repo_id, tagged, folder, manifest)
        record = {"repo_id": repo_id, "version": version, "manifest_sha256": manifest_hash,
                  "hub_commit": tagged, "account": account, "status": "complete"}
        if confirm:
            write_json(folder / "push.json", record)
        print(f"Đã có bản phát hành khớp hoàn toàn: {repo_id}@{version} ({tagged})")
        return record

    progress_path = folder / "push.json"
    progress = read_json(progress_path) if progress_path.exists() else None
    if progress is not None:
        identity = (progress.get("repo_id"), progress.get("version"), progress.get("manifest_sha256"))
        if identity != (repo_id, version, manifest_hash):
            raise RuntimeError("Tiến độ push không thuộc bộ phát hành hiện tại")
        _verify_remote(api, repo_id, progress["hub_commit"], folder, manifest)
        print(f"Tiếp tục gắn tag cho commit {progress['hub_commit']}; không upload lại")
        if not confirm:
            return progress
        commit_sha = progress["hub_commit"]
    else:
        base_commit = api.model_info(repo_id).sha if exists else None
        remote_files = api.list_repo_files(repo_id, revision=base_commit) if exists else []
        upload, delete = plan_file_changes([*manifest["files"], MANIFEST_FILE], remote_files)
        plan = {"repo_id": repo_id, "version": version, "manifest_sha256": manifest_hash,
                "account": account, "base_commit": base_commit, "create_repo": not exists,
                "upload": upload, "delete": delete}
        print(f"Tài khoản: {account}; repo: {repo_id}; tag: {version}")
        print(f"Nguồn Git: {manifest['git']}; commit cha trên Hub: {base_commit}")
        print(f"Upload {len(upload)} file: {', '.join(upload)}")
        print(f"Xoá {len(delete)} file: {', '.join(delete) if delete else '(không có)'}")
        plan_path = folder / "push_plan.json"
        if not confirm:
            write_json(plan_path, plan)
            print("Đã lưu kế hoạch; chưa thay đổi Hub. Duyệt kế hoạch rồi chạy --confirm.")
            return plan
        if not plan_path.exists():
            raise RuntimeError("Chưa có kế hoạch được xem trước; chạy push không có --confirm trước")
        approved_plan = read_json(plan_path)
        if plan != approved_plan:
            raise RuntimeError("Repo hoặc kế hoạch đã thay đổi; cần xem và duyệt kế hoạch mới")
        if input(f"Gõ '{repo_id}' để xác nhận kế hoạch upload/xoá: ").strip() != repo_id:
            raise RuntimeError("Xác nhận không khớp; chưa thay đổi Hub")
        # Kiểm tra lại sau khoảng chờ xác nhận, trước khi bắt đầu thao tác ghi.
        check_verified(folder)
        if not exists:
            api.create_repo(repo_id, repo_type="model")
            base_commit = api.model_info(repo_id).sha
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name)) for name in upload]
        operations += [CommitOperationDelete(path_in_repo=name) for name in delete]
        commit = api.create_commit(
            repo_id=repo_id, operations=operations, revision="main", parent_commit=base_commit,
            commit_message=f"Release {version}",
            commit_description=f"Nguồn Git {manifest['git']['commit']}; manifest SHA-256 {manifest_hash}",
        )
        commit_sha = commit.oid
        progress = {"repo_id": repo_id, "version": version, "manifest_sha256": manifest_hash,
                    "hub_commit": commit_sha, "account": account, "status": "uploaded"}
        write_json(progress_path, progress)

    _verify_remote(api, repo_id, commit_sha, folder, manifest)
    api.create_tag(repo_id, tag=version, revision=commit_sha, tag_message=f"Release {version}")
    if _tag_commit(api, repo_id, version) != commit_sha:
        raise RuntimeError("Tag trên Hub không trỏ tới commit đã kiểm chứng")
    progress["status"] = "complete"
    write_json(progress_path, progress)
    print(f"Đã phát hành và kiểm tra: {repo_id}@{version} ({commit_sha})")
    return progress
