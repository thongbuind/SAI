"""Push: đưa một thư mục export lên Hugging Face Hub thành MỘT commit và gắn tag phiên bản.

An toàn:
- Chỉ push khi ``verify.json`` đạt và khớp đúng ``model.safetensors`` hiện tại.
- Mặc định chỉ in kế hoạch (file thêm/xoá, tài khoản, repo, tag). Phải có ``--confirm`` VÀ gõ đúng
  repo_id mới thật sự push.
- Không ghi đè tag đã tồn tại. Không tự tạo repo trừ khi có ``--create-repo``.
- Dùng token của người dùng đã đăng nhập bằng ``hf auth login``; code không đọc hay nhận token.
"""

import json
from pathlib import Path
from foundation_lab.state.lineage import sha256_file

LOCAL_ONLY_FILES = {"release.json", "verify.json", "push.json"}
HUB_MANAGED_FILES = {".gitattributes"}

def plan_file_changes(local_files, remote_files) -> tuple[list[str], list[str]]:
    """(file sẽ upload, file trên Hub sẽ bị xoá vì không còn trong bản export)."""
    upload = sorted(set(local_files))
    delete = sorted(set(remote_files) - set(local_files) - HUB_MANAGED_FILES)
    return upload, delete

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _check_verified(export_dir: Path):
    verify_path = export_dir / "verify.json"
    if not verify_path.exists():
        raise RuntimeError(f"Chưa verify {export_dir}; chạy: python -m hf_release verify {export_dir}")
    report = _read_json(verify_path)
    if not report["passed"]:
        raise RuntimeError(f"Verify chưa đạt: {report['checks']}")
    if report["model_safetensors_sha256"] != sha256_file(export_dir / "model.safetensors"):
        raise RuntimeError("model.safetensors đã thay đổi sau khi verify; chạy verify lại")

def push_release(export_dir, *, confirm: bool = False, create_repo: bool = False):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    export_dir = Path(export_dir)
    manifest = _read_json(export_dir / "release.json")
    repo_id, version = manifest["repo_id"], manifest["version"]
    _check_verified(export_dir)

    api = HfApi()
    account = api.whoami()["name"]
    exists = api.repo_exists(repo_id)
    if not exists and not create_repo:
        raise RuntimeError(f"Repo {repo_id} chưa tồn tại; thêm --create-repo nếu muốn tạo mới")
    remote_files = api.list_repo_files(repo_id) if exists else []
    if exists and version in {tag.name for tag in api.list_repo_refs(repo_id).tags}:
        raise RuntimeError(f"Tag {version} đã có trên {repo_id}; tăng version trong release spec")

    local_files = sorted(
        p.relative_to(export_dir).as_posix() for p in export_dir.rglob("*")
        if p.is_file() and p.name not in LOCAL_ONLY_FILES
    )
    upload, delete = plan_file_changes(local_files, remote_files)
    git = manifest["git"]
    print(f"Tài khoản: {account}")
    print(f"Repo: {repo_id} ({'đã có' if exists else 'TẠO MỚI'})  |  tag: {version}")
    print(f"Git commit lúc export: {git['commit']}" + (" (dirty)" if git["dirty"] else ""))
    print(f"Upload {len(upload)} file: {', '.join(upload)}")
    print(f"Xoá {len(delete)} file trên Hub: {', '.join(delete) if delete else '(không có)'}")
    if not confirm:
        print("Chế độ xem trước: chưa push gì. Thêm --confirm để push.")
        return None
    if input(f"Gõ '{repo_id}' để xác nhận push: ").strip() != repo_id:
        print("Không khớp, huỷ push.")
        return None

    if not exists:
        api.create_repo(repo_id, repo_type="model")
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(export_dir / name)) for name in upload
    ] + [CommitOperationDelete(path_in_repo=name) for name in delete]
    commit = api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Release {version}",
        commit_description=f"Export từ git commit {git['commit']}",
    )
    api.create_tag(repo_id, tag=version, revision=commit.oid, tag_message=f"Release {version}")
    record = {"repo_id": repo_id, "version": version, "hub_commit": commit.oid, "account": account}
    with open(export_dir / "push.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"Đã push {repo_id}@{version} (commit {commit.oid}). Nhớ gắn git tag {version} cho commit {git['commit']}.")
    return record
