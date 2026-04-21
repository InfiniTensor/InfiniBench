#!/usr/bin/env python3
"""Git utility functions for CI/CD integration."""

import subprocess
import os
from pathlib import Path
from typing import Dict


def get_git_info(project_root: Path = None) -> Dict[str, str]:
    """
    获取当前Git仓库信息。

    Args:
        project_root: 项目根目录，默认为当前文件所在目录的父目录的父目录

    Returns:
        包含Git信息的字典
    """
    if project_root is None:
        # Looks up the project root from the current file
        current = Path(__file__).parent
        while current != current.parent:
            if (current / ".git").exists():
                project_root = current
                break
            current = current.parent

    def run_cmd(cmd):
        try:
            result = (
                subprocess.check_output(
                    cmd,
                    shell=True,
                    cwd=project_root,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )
            return result if result else "unknown"
        except Exception:
            return "unknown"

    # Check to see if it is in the Git repository
    is_git = run_cmd("git rev-parse --git-dir") != "unknown"

    if not is_git:
        return {
            "is_git_repo": False,
            "commit": "not_in_git_repo",
            "short_commit": "not_in_git_repo",
            "branch": "not_in_git_repo",
            "commit_message": "Not in Git repository",
            "commit_author": "unknown",
            "commit_date": "unknown",
        }

    return {
        "is_git_repo": True,
        "commit": run_cmd("git rev-parse HEAD"),
        "short_commit": run_cmd("git rev-parse --short HEAD"),
        "branch": run_cmd("git rev-parse --abbrev-ref HEAD"),
        "commit_message": run_cmd("git log -1 --pretty=%s"),
        "commit_author": run_cmd("git log -1 --pretty=%an"),
        "commit_date": run_cmd("git log -1 --pretty=%ci"),
        "commit_body": run_cmd("git log -1 --pretty=%b"),
    }


def get_ci_environment_info() -> Dict[str, str]:
    """
    Get CI environment information
    """
    ci_info = {"ci_provider": "local"}

    # GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        ci_info.update(
            {
                "ci_provider": "github_actions",
                "ci_run_id": os.environ.get("GITHUB_RUN_ID", ""),
                "ci_run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
                "ci_repository": os.environ.get("GITHUB_REPOSITORY", ""),
                "ci_ref": os.environ.get("GITHUB_REF", ""),
                "ci_sha": os.environ.get("GITHUB_SHA", ""),
            }
        )

    # GitLab CI
    elif os.environ.get("GITLAB_CI") == "true":
        ci_info.update(
            {
                "ci_provider": "gitlab_ci",
                "ci_pipeline_id": os.environ.get("CI_PIPELINE_ID", ""),
                "ci_job_id": os.environ.get("CI_JOB_ID", ""),
                "ci_commit_sha": os.environ.get("CI_COMMIT_SHA", ""),
                "ci_commit_branch": os.environ.get("CI_COMMIT_BRANCH", ""),
            }
        )

    # Jenkins
    elif os.environ.get("JENKINS_URL"):
        ci_info.update(
            {
                "ci_provider": "jenkins",
                "ci_build_id": os.environ.get("BUILD_ID", ""),
                "ci_build_number": os.environ.get("BUILD_NUMBER", ""),
                "ci_job_name": os.environ.get("JOB_NAME", ""),
            }
        )

    return ci_info
