#!/usr/bin/env python3
"""MkDocs hook for building mkslides presentations.

mkslides generates standalone HTML that does not need processing by MkDocs.
Running in on_post_build (after MkDocs has finished) avoids writing into
docs_dir during the build, which would otherwise trigger a redundant rebuild.
"""

import logging
import os
import subprocess
import shutil

log = logging.getLogger("mkdocs.hooks")

SLIDES_DIRS = ("stencil",)

_built = False


def _build_slides(docs_dir, site_dir):
    """Build mkslides presentations into the site output."""
    if not shutil.which("mkslides"):
        log.warning("mkslides not installed, skipping slides")
        return

    for slides_dir in SLIDES_DIRS:
        slides_src = os.path.join(docs_dir, slides_dir)
        slides_dst = os.path.join(site_dir, slides_dir)

        if not os.path.isdir(slides_src):
            continue

        log.info("Building slides from %s into %s", slides_src, slides_dst)
        try:
            result = subprocess.run(
                ["mkslides", "build", slides_src, "-d", slides_dst],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                log.info("Slides built successfully")
            else:
                log.warning(
                    "mkslides exited with %d: %s",
                    result.returncode, result.stderr.strip(),
                )
        except Exception:
            log.exception("mkslides build failed")


def on_post_build(config):
    """Build mkslides output into the MkDocs site directory."""
    global _built
    if _built:
        return
    _built = True
    docs_dir = config.docs_dir if config else "documentation"
    site_dir = config.site_dir if config else "site"
    _build_slides(docs_dir, site_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from types import SimpleNamespace

    config = SimpleNamespace(docs_dir="documentation", site_dir="site")
    on_post_build(config)
