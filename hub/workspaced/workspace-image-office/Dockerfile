# "office" workspace image: legion-workspace:base + LibreOffice headless +
# Python office libraries, for rooms whose task-assigned agents have an
# executable bundled skill that needs them (e.g. xlsx recalculation).
# internal design notes Stage 5 §7.1 — image selection itself lives
# in hub/backend/app/core/workspace_image.py (resolve_workspace_image).
#
# Build legion-workspace:base first:
#   docker build -t legion-workspace:base hub/workspaced/workspace-image
#   docker build -t legion-workspace:office hub/workspaced/workspace-image-office
FROM legion-workspace:base

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-calc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    openpyxl \
    pandas \
    lxml \
    Pillow

USER dev
