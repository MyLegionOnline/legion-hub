FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    ripgrep \
    fonts-dejavu-core \
    fonts-liberation \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash dev

# Debian's system Python is "externally managed" (PEP 668): a plain
# `pip install X` refuses to run unless the caller passes
# --break-system-packages every single time. That protection exists to stop
# pip from clobbering OS-managed packages on a real, persistent machine —
# meaningless here, since this container IS the sandbox (ephemeral, one per
# room, no OS package manager depends on its Python). Disabling it lets an
# agent's ad hoc `pip install pandas` just work instead of needing to
# discover the flag through trial and error.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Pillow + numpy ship in the base image (not just workspace-image-office) so
# that every room's bundled skills can rely on them without a runtime pip
# install: that install needs network (network_enabled is disabled in some
# rooms) and is best-effort, so a missing skill dependency would otherwise
# fail silently mid-session.
# reportlab/openpyxl + Cyrillic-capable TTF fonts (fonts-dejavu-core,
# fonts-liberation above, under /usr/share/fonts/truetype) are what document
# tasks (PDF/XLSX hand-outs) reach for first. Without them an agent burns a
# dozen tool calls on installs and font hunting — and installs outrun the
# 120 s synchronous bash cap.
RUN pip3 install --no-cache-dir --break-system-packages \
    Pillow \
    numpy \
    reportlab \
    openpyxl

WORKDIR /workspace
RUN chown dev:dev /workspace

USER dev

CMD ["sleep", "infinity"]
